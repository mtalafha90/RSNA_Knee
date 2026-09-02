"""Score a checkpoint against the 58 expert-labelled studies.

## Why this exists

The training surface is report-derived. So is the validation surface the
trainer selects its epoch on: 548 unseen-scanner studies whose labels come from
radiology reports, not from a radiologist reading the images for this task. The
hidden competition test is scored against **expert** labels.

Those are not the same target, and this project has now measured the gap three
times:

```text
B15    +0.167 teacher agreement    -0.008 expert AUC
B25X   +0.058 weak surface         +0.002 on eleven targets
B50    +0.011 report-derived       -0.002 expert-58   ->  -0.001 hidden
```

The last row is the one that should change how a result is read. B50's
report-derived comparison was well run and passed every clause of a rule frozen
in advance: `+0.011219` on 548 studies, all twelve targets improved, using 37%
of the available headroom. Its expert-58 audit said `-0.002432` and was recorded
as inconclusive, correctly, because 58 studies resolve to about `+/-0.03`.
Carried to full scale as B51, the hidden test returned `-0.001`.

**The 58-study audit, dismissed as inconclusive, pointed the right way and the
548-study report surface did not.** One agreement at n=58 could be coincidence.
Three in a row, in the same direction, is a pattern worth acting on.

## What this does and does not tell you

It cannot confirm a small gain. Fifty-eight studies resolve to roughly `+/-0.03`,
so anything inside that band is silence, not support. What it can do is catch a
report-surface gain that fails to reach expert truth -- which is exactly what
has happened three times, and exactly what the report surface cannot see.

Read it as a veto, never as a confirmation.

## The ceiling, which decides whether the comparison could resolve anything

Two checkpoints that order almost every study pair identically cannot differ in
AUC by much, whatever their mechanisms do. `discordant_pair_fraction` measures
that cap before you look at the delta. B48 and B49 were both judged against a
`+0.010` threshold with ceilings of `0.0015` and `0.0024` -- neither could have
passed whatever it did. Reporting the delta without the ceiling is how that
happens.

## Governance

The 58 gold studies are held out of every run in this project and must stay that
way. This module reads them and never writes a label, a weight or a gradient.
Auditing a checkpoint against them does not make them a validation surface: it
does not select an epoch, a seed, or a setting. Selecting anything on 58 studies
would spend the only expert-truth proxy the project has.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import TARGETS
from .data.coverage import require_dicom_coverage
from .data.dataset import StudyDataset
from .data.series_policy import audit_series_surface
from .data.tables import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv
from .evaluation import macro_auc
from .imaging.crop_policy import validate_crop_focus_policy
from .runtime import autocast

EXPECTED_GOLD_STUDIES = 58
EXPECTED_GOLD_SERIES = 336

# What 58 studies can and cannot resolve. Not a threshold to pass -- a width
# inside which the audit says nothing at all.
EXPERT_RESOLUTION = 0.03

# The reading rule, written here rather than decided after seeing a number.
VETO_DELTA = -0.020
SUPPORT_DELTA = 0.010


def load_expert_surface(config: dict, data_root: str | Path) -> dict:
    """The 58 gold studies, their hard expert labels, and their MRI series."""
    settings = dict(config)
    root = Path(data_root).resolve()
    settings["data_root"] = str(root)

    train = load_train_csv(root / settings.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != EXPECTED_GOLD_STUDIES:
        raise ValueError(
            f"expected {EXPECTED_GOLD_STUDIES} expert studies, found {len(gold)}"
        )
    if gold[TARGETS].isna().any().any():
        raise ValueError("the expert surface has missing labels; it must be complete")

    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    _summary, index = audit_series_surface(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts):
        raise ValueError("an expert study has no MRI series with a recognised plane")
    if int(sum(counts)) != EXPECTED_GOLD_SERIES:
        raise ValueError(
            f"the expert MRI surface changed: {sum(counts)} series, "
            f"expected {EXPECTED_GOLD_SERIES}"
        )
    require_dicom_coverage(root, index, label="expert-58 surface")

    return {
        "settings": settings,
        "root": root,
        "uids": uids,
        "truth": truth,
        "index": index,
        "series_total": int(sum(counts)),
        "metadata_repair": metadata_stats,
    }


@torch.no_grad()
def predict_expert_surface(model, runtime, surface: dict, dataset_config, crop_policy) -> np.ndarray:
    """One probability row per expert study, in the surface's own UID order."""
    validate_crop_focus_policy(crop_policy)
    dataset = StudyDataset(
        surface["uids"],
        surface["index"],
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
    )
    was_training = model.training
    model.eval()
    rows: list[np.ndarray] = []
    try:
        for position in range(len(surface["uids"])):
            item = dataset[position]
            volumes = [v.to(runtime.device, non_blocking=True) for v in item["volumes"]]
            with autocast(runtime):
                out = model(
                    volumes,
                    item["present"].to(runtime.device),
                    item["series_meta"].to(runtime.device),
                    item["slice_position"].to(runtime.device),
                )
            rows.append(
                torch.sigmoid(out.logits.detach().float()).cpu().numpy().reshape(-1)
            )
            del volumes, out, item
    finally:
        model.train(was_training)
    return np.stack(rows)


def expert_scores(truth: np.ndarray, probabilities: np.ndarray) -> dict:
    """Macro AUC on hard expert labels. Every cell is supervised, so weight is 1."""
    weight = np.ones_like(truth, dtype=np.float64)
    return macro_auc(truth, weight, probabilities)


def discordant_pair_fraction(
    truth: np.ndarray, left: np.ndarray, right: np.ndarray
) -> dict:
    """The largest AUC difference these two prediction sets could possibly show.

    AUC counts positive-negative pairs. If two models order a fraction `d` of
    those pairs differently, then `|AUC_left - AUC_right| <= d` for that target,
    exactly. Reported per target and macro-averaged.

    A delta larger than its own ceiling is arithmetically impossible and means a
    bug. A threshold larger than the ceiling means the comparison could not have
    passed whatever the mechanism did -- which is how B48 and B49 were judged
    against `+0.010` with ceilings of `0.0015` and `0.0024`.
    """
    truth = np.asarray(truth, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.shape != truth.shape:
        raise ValueError("truth and both prediction sets must have the same shape")

    per_target: dict[str, float] = {}
    for index, name in enumerate(TARGETS):
        labels = truth[:, index]
        positives = np.flatnonzero(labels > 0.5)
        negatives = np.flatnonzero(labels <= 0.5)
        if positives.size == 0 or negatives.size == 0:
            per_target[name] = float("nan")
            continue
        a = np.sign(left[positives, None, index] - left[None, negatives, index])
        b = np.sign(right[positives, None, index] - right[None, negatives, index])
        # A tie contributes half a pair to AUC, so a tie against a strict
        # ordering is half a discordance rather than none.
        per_target[name] = float(np.abs(a - b).sum() / (2.0 * a.size))

    defined = [value for value in per_target.values() if np.isfinite(value)]
    return {
        "ceiling": float(np.mean(defined)) if defined else float("nan"),
        "per_target_ceiling": per_target,
    }


def read_predictions(path: str | Path, uids: list[str]) -> np.ndarray:
    """Load a saved prediction file, refusing one that is not this surface."""
    frame = pd.read_csv(path)
    required = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != required:
        raise ValueError(f"prediction columns changed: {path}")
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if frame["StudyInstanceUID"].tolist() != list(uids):
        raise ValueError(
            f"{path} is not aligned with this expert surface; the UID order differs"
        )
    return frame[TARGETS].to_numpy(np.float64)


def write_predictions(path: str | Path, uids: list[str], probabilities: np.ndarray) -> Path:
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", list(uids))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def read_the_delta(delta: float, ceiling: float) -> str:
    """The rule, applied. Written before any B53 number existed."""
    if not np.isfinite(delta):
        return "undefined: no target had both classes present"
    if abs(delta) > ceiling + 1e-12:
        return (
            f"IMPOSSIBLE: |delta| {abs(delta):.6f} exceeds the ceiling {ceiling:.6f}. "
            "This is a bug, not a result."
        )
    if delta <= VETO_DELTA:
        return (
            f"VETO: {delta:+.6f} on expert truth. This is B15's failure again -- a "
            "report-surface gain that does not reach expert labels. Do not submit."
        )
    if delta >= SUPPORT_DELTA:
        return (
            f"SUPPORTED on expert truth: {delta:+.6f}, above the {SUPPORT_DELTA:+.3f} "
            "threshold and outside what 58 studies call silence."
        )
    return (
        f"INCONCLUSIVE: {delta:+.6f} sits inside the +/-{EXPERT_RESOLUTION:.2f} that "
        "58 studies can resolve. This is the likely outcome and it is not a "
        "failure; it means the expert surface has no opinion, and the result "
        "rests on the report-derived comparison alone. Record that it does."
    )


def audit(
    config: dict,
    *,
    data_root: str | Path,
    model,
    runtime,
    dataset_config,
    crop_policy,
    out_dir: str | Path,
    label: str,
    against: str | Path | None = None,
) -> dict:
    """Score one checkpoint on the expert surface, optionally against another."""
    surface = load_expert_surface(config, data_root)
    probabilities = predict_expert_surface(model, runtime, surface, dataset_config, crop_policy)
    scores = expert_scores(surface["truth"], probabilities)

    out = Path(out_dir)
    predictions_path = write_predictions(
        out / f"{label}_expert58_predictions.csv", surface["uids"], probabilities
    )

    report = {
        "label": label,
        "studies": len(surface["uids"]),
        "series": surface["series_total"],
        "macro_auc": scores["macro_auc"],
        "per_target_auc": scores["per_target_auc"],
        "targets_defined": scores["targets_defined"],
        "predictions": str(predictions_path),
        "resolution": EXPERT_RESOLUTION,
        "governance": (
            "The 58 expert studies are held out of every run and are read here "
            "only. This audit selects nothing: no epoch, no seed, no setting. "
            "It is a veto on a report-derived gain, never a confirmation of one."
        ),
    }

    if against is not None:
        reference = read_predictions(against, surface["uids"])
        reference_scores = expert_scores(surface["truth"], reference)
        delta = float(scores["macro_auc"] - reference_scores["macro_auc"])
        ceiling = discordant_pair_fraction(surface["truth"], probabilities, reference)
        improved = sum(
            1
            for name in TARGETS
            if np.isfinite(scores["per_target_auc"][name])
            and np.isfinite(reference_scores["per_target_auc"][name])
            and scores["per_target_auc"][name] > reference_scores["per_target_auc"][name]
        )
        report["comparison"] = {
            "against": str(against),
            "reference_macro_auc": reference_scores["macro_auc"],
            "delta": delta,
            "ceiling": ceiling["ceiling"],
            "per_target_ceiling": ceiling["per_target_ceiling"],
            "headroom_used": (
                float(abs(delta) / ceiling["ceiling"])
                if ceiling["ceiling"] > 0
                else float("nan")
            ),
            "targets_improved": improved,
            "targets_compared": len(TARGETS),
            "reading": read_the_delta(delta, ceiling["ceiling"]),
        }

    report_path = out / f"{label}_expert58_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a checkpoint against the 58 expert-labelled studies"
    )
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--data-root", default="artefacts/data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default="artefacts/base_model.pt")
    parser.add_argument("--out-dir", default="artefacts/expert58")
    parser.add_argument("--label", required=True, help="names the written files, e.g. b53")
    parser.add_argument(
        "--against",
        default=None,
        help="a prediction CSV from an earlier audit, to compare against",
    )
    args = parser.parse_args()

    # Imported here so the module can be read and tested without the trainer.
    from .checkpoints import load_base_checkpoint  # noqa: PLC0415
    from .imaging.crop_policy import CROP_FRACTION  # noqa: PLC0415
    from .model.study_model import KneeStudyModel  # noqa: PLC0415
    from .runtime import resolve_runtime  # noqa: PLC0415
    from .training.loop import make_dataset_config, read_config  # noqa: PLC0415

    settings = dict(read_config(args.config))
    runtime = resolve_runtime(settings)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_model, _base_payload = load_base_checkpoint(
        args.base_checkpoint, expected_arm="llm_fill", device="cpu"
    )
    model = KneeStudyModel(
        base_model,
        grid_size=int(payload["model_state"]["grid_size"]),
        top_k=int(payload["model_state"]["top_k"]),
        temperature=float(payload["model_state"]["temperature"]),
        encoder_trainable_stages=int(payload["model_state"]["encoder_trainable_stages"]),
        encoder_chunk_size=int(payload["model_state"]["encoder_chunk_size"]),
        adapt_hierarchy=True,
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(runtime.device)

    root = Path(args.data_root).resolve()
    dataset_config = make_dataset_config(settings, root, train=False)
    dataset_config.tta_center_offsets = ()
    crop_policy = {
        "crop_fraction": float(settings.get("b20_crop_fraction", CROP_FRACTION)),
        "policy": settings.get("b20_crop_policy", "b20_crop_focus_v1"),
    }

    audit(
        settings,
        data_root=root,
        model=model,
        runtime=runtime,
        dataset_config=dataset_config,
        crop_policy=crop_policy,
        out_dir=args.out_dir,
        label=args.label,
        against=args.against,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EXPECTED_GOLD_SERIES",
    "EXPECTED_GOLD_STUDIES",
    "EXPERT_RESOLUTION",
    "audit",
    "discordant_pair_fraction",
    "expert_scores",
    "load_expert_surface",
    "predict_expert_surface",
    "read_predictions",
    "read_the_delta",
    "write_predictions",
]
