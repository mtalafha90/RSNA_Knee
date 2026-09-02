"""Producing submission.csv from a checkpoint this package trained.

Runs on however many GPUs are visible. Two shards the test set by row index and
halves the wall clock; one is slower but works, which is what makes a smoke test
on a local card possible at all. The archive's launcher refuses to start below
two devices, so its submission path could only ever be exercised on Kaggle --
and it failed there three times on things the three visible studies could not
show.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from ..checkpoints import load_base_checkpoint, sha256_file
from ..constants import TARGETS
from ..imaging.crop_policy import CROP_FRACTION
from ..model.study_model import KneeStudyModel
from ..runtime import resolve_runtime
from .runner import ON_UNREADABLE_FALLBACK, ON_UNREADABLE_MODES, FALLBACK_PROBABILITY, infer_shard
from .test_surface import build_test_surface

SUBMISSION_VERSION = "rsna_knee_streamed_views_noabort_v1"
TTA_OFFSETS = (-1, 0, 1)
REQUIRED_ENCODER_CHUNK = 4


def load_checkpoint(path, *, base_checkpoint, device, expected_sha256: str | None = None):
    """Rebuild the trained model, refusing to guess any of its geometry."""
    checkpoint = Path(path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    observed = sha256_file(checkpoint)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            "this run requires the declared checkpoint: "
            f"expected {expected_sha256}, got {observed}"
        )

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model_state") or {}
    for key in ("grid_size", "top_k", "temperature", "encoder_chunk_size"):
        if key not in state:
            raise ValueError(
                f"checkpoint model_state has no {key}; the geometry would be "
                "guessed, and top_k and temperature are not weights so a wrong "
                "value survives a strict load and changes every prediction"
            )
    if int(state["encoder_chunk_size"]) != REQUIRED_ENCODER_CHUNK:
        raise ValueError(
            f"trained encoder chunk is {state['encoder_chunk_size']}, not "
            f"{REQUIRED_ENCODER_CHUNK}; the runtime budget was calibrated on {REQUIRED_ENCODER_CHUNK}"
        )
    if bool(payload.get("gold_labels_used", True)):
        raise ValueError("this checkpoint used expert labels and must not be submitted")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("this checkpoint took expert gradients and must not be submitted")

    base_model, base_payload = load_base_checkpoint(
        base_checkpoint, expected_arm="llm_fill", device="cpu"
    )
    if sha256_file(Path(base_checkpoint)) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("base checkpoint fingerprint does not match the one this run trained from")

    model = KneeStudyModel(
        base_model,
        grid_size=int(state["grid_size"]),
        top_k=int(state["top_k"]),
        temperature=float(state["temperature"]),
        encoder_trainable_stages=int(state.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(state["encoder_chunk_size"]),
        adapt_hierarchy=True,
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload, base_payload, observed


def generate_submission(
    config: dict,
    *,
    data_root,
    checkpoint,
    base_checkpoint,
    out_path="submission.csv",
    expected_checkpoint_sha256: str | None = None,
    on_unreadable: str = ON_UNREADABLE_FALLBACK,
    fallback_probability: float = FALLBACK_PROBABILITY,
    max_hours: float = 8.25,
    reserve_minutes: float = 30.0,
    limit: int | None = None,
    allow_single_gpu: bool = False,
) -> Path:
    """Write submission.csv and a manifest recording exactly what produced it."""
    if on_unreadable not in ON_UNREADABLE_MODES:
        raise ValueError(f"on_unreadable must be one of {ON_UNREADABLE_MODES}")

    runtime = resolve_runtime(dict(config))
    devices = (
        [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [torch.device("cpu")]
    )
    print(
        f"[submit] {len(devices)} device(s): {[str(d) for d in devices]}; "
        f"TTA offsets={list(TTA_OFFSETS)}; one view at a time; "
        f"runtime guard=telemetry only; on_unreadable={on_unreadable}",
        flush=True,
    )
    # A full hidden run on one T4 is roughly nine hours against a nine-hour
    # ceiling, so it does not finish. Two T4s shard it to about four. The
    # archive's launcher refuses below two for exactly that reason, and this one
    # keeps the refusal -- but makes it waivable, because a submission path that
    # can only be exercised on Kaggle is one whose defects are found there, at a
    # submission slot each. --limit is a smoke test and waives it on its own.
    if len(devices) < 2 and not (allow_single_gpu or limit is not None):
        raise RuntimeError(
            f"a full run needs two GPUs; {len(devices)} visible. On two T4s the "
            "hidden set takes about four hours, on one about nine against a "
            "nine-hour ceiling. Pass --limit N for a smoke test on this card, or "
            "--allow-single-gpu if you mean it."
        )
    if len(devices) < 2:
        print(
            f"[submit] {len(devices)} device(s), which is a smoke test rather than "
            "a submission: a full hidden run needs two.",
            flush=True,
        )

    surface = build_test_surface(config, data_root)
    uids = surface["uids"]
    if limit is not None:
        uids = uids[: int(limit)]
        print(f"[submit] limited to the first {len(uids)} studies", flush=True)
    if surface["studies_without_series"]:
        print(
            f"[submit] {len(surface['studies_without_series'])} study/studies have "
            "no series with a recognised plane",
            flush=True,
        )

    models = []
    payload = base_payload = None
    checkpoint_sha = ""
    for device in devices:
        model, payload, base_payload, checkpoint_sha = load_checkpoint(
            checkpoint, base_checkpoint=base_checkpoint, device=device,
            expected_sha256=expected_checkpoint_sha256,
        )
        models.append(model)
    print(
        f"[submit] checkpoint {checkpoint_sha}; epoch "
        f"{payload.get('selected_epoch')} selected at {payload.get('selection_value')}, "
        f"trained on {payload.get('training_studies')} studies",
        flush=True,
    )

    settings = surface["settings"]
    crop_fraction = float(settings.get("b20_crop_fraction", CROP_FRACTION))
    gap = int(settings.get("b7_triplet_gap", 1))
    shards = [list(range(rank, len(uids), len(devices))) for rank in range(len(devices))]
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(
                infer_shard,
                rank=rank, indices=shards[rank], uids=uids, index=surface["index"],
                model=models[rank], data_root=surface["root"], split="test",
                tta_offsets=TTA_OFFSETS, gap=gap, crop_fraction=crop_fraction,
                device=devices[rank], on_unreadable=on_unreadable,
                fallback_probability=fallback_probability,
                max_hours=max_hours, reserve_minutes=reserve_minutes,
                global_started=started,
                physical_scale_policy=settings.get("physical_scale_policy"),
            )
            for rank in range(len(devices))
        ]
        rows: list[tuple] = []
        failures: list[dict] = []
        for future in futures:
            shard_rows, shard_failures = future.result()
            rows.extend(shard_rows)
            failures.extend(shard_failures)

    rows.sort(key=lambda row: row[0])
    failures.sort(key=lambda record: record["index"])
    if [row[0] for row in rows] != list(range(len(uids))):
        raise RuntimeError("output row indices are incomplete or duplicated")
    if [row[1] for row in rows] != uids:
        raise RuntimeError("the study order changed between input and output")

    frame = pd.DataFrame(np.stack([row[2] for row in rows]), columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", uids)
    if frame.isna().any().any():
        raise RuntimeError("submission contains blanks")
    values = frame[TARGETS].to_numpy()
    if values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError("submission contains probabilities outside [0, 1]")

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    shape_rows = [shape for row in rows for shape in row[3]]
    heights = np.asarray([s[0] for s in shape_rows] or [0], np.int64)
    widths = np.asarray([s[1] for s in shape_rows] or [0], np.int64)
    manifest = {
        "version": SUBMISSION_VERSION,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "base_checkpoint_sha256": sha256_file(Path(base_checkpoint)),
        "selected_epoch": payload.get("selected_epoch"),
        "selection_metric": payload.get("selection_metric"),
        "selection_value": payload.get("selection_value"),
        "training_studies": payload.get("training_studies"),
        "train_splits": payload.get("train_splits"),
        "augmentation_enabled": payload.get("augmentation_enabled"),
        "fill_policy": payload.get("fill_policy"),
        "fixed_endpoint": False,
        "prediction": "combined sparse-MIL logits; raw sigmoid probability",
        "thresholding_used": False,
        "blending_used": False,
        "tta_center_offsets": list(TTA_OFFSETS),
        "tta_aggregation": "mean of per-view sigmoid probabilities",
        "devices": [str(d) for d in devices],
        "full_run": len(devices) >= 2 and limit is None,
        "study_sharding": f"test-row index modulo {len(devices)}",
        "execution": {
            "tta_materialization": "one complete study view at a time",
            "native_volume_normalizations_per_series": 1,
            "runtime_projection": "telemetry_only_no_exception",
            "host_trim_after_each_study": True,
            "encoder_chunk_size": REQUIRED_ENCODER_CHUNK,
        },
        "on_unreadable": on_unreadable,
        "fallback_probability": float(fallback_probability),
        "studies_predicted_from_fallback": len(failures),
        "studies_predicted_from_fallback_fraction": len(failures) / len(uids),
        "fallback_studies": failures[:50],
        "studies_without_series": len(surface["studies_without_series"]),
        "test_rows": int(len(frame)),
        "test_series_total": int(sum(surface["counts"])),
        "coverage": surface["coverage"],
        "metadata_repair": surface["metadata_repair"],
        "geometry": {
            "n_series": len(shape_rows),
            "rectangular_series": int(np.sum(heights != widths)),
            "height_median": float(np.median(heights)),
            "width_median": float(np.median(widths)),
        },
        "runtime": runtime.describe(),
        "runtime_elapsed_hours": (time.monotonic() - started) / 3600.0,
        "submission_sha256": sha256_file(output),
        "governance": (
            "This checkpoint selected its epoch on a held-out report-labelled "
            "split. Its validation number is a selection statistic, not evidence "
            "of an effect, and must not be quoted as one."
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(output, flush=True)
    print(manifest_path, flush=True)
    if failures:
        print(
            f"[submit] WARNING {len(failures)}/{len(uids)} studies "
            f"({100.0 * len(failures) / len(uids):.2f}%) were predicted at "
            f"{fallback_probability} because they could not be read",
            flush=True,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a submission from a trained checkpoint")
    parser.add_argument("--config", default="config/training.yaml")
    parser.add_argument("--data-root", default="artefacts/data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default="artefacts/base_model.pt")
    parser.add_argument("--out-path", default="submission.csv")
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--on-unreadable", choices=ON_UNREADABLE_MODES, default=ON_UNREADABLE_FALLBACK)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="score only the first N studies, for a smoke test on a local card",
    )
    parser.add_argument(
        "--allow-single-gpu", action="store_true",
        help=(
            "run the whole test set on one GPU. A hidden run then takes about "
            "nine hours against a nine-hour ceiling, so this is for local use"
        ),
    )
    args = parser.parse_args()

    from ..training.loop import read_config  # noqa: PLC0415

    generate_submission(
        read_config(args.config),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_path=args.out_path,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        on_unreadable=args.on_unreadable,
        limit=args.limit,
        allow_single_gpu=args.allow_single_gpu,
    )


if __name__ == "__main__":
    main()


__all__ = ["SUBMISSION_VERSION", "TTA_OFFSETS", "generate_submission", "load_checkpoint"]
