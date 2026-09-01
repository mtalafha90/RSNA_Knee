"""The training run: what changes, what is held fixed, and how the epoch is chosen."""
from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
import argparse
import json
import math
import numpy as np
import time
import torch
import yaml

from ..checkpoints import load_base_checkpoint, require_base_checkpoint, sha256_file
from ..constants import CONSTRUCTION_SEED_OFFSET, DEFAULT_SEED, LOADER_SEED_OFFSET, TARGETS
from ..data.augmentation import AugmentationPolicy, AugmentedStudyDataset, DEFAULT_SLICE_JITTER, verify_augmentation_reaches_pixels
from ..data.coverage import require_dicom_coverage
from ..data.dataset import StudyDataset, collate_studies
from ..data.series_policy import FROZEN_SERIES_SIGNATURE, audit_series_surface, load_series_policy
from ..data.splits import SPLIT_EXCLUDED, SPLIT_SEEN_SCANNERS, SPLIT_TRAIN, SPLIT_UNSEEN_SCANNERS, load_selection_gate
from ..data.tables import backfill_series_metadata, load_series_csv
from ..evaluation import evaluate_split
from ..model.encoder import MAX_TRAINABLE_STAGES, encoder_state_sha256
from ..model.study_model import KneeStudyModel, require_geometry_contract
from ..runtime import make_scaler, resolve_runtime, seed_everything
from .losses import batch_scales, move_study_to_device, study_losses
from .memory import GRAD_CLIP, HEAD_LR, WEIGHT_DECAY, _format_memory_state, _memory_state, _trim_host_memory
from .supervision import make_dataset_config, target_balance_multipliers
from .surface import build_report_only_surface, config_sha256, fill_artifacts, indices_for_split, uid_sha256


EXPERIMENT = "AUGMENTED_TRAINING"


VERSION = "augmented_training_v1"


DEFAULT_RUN_ROOT = "runs/augmented_training"


CHECKPOINT_NAME = "best_model.pt"


# B53 changes one thing against B52 and inherits every other value from it.
DEFAULT_EPOCHS = 6


# The B52 numbers B53 is measured against, on the same 548 unseen-scanner
# studies. Selection statistics, not effect sizes.
BASELINE_GATE_SPLIT_MACRO_AUC = 0.802666


BASELINE_ALL_DATA_MACRO_AUC = 0.834998


def _build_train_dataset(
    uids, index, dataset_config, crop_policy, targets, weights, *, policy, seed, slice_jitter
):
    return AugmentedStudyDataset(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
        policy=policy,
        seed=seed,
        slice_jitter=slice_jitter,
    )


def _build_valid_dataset(uids, index, dataset_config, crop_policy, targets, weights):
    """Validation is never augmented, so epochs stay comparable with each other."""
    return StudyDataset(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )


def preflight(
    model, runtime, train_dataset, multiplier_t, aux_weight: float, scaler
) -> dict:
    """One forward and backward pass, plus the augmentation check."""
    augmentation = verify_augmentation_reaches_pixels(train_dataset)
    print(
        f"[preflight] augmentation reaches the pixels: "
        f"{augmentation['series_that_changed']}/{augmentation['series_compared']} series "
        f"changed, max |diff| {augmentation['max_absolute_difference']:.6f}",
        flush=True,
    )

    items = [train_dataset[index] for index in range(min(2, len(train_dataset)))]
    for item, scale in zip(items, batch_scales(items, multiplier_t.detach().cpu())):
        tensors = move_study_to_device(item, runtime.device)
        _out, total, _combined, _local = study_losses(
            model, runtime, tensors, multiplier_t, aux_weight
        )
        scaler.scale(total * float(scale)).backward()

    moved = sum(
        1
        for parameter in model.base.encoder.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
    )
    model.zero_grad(set_to_none=True)
    if moved == 0:
        raise RuntimeError("preflight: no gradient reached the encoder")

    print(
        f"[preflight] PASS encoder tensors with gradient={moved} "
        f"{_format_memory_state(_memory_state(runtime))}",
        flush=True,
    )
    return {"encoder_tensors_with_gradient": moved, "augmentation": augmentation}


__all__ = [
    "EXPERIMENT",
    "VERSION",
    "DEFAULT_RUN_ROOT",
    "CHECKPOINT_NAME",
    "DEFAULT_EPOCHS",
    "AugmentationPolicy",
    "augment_series",
    "AugmentedStudyDataset",
    "verify_augmentation_reaches_pixels",
    "preflight",
    "train",
]


DEFAULT_ENCODER_LR_SCALE = 0.10


DEFAULT_ENCODER_STAGES = MAX_TRAINABLE_STAGES


DEFAULT_HIERARCHY_LR_SCALE = 0.05


def parameter_groups(
    model: KneeStudyModel,
    *,
    head_lr: float,
    encoder_lr_scale: float,
    hierarchy_lr_scale: float,
) -> list[dict]:
    """Three rates: fresh head fastest, pretrained encoder and hierarchy slower.

    A pretrained feature is worth more than a randomly initialised head and is
    easily destroyed by the head's step size, so the encoder keeps a reduced
    rate even though far more of it now trains.
    """
    encoder = [p for p in model.base.encoder.parameters() if p.requires_grad]
    hierarchy = [p for p in model.hierarchy_parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]

    if not encoder:
        raise RuntimeError(
            "the encoder must train; none of it requires gradients. Check "
            "encoder_trainable_stages."
        )

    groups = [{"params": head, "lr": float(head_lr), "name": "sparse_head"}]
    groups.append(
        {
            "params": encoder,
            "lr": float(head_lr) * float(encoder_lr_scale),
            "name": "encoder",
        }
    )
    if hierarchy:
        groups.append(
            {
                "params": hierarchy,
                "lr": float(head_lr) * float(hierarchy_lr_scale),
                "name": "study_hierarchy",
            }
        )

    seen: set[int] = set()
    for group in groups:
        for parameter in group["params"]:
            if id(parameter) in seen:
                raise RuntimeError("a parameter reached the optimiser twice")
            seen.add(id(parameter))
    return groups


DEFAULT_TRAIN_SPLIT = SPLIT_TRAIN


# The gate's `train` rows alone are about a third of the report-only population.
# For a competition run every row that is not the validation surface is training
# data: the seen-scanner comparator and the rows B48/B49 spent were withheld to
# keep a *selection* surface clean, and selection here happens only on unseen
# scanners. Nothing that validates B52 is trained on.
ALL_DATA_TRAIN_SPLITS = (SPLIT_TRAIN, SPLIT_SEEN_SCANNERS, SPLIT_EXCLUDED)


VALIDATION_SPLIT = SPLIT_UNSEEN_SCANNERS


def select_train_and_validation(
    all_uids: list, domain_rows, train_splits: tuple = (DEFAULT_TRAIN_SPLIT,)
) -> tuple:
    """Split the report-only surface into training and unseen-scanner validation.

    `indices_for_split` returns a NumPy array and already refuses an empty
    split, so the only check worth adding here is the one it cannot make: that
    no study appears on both sides. A leak there would raise the validation
    score and silently corrupt every checkpoint choice made from it.
    """
    names = tuple(train_splits or (DEFAULT_TRAIN_SPLIT,))
    if VALIDATION_SPLIT in names:
        raise ValueError(
            f"{VALIDATION_SPLIT} is the validation surface and cannot also train"
        )
    train_indices = np.unique(
        np.concatenate(
            [indices_for_split(all_uids, domain_rows, name) for name in names]
        )
    )
    valid_indices = indices_for_split(all_uids, domain_rows, VALIDATION_SPLIT)

    train_uids = [all_uids[int(index)] for index in train_indices]
    valid_uids = [all_uids[int(index)] for index in valid_indices]

    overlap = sorted(set(train_uids) & set(valid_uids))
    if overlap:
        raise RuntimeError(
            f"train and validation splits share {len(overlap)} studies "
            f"(for example {overlap[0]}); selection would be measured on training data"
        )
    return train_indices, valid_indices, train_uids, valid_uids


INHERITED_ENCODER_STAGES = 1


def read_config(path: str | Path) -> dict:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def train(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    encoder_trainable_stages: int = DEFAULT_ENCODER_STAGES,
    encoder_lr_scale: float = DEFAULT_ENCODER_LR_SCALE,
    hierarchy_lr_scale: float = DEFAULT_HIERARCHY_LR_SCALE,
    augment: bool = True,
    slice_jitter: int = DEFAULT_SLICE_JITTER,
    train_splits: tuple = (DEFAULT_TRAIN_SPLIT,),
    gradient_checkpointing: bool = True,
    seed: int = DEFAULT_SEED,
    out_root: str | Path = DEFAULT_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """B52's run with the augmentation actually applied, and nothing else moved."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["seed"] = int(seed)

    if not 1 <= int(encoder_trainable_stages) <= MAX_TRAINABLE_STAGES:
        raise ValueError(f"training needs the encoder; stages must be 1..{MAX_TRAINABLE_STAGES}")
    if int(epochs) < 1:
        raise ValueError("training needs at least one epoch")

    policy = AugmentationPolicy.from_config(settings) if augment else AugmentationPolicy.disabled()
    if augment and policy.is_disabled():
        raise ValueError(
            "this run applies augmentation, but every "
            "configured value is zero. Check b7_rotation_deg and friends."
        )

    domain_payload, domain_rows, domain_meta = load_selection_gate(domain_split)
    seed_everything(int(seed) + CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(
        f"[train] epochs={epochs} encoder_stages={encoder_trainable_stages} "
        f"augment={augment} slice_jitter={slice_jitter}",
        flush=True,
    )
    print(f"[train] augmentation: {policy.active() or 'none'}", flush=True)
    print(f"[train] split sha={domain_meta['sha256']}", flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_base_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(
        root / settings.get("train_csv", "train.csv")
    ) != expected_train_sha:
        raise ValueError("domain split source train.csv fingerprint mismatch")

    fill_artifacts = fill_artifacts(labels_root)
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = build_report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )

    train_indices, valid_indices, train_uids, valid_uids = select_train_and_validation(
        all_uids, domain_rows, tuple(train_splits)
    )
    print(
        f"[train] training on {len(train_uids)} studies from {list(train_splits)}; "
        f"validating on {len(valid_uids)} from {VALIDATION_SPLIT}",
        flush=True,
    )

    train_targets, train_weights = all_targets[train_indices], all_weights[train_indices]
    valid_targets, valid_weights = all_targets[valid_indices], all_weights[valid_indices]
    target_multiplier = target_balance_multipliers(train_weights)

    series_policy = load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != FROZEN_SERIES_SIGNATURE
    ):
        raise ValueError("this run requires the frozen series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    _train_summary, train_index = audit_series_surface(series, train_uids)
    _valid_summary, valid_index = audit_series_surface(series, valid_uids)

    # A study whose DICOM folders are absent yields a zero placeholder for every
    # series, and the model raises "B42 study has no readable MRI series" from
    # inside the first epoch, naming neither the study nor the path. Checking
    # here costs one stat call per series and turns twenty wasted minutes into a
    # message that says which of the two causes it is.
    coverage = {
        "train": require_dicom_coverage(root, train_index, label="training surface"),
        "validation": require_dicom_coverage(root, valid_index, label="validation surface"),
    }
    print(
        f"[train] DICOM coverage: {coverage['train']['series_found']}/"
        f"{coverage['train']['series_total']} training series, "
        f"{coverage['validation']['series_found']}/"
        f"{coverage['validation']['series_total']} validation series",
        flush=True,
    )

    crop_policy = require_geometry_contract(settings)

    # Both configs are built with train=False, and that is deliberate. The flag
    # sets fields this dataset does not read -- which is the whole finding --
    # so B53 leaves it alone rather than pretending it does something, and
    # applies the augmentation where it can actually be observed.
    train_config = make_dataset_config(settings, root, train=False)
    train_config.tta_center_offsets = ()
    valid_config = make_dataset_config(settings, root, train=False)
    valid_config.tta_center_offsets = ()

    train_dataset = _build_train_dataset(
        train_uids, train_index, train_config, crop_policy, train_targets, train_weights,
        policy=policy, seed=int(seed), slice_jitter=int(slice_jitter),
    )
    valid_dataset = _build_valid_dataset(
        valid_uids, valid_index, valid_config, crop_policy, valid_targets, valid_weights
    )

    batch_size = int(settings.get("b42_effective_batch", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_studies,
        **runtime.loader_kwargs(seed=int(seed) + LOADER_SEED_OFFSET),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_studies,
        **runtime.loader_kwargs(seed=int(seed) + LOADER_SEED_OFFSET),
    )

    model = KneeStudyModel(
        base_model,
        grid_size=int(settings["b37_grid_size"]),
        top_k=int(settings["b37_top_k"]),
        temperature=float(settings["b37_temperature"]),
        encoder_trainable_stages=int(encoder_trainable_stages),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        adapt_hierarchy=True,
    ).to(runtime.device)
    model.gradient_checkpointing = bool(gradient_checkpointing)
    model.train()

    trainable = model.trainable_parameter_summary()
    print(f"[train] trainable={trainable}", flush=True)

    head_lr = float(settings.get("b37_head_lr", HEAD_LR))
    groups = parameter_groups(
        model,
        head_lr=head_lr,
        encoder_lr_scale=float(encoder_lr_scale),
        hierarchy_lr_scale=float(hierarchy_lr_scale),
    )
    for group in groups:
        print(
            f"[train]   {group['name']:<16} lr={group['lr']:.3e} "
            f"params={sum(p.numel() for p in group['params']):,}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("b37_weight_decay", WEIGHT_DECAY))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=float(settings.get("b7_min_lr", 1e-6))
    )
    scaler = make_scaler(runtime)
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(settings["b37_local_aux_weight"])
    clip = float(settings.get("b37_grad_clip", GRAD_CLIP))
    clipped = [p for group in groups for p in group["params"]]

    optimizer.zero_grad(set_to_none=True)
    preflight = preflight(model, runtime, train_dataset, multiplier_t, aux_weight, scaler)
    optimizer.zero_grad(set_to_none=True)
    if preflight_only:
        return None

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / CHECKPOINT_NAME
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint_path}")

    history: list[dict] = []
    best_macro = -float("inf")
    best_epoch = 0

    for epoch in range(1, int(epochs) + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        # A different augmentation draw each epoch, reproducible from the seed.
        train_dataset.set_epoch(epoch)
        model.train()
        total_sum = 0.0
        batches = 0

        for items in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scales = batch_scales(items, multiplier_cpu)
            batch_total = 0.0
            for item, scale in zip(items, scales):
                tensors = move_study_to_device(item, runtime.device)
                _out, total, _combined, _local = study_losses(
                    model, runtime, tensors, multiplier_t, aux_weight
                )
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clipped, clip)
            scaler.step(optimizer)
            scaler.update()
            total_sum += batch_total
            batches += 1
            _trim_host_memory()

        scheduler.step()
        scores = evaluate_split(model, runtime, valid_loader, multiplier_t, aux_weight)
        row = {
            "epoch": epoch,
            "train_loss": total_sum / max(batches, 1),
            "validation_loss": scores["loss"],
            "validation_macro_auc": scores["macro_auc"],
            "validation_per_target_auc": scores["per_target_auc"],
            "targets_defined": scores["targets_defined"],
            "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            "epoch_minutes": round((time.monotonic() - started) / 60.0, 1),
        }
        history.append(row)
        print(
            f"[train] E{epoch:>2} train={row['train_loss']:.6f} "
            f"val={row['validation_loss']:.6f} "
            f"macroAUC={row['validation_macro_auc']:.6f} "
            f"({row['epoch_minutes']} min)",
            flush=True,
        )

        if np.isfinite(row["validation_macro_auc"]) and row["validation_macro_auc"] > best_macro:
            best_macro = float(row["validation_macro_auc"])
            best_epoch = epoch
            payload = {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "selected_epoch": epoch,
                "selection_metric": f"macro_auc on {VALIDATION_SPLIT}",
                "selection_value": best_macro,
                "epochs_planned": int(epochs),
                "seed": int(seed),
                "encoder_trainable_stages": int(encoder_trainable_stages),
                "encoder_lr_scale": float(encoder_lr_scale),
                "hierarchy_lr_scale": float(hierarchy_lr_scale),
                "train_splits": list(train_splits),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "head_lr": head_lr,
                # The one change, and the evidence that it happened. B52 wrote
                # augmentation_enabled: true while training on identical pixels
                # every epoch; a boolean nobody measured is what made that
                # possible, so B53 records the measurement instead.
                "augmentation_enabled": bool(augment),
                "augmentation_policy": policy.to_dict(),
                "augmentation_verified": preflight["augmentation"],
                "slice_jitter": int(slice_jitter),
                "changed_from_b52": {
                    "augmentation": [
                        "configured but never applied to the B42 dataset",
                        "applied to every present series, verified at preflight",
                    ],
                },
                "b52_reference": {
                    "gate_split_macro_auc": BASELINE_GATE_SPLIT_MACRO_AUC,
                    "all_data_macro_auc": BASELINE_ALL_DATA_MACRO_AUC,
                    "note": "same 548 unseen-scanner studies; selection statistics",
                },
                "base_checkpoint": str(base_path),
                "base_checkpoint_sha256": sha256_file(base_path),
                "base_state": model.base.state_dict(),
                "head_state": model.head.state_dict(),
                "model_state": model.state(),
                "encoder_sha256_initial": encoder_initial_sha,
                "encoder_sha256_final": encoder_state_sha256(model.base.encoder),
                "training_studies": len(train_uids),
                "validation_studies": len(valid_uids),
                "training_uids_sha256": uid_sha256(train_uids),
                "gold_labels_used": False,
                "gold_studies_used_in_gradient": 0,
                "target_balance_multiplier": {
                    name: float(target_multiplier[index])
                    for index, name in enumerate(TARGETS)
                },
                "label_confidence": confidence,
                "fill_policy": fill_policy,
                "fill_audit": fill_audit,
                "fill_artifacts": fill_artifacts,
                "supervision": supervision,
                "series_policy_signature": FROZEN_SERIES_SIGNATURE,
                "metadata_repair": metadata_stats,
                "dicom_coverage": coverage,
                "domain_split_sha256": domain_meta["sha256"],
                "config_sha256": config_sha256(settings),
                "source_sha256": {"training": sha256_file(Path(__file__))},
                "history": history,
                "governance": (
                    "This run selects its checkpoint on a held-out report-labelled "
                    "split, which is competition practice and deliberately not "
                    "the frozen-endpoint policy the scientific line uses. Its "
                    "validation number is a selection statistic, not evidence of "
                    "an effect, and must not be quoted as one. It is comparable "
                    "with B52's number on the same split and with nothing else."
                ),
            }
            torch.save(payload, checkpoint_path)
            print(f"[train]     new best at epoch {epoch}: {best_macro:.6f}", flush=True)

    if best_epoch == 0:
        raise RuntimeError("the run finished without a usable validation score")

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train] best epoch {best_epoch} macroAUC {best_macro:.6f}", flush=True)
    reference = (
        BASELINE_ALL_DATA_MACRO_AUC
        if tuple(train_splits) == ALL_DATA_TRAIN_SPLITS
        else BASELINE_GATE_SPLIT_MACRO_AUC
    )
    print(
        f"[train] against B52 on the same split: {best_macro - reference:+.6f} "
        f"({best_macro:.6f} vs {reference:.6f})",
        flush=True,
    )
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train the knee MRI model with augmentation applied")
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--encoder-stages", type=int, default=DEFAULT_ENCODER_STAGES,
        help=f"1..{MAX_TRAINABLE_STAGES}; the frozen contract used {INHERITED_ENCODER_STAGES}",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=DEFAULT_ENCODER_LR_SCALE)
    parser.add_argument("--hierarchy-lr-scale", type=float, default=DEFAULT_HIERARCHY_LR_SCALE)
    parser.add_argument(
        "--no-augment", action="store_true",
        help="turn the one change off, reproducing B52's actual behaviour",
    )
    parser.add_argument(
        "--slice-jitter", type=int, default=DEFAULT_SLICE_JITTER,
        help=(
            "shift a series' slice centres by up to N either way. Off by "
            "default: it changes which slices are chosen, which is a second "
            "change and belongs in its own experiment"
        ),
    )
    parser.add_argument(
        "--all-data", action="store_true",
        help=(
            "train on every split except the unseen-scanner validation surface "
            f"({', '.join(ALL_DATA_TRAIN_SPLITS)}) instead of the gate's train rows alone"
        ),
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="faster, uses more GPU memory; identical maths",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    train(
        read_config(args.config),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        epochs=args.epochs,
        encoder_trainable_stages=args.encoder_stages,
        encoder_lr_scale=args.encoder_lr_scale,
        hierarchy_lr_scale=args.hierarchy_lr_scale,
        augment=not args.no_augment,
        slice_jitter=args.slice_jitter,
        train_splits=ALL_DATA_TRAIN_SPLITS if args.all_data else (DEFAULT_TRAIN_SPLIT,),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        out_root=args.out_root,
        preflight_only=args.preflight_only,
    )

