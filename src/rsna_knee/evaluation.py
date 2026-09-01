"""Macro ROC AUC over the cells the reports actually supervise."""
from __future__ import annotations

import numpy as np
import torch

from .constants import TARGETS
from .training.losses import move_study_to_device, study_losses
from .training.memory import _trim_host_memory


def masked_binary_targets(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Report states as binary, NaN where nothing supervises the cell.

    The soft targets are 0.85 and 0.05, so the state boundary is 0.5 -- the same
    rule the scanner split was built with. Copied in behaviour from
    `b48_global_conditioned_sparse_eval._masked_target_matrix` so B52's numbers
    sit on the same scale as every other experiment's.
    """
    value = (np.asarray(target, dtype=np.float64) > 0.5).astype(np.float64)
    value[np.asarray(weight, dtype=np.float64) <= 0] = np.nan
    return value


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based binary AUC with ties and NaN masking."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    positives = y_true == 1
    n_pos = int(positives.sum())
    n_neg = int(positives.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(y_score.size, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [sorted_scores.size]))
    for start, stop in zip(starts, stops):
        if stop - start > 1:
            ranks[order[start:stop]] = ranks[order[start:stop]].mean()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(target: np.ndarray, weight: np.ndarray, prediction: np.ndarray) -> dict:
    """Per-target and macro AUC over the cells the reports actually supervise."""
    masked = masked_binary_targets(target, weight)
    per_target = {
        name: float(fast_auc(masked[:, index], prediction[:, index]))
        for index, name in enumerate(TARGETS)
    }
    defined = [value for value in per_target.values() if np.isfinite(value)]
    return {
        "macro_auc": float(np.mean(defined)) if defined else float("nan"),
        "per_target_auc": per_target,
        "targets_defined": len(defined),
    }


@torch.no_grad()
def evaluate_split(model, runtime, loader, multiplier_t, aux_weight: float) -> dict:
    """Score one split without touching gradients or the training mode flag."""
    was_training = model.training
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    losses: list[float] = []

    for items in loader:
        for item in items:
            tensors = move_study_to_device(item, runtime.device)
            out, total, _combined, _local = study_losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            predictions.append(
                torch.sigmoid(out.logits.detach().float()).cpu().numpy().reshape(-1)
            )
            targets.append(item["target"].numpy().reshape(-1))
            weights.append(item["weight"].numpy().reshape(-1))
            losses.append(float(total.detach().item()))
        _trim_host_memory()

    model.train(was_training)
    scores = macro_auc(
        np.stack(targets), np.stack(weights), np.stack(predictions)
    )
    scores["loss"] = float(np.mean(losses)) if losses else float("nan")
    scores["studies"] = len(predictions)
    return scores

