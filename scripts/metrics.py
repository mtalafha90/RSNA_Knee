"""Macro-AUC with a bootstrap confidence interval.

The competition scores macro-averaged AUC-ROC, so that is what every candidate
label set and every model should be judged by here. The confidence interval
matters more than usual: the gold set has 58 studies, and a point estimate from
58 studies can easily move by more than the difference between two genuinely
different extractors.

AUC is computed from ranks (the Mann-Whitney identity) rather than through
scikit-learn, because the bootstrap calls it tens of thousands of times.
"""

from __future__ import annotations

import numpy as np


def auc_from_ranks(truth: np.ndarray, score: np.ndarray) -> float:
    """AUC for one label. Returns NaN when the label has no positives or no negatives.

    Ties are handled by averaging ranks, which is what gives a binary 0/1 score
    exactly the mean of its sensitivity and specificity.
    """
    positives = truth == 1
    n_pos = int(positives.sum())
    n_neg = int(len(truth) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)

    # Average the ranks within each group of equal scores, so ties count as half.
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(sorted_scores) + 1):
        if end == len(sorted_scores) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end

    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Mean AUC across labels, skipping any label that has only one class present."""
    per_label = [auc_from_ranks(truth[:, k], score[:, k]) for k in range(truth.shape[1])]
    usable = [value for value in per_label if not np.isnan(value)]
    return float(np.mean(usable)) if usable else float("nan")


def bootstrap_macro_auc(
    truth: np.ndarray,
    score: np.ndarray,
    resamples: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate and 95% interval, resampling studies rather than cells.

    Studies are the independent unit: the twelve labels of one study are
    correlated, so resampling cells would understate the uncertainty.
    """
    rng = np.random.default_rng(seed)
    n = len(truth)
    draws = []
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        value = macro_auc(truth[index], score[index])
        if not np.isnan(value):
            draws.append(value)

    draws = np.array(draws)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "macro_auc": macro_auc(truth, score),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_width": float(high - low),
        "above_chance": float((draws > 0.5).mean()),
    }
