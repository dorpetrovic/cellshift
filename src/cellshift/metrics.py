"""Metrics on predicted versus true expression *changes*, with bootstrap intervals.

Arrays are shaped (n_knockdowns, n_genes): one row per test knockdown, values are
log-fold-changes versus the line's non-targeting controls. Every metric is computed
per knockdown and then averaged, so the bootstrap resamples knockdowns.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _top_idx(true_row: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.abs(true_row))[:k]


def pearson_top_de(pred: np.ndarray, true: np.ndarray, k: int = 200) -> np.ndarray:
    """Per-knockdown Pearson r on the k genes that change most in the truth."""
    out = np.empty(len(true))
    for i, (p, t) in enumerate(zip(pred, true)):
        idx = _top_idx(t, k)
        a, b = p[idx] - p[idx].mean(), t[idx] - t[idx].mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        out[i] = (a * b).sum() / denom if denom > 0 else 0.0
    return out


def mae_top_de(pred: np.ndarray, true: np.ndarray, k: int = 200) -> np.ndarray:
    return np.array([np.abs(p[_top_idx(t, k)] - t[_top_idx(t, k)]).mean() for p, t in zip(pred, true)])


def topk_recall(pred: np.ndarray, true: np.ndarray, k: int = 20) -> np.ndarray:
    """Fraction of the true top-k changed genes that are also in the predicted top-k."""
    return np.array([len(set(_top_idx(t, k)) & set(_top_idx(p, k))) / k for p, t in zip(pred, true)])


def coverage(lo: np.ndarray, hi: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-knockdown fraction of genes whose true change lies inside [lo, hi]."""
    return ((true >= lo) & (true <= hi)).mean(axis=1)


def bootstrap_ci(per_item: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile bootstrap (1-alpha) interval over items."""
    per_item = np.asarray(per_item, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_item), size=(n_boot, len(per_item)))
    boots = per_item[idx].mean(axis=1)
    return float(per_item.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean of (a - b) over the same items with a bootstrap interval.

    Use this, not two separate intervals, to claim model A beats model B.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, "paired comparison needs the same items in the same order"
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "pearson_top200": pearson_top_de,
    "mae_top200": mae_top_de,
    "recall_top20": topk_recall,
}
