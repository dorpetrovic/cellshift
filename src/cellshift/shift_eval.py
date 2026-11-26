"""Module 3: can the model be trusted under shift? Reusable, domain-agnostic tools.

Nothing in here knows about cells. Inputs are plain arrays:
- per-item scores (one number per test item, e.g. Pearson per knockdown),
- predictions / truths shaped (n_items, n_outputs),
- features shaped (n_items, d).
That is on purpose: the ESM-2 / ProteinGym calibration report reuses this file
with "assay" in place of "cell line" and "variant" in place of "knockdown".

The four questions, and the functions that answer them:
Q1 accuracy drop ............ ``accuracy_drop``
Q2 interval coverage ........ ``coverage_report`` (+ ``cellshift.conformal``)
Q3 label-free warning score . ``knn_distance``, ``ensemble_disagreement``,
                              ``warning_quality``, ``risk_coverage``
Q4 cheap fixes .............. ``few_shot_affine``, ``few_shot_indices``,
                              ``weighted_conformal_quantile``
"""
from __future__ import annotations

import numpy as np

from cellshift import metrics

# ---------- Q1: how much worse on a new domain? ----------

def accuracy_drop(score_in, score_out, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """mean(score_in) - mean(score_out) with a bootstrap interval.

    The two sets hold *different* items (same-line test vs new-line test), so each is
    resampled on its own (unpaired). Positive = the model got worse on the new domain.
    Tiny example: in = [0.6, 0.5, 0.7], out = [0.3, 0.4, 0.2] -> drop = 0.6 - 0.3 = 0.3.
    """
    a, b = np.asarray(score_in, float), np.asarray(score_out, float)
    rng = np.random.default_rng(seed)
    boots = (a[rng.integers(0, len(a), (n_boot, len(a)))].mean(1)
             - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1))
    return float(a.mean() - b.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ---------- Q2: do the intervals still cover? ----------

def coverage_report(lo, hi, y, target: float = 0.9, n_boot: int = 2000, seed: int = 0) -> dict:
    """Coverage of [lo, hi] with a bootstrap interval over items, plus the gap to target.

    gap < 0 means under-coverage (intervals too narrow: overconfident).
    """
    per_item = metrics.coverage(lo, hi, y)
    m, l, h = metrics.bootstrap_ci(per_item, n_boot=n_boot, seed=seed)
    width = float(np.mean(np.asarray(hi) - np.asarray(lo)))
    return {"coverage": m, "ci": (l, h), "gap": m - target, "mean_width": width}


# ---------- Q3: can we tell, without labels, when to worry? ----------

def knn_distance(X_train, X_test, k: int = 5) -> np.ndarray:
    """Per test item: mean cosine distance to its k nearest training items. High = unfamiliar."""
    def _norm(a):
        a = np.asarray(a, float)
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    d = 1.0 - _norm(X_test) @ _norm(X_train).T
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


def ensemble_disagreement(member_preds) -> np.ndarray:
    """member_preds: (n_members, n_items, n_outputs). Per item: mean std across members."""
    return np.asarray(member_preds, float).std(axis=0).mean(axis=1)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def warning_quality(score, error, top_frac: float = 0.25) -> dict:
    """How well a label-free score predicts the real error.

    spearman: rank correlation of score with error (1 = perfect ordering).
    auroc: chance a "bad" item (error in the worst ``top_frac``) gets a higher score than a
           "fine" item. 0.5 = useless, 1.0 = perfect.
    """
    score, error = np.asarray(score, float), np.asarray(error, float)
    rs, re = _ranks(score), _ranks(error)
    rs_c, re_c = rs - rs.mean(), re - re.mean()
    spearman = float((rs_c * re_c).sum() / np.sqrt((rs_c ** 2).sum() * (re_c ** 2).sum()))
    bad = error >= np.quantile(error, 1 - top_frac)
    n_bad, n_ok = bad.sum(), (~bad).sum()
    auroc = float((rs[bad].sum() - n_bad * (n_bad + 1) / 2) / (n_bad * n_ok)) if n_bad and n_ok else float("nan")
    return {"spearman": spearman, "auroc": auroc}


def risk_coverage(score, error, fractions=None):
    """If we only trust the fraction f of items with the lowest warning score, what is their
    mean error? A useful score gives a curve that rises with f."""
    fractions = np.linspace(0.1, 1.0, 10) if fractions is None else np.asarray(fractions)
    order = np.argsort(np.asarray(score, float))
    err = np.asarray(error, float)[order]
    n = len(err)
    return fractions, np.array([err[: max(1, round(f * n))].mean() for f in fractions])


# ---------- Q4: cheap fixes ----------

def few_shot_indices(n_items: int, n_shot: int, seed: int = 0):
    """Pick ``n_shot`` labelled items from the new domain; the rest stay as test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    return np.sort(perm[:n_shot]), np.sort(perm[n_shot:])


def few_shot_affine(y_few, pred_few, pred_rest):
    """Fit y = a * pred + b on the few labelled items (all outputs pooled), apply to the rest.

    Fixes the common shift where the new domain's responses are uniformly bigger/smaller.
    Returns (corrected_pred_rest, a, b).
    """
    p, t = np.ravel(pred_few).astype(float), np.ravel(y_few).astype(float)
    pc = p - p.mean()
    a = float((pc * (t - t.mean())).sum() / max((pc ** 2).sum(), 1e-12))
    b = float(t.mean() - a * p.mean())
    return a * np.asarray(pred_rest, float) + b, a, b


def weighted_conformal_quantile(scores_cal, w_cal, w_test: float, alpha: float = 0.1) -> float:
    """Weighted split-conformal quantile (Tibshirani et al. 2019, covariate shift).

    w = p_test(x) / p_cal(x), e.g. from a classifier trained to tell calibration items from
    test items. Calibration items that look like the test domain count more.
    With all weights equal this is exactly ``conformal.conformal_quantile``.
    """
    s, w = np.ravel(scores_cal).astype(float), np.ravel(w_cal).astype(float)
    order = np.argsort(s)
    s, w = s[order], w[order]
    p = w / (w.sum() + w_test)
    cum = np.cumsum(p)
    idx = np.searchsorted(cum, 1 - alpha - 1e-12)
    return float(np.inf) if idx >= len(s) else float(s[idx])
