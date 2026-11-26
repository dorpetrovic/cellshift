"""Split conformal intervals for knockdown-effect predictions.

Nonconformity score: |y - y_hat| / sigma_hat, where sigma_hat is a per-entry scale
(e.g. deep-ensemble std). With sigma_hat = 1 this is plain absolute-residual conformal.

The project's core comparison is *where the calibration rows come from*:
- standard: held-out knockdowns from the same lines as training
- leave-one-line-out: rows from a training line that was held out of fitting, which
  mimics the shift the test line will bring.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """Finite-sample-corrected (1-alpha) quantile of calibration scores."""
    s = np.sort(np.ravel(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if k > n else float(s[k - 1])


def calibrate(y_cal, pred_cal, sigma_cal=None, alpha: float = 0.1) -> float:
    sigma_cal = np.ones_like(pred_cal) if sigma_cal is None else np.clip(sigma_cal, 1e-8, None)
    return conformal_quantile(np.abs(y_cal - pred_cal) / sigma_cal, alpha)


def intervals(pred, q: float, sigma=None):
    sigma = np.ones_like(pred) if sigma is None else np.clip(sigma, 1e-8, None)
    return pred - q * sigma, pred + q * sigma
