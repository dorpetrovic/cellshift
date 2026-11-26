"""Baselines that every model must be compared against.

Each baseline is fit on training rows and predicts a (n_test, n_genes) matrix of
expression changes. ``lines``/``knockdowns`` are per-row labels; ``Y`` holds the
true changes for training rows.
"""
from __future__ import annotations

import numpy as np


def zero_change(n_test: int, n_genes: int) -> np.ndarray:
    """Predict no change at all (the control mean)."""
    return np.zeros((n_test, n_genes))


def mean_response(Y_train: np.ndarray, n_test: int) -> np.ndarray:
    """Predict the average change across all training knockdowns."""
    return np.repeat(Y_train.mean(axis=0, keepdims=True), n_test, axis=0)


def same_knockdown_other_lines(Y_train, kd_train, kd_test, fallback: np.ndarray | None = None) -> np.ndarray:
    """Average of the same knockdown's response in the training lines (split 2's strong baseline)."""
    kd_train = np.asarray(kd_train, dtype=object)
    fb = Y_train.mean(axis=0) if fallback is None else fallback
    out = []
    for k in kd_test:
        m = kd_train == k
        out.append(Y_train[m].mean(axis=0) if m.any() else fb)
    return np.vstack(out)


class Ridge:
    """Closed-form ridge from features X to responses Y (multi-output)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray) -> Ridge:
        self.x_mean, self.y_mean = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.x_mean, Y - self.y_mean
        d = Xc.shape[1]
        self.W = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ Yc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) @ self.W + self.y_mean


def nearest_neighbour(X_train, Y_train, X_test, k: int = 5) -> np.ndarray:
    """Average response of the k training knockdowns closest in feature space (cosine)."""
    def _norm(a):
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    sims = _norm(X_test) @ _norm(X_train).T
    nn = np.argsort(-sims, axis=1)[:, :k]
    return Y_train[nn].mean(axis=1)
