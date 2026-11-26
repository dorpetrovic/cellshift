"""Loading the four CRISPRi screens and turning cells into pseudobulk changes.

Heavy dependencies (anndata, scanpy, pertpy) are imported inside functions so the
core library and its tests stay light.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CONTROL_LABELS = ("non-targeting", "control", "non_targeting", "NT")


def pseudobulk_lfc(X, perturbation, control_labels=CONTROL_LABELS, pseudocount: float = 1.0, min_cells: int = 20):
    """Per-knockdown log-fold-change of mean normalized expression versus controls.

    X: (n_cells, n_genes) normalized counts (e.g. counts per 10k), dense or scipy sparse.
    perturbation: per-cell knockdown label.
    Returns (knockdowns, lfc) where lfc is (n_knockdowns, n_genes).
    """
    perturbation = np.asarray(perturbation, dtype=object)
    is_ctrl = np.isin(perturbation, control_labels)
    if not is_ctrl.any():
        raise ValueError("no control cells found; check control_labels")
    ctrl_mean = np.asarray(X[is_ctrl].mean(axis=0)).ravel()
    kds, rows = [], []
    for k in sorted(set(perturbation[~is_ctrl])):
        m = perturbation == k
        if m.sum() < min_cells:
            continue
        mean_k = np.asarray(X[m].mean(axis=0)).ravel()
        kds.append(k)
        rows.append(np.log2(mean_k + pseudocount) - np.log2(ctrl_mean + pseudocount))
    return np.array(kds, dtype=object), np.vstack(rows)


def load_replogle(line: str, cache_dir: str | Path = "data/raw"):
    """K562 essential or RPE1 screen via pertpy. Returns an AnnData."""
    import pertpy as pt  # noqa: WPS433

    loaders = {"K562": pt.data.replogle_2022_k562_essential, "RPE1": pt.data.replogle_2022_rpe1}
    return loaders[line]()


def load_nadig(line: str, cache_dir: str | Path = "data/raw"):
    """HepG2 or Jurkat screen from GEO GSE264667.

    TODO (week 1): confirm file names and cell lines on GEO, download with checksums,
    and map the guide/perturbation column to the same labels as the Replogle data.
    """
    raise NotImplementedError("week 1 task: see docstring")
