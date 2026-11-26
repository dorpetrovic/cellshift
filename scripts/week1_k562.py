"""Module 1, first task: one cell line (K562), end to end, as small as possible.

This is a smoke test that touches every later module once (data, features, a split,
baselines, metrics) so problems show up in week 1, not week 6.

Steps: load -> normalize -> pseudobulk change per knockdown -> pick genes ->
ESM-2 (8M) protein embedding per knocked-down gene -> unseen-knockdown split ->
two baselines + ridge on ESM-2 -> metrics with bootstrap intervals.

Run:  python scripts/week1_k562.py
Needs: pip install -e ".[data,models,dev]"   (about 16 GB RAM; otherwise use Colab)
Not yet run on the real data: expect to fix a column name or two on first run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cellshift import baselines, metrics, splits
from cellshift.data import pseudobulk_lfc
from cellshift.features import esm2_embed, fetch_sequences

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def load_k562():
    import pertpy as pt
    import scanpy as sc

    adata = pt.data.replogle_2022_k562_essential()          # downloads once, then cached
    print(adata)                                             # cells x genes; look at obs columns
    print(adata.obs["perturbation"].value_counts().head())   # 'control' = non-targeting guides
    sc.pp.normalize_total(adata, target_sum=1e4)             # counts per 10k: removes depth differences
    return adata


def on_target_ok(adata, kds, lfc, gene_names, max_lfc=-0.5):
    """QC: keep knockdowns where the targeted gene's own expression actually dropped."""
    idx = {g: i for i, g in enumerate(gene_names)}
    keep = np.array([k in idx and lfc[j, idx[k]] < max_lfc for j, k in enumerate(kds)])
    print(f"on-target QC: kept {keep.sum()} of {len(kds)} knockdowns")
    return keep


def pick_genes(lfc, n=2000):
    """The n genes whose response varies most across knockdowns."""
    return np.argsort(-lfc.var(axis=0))[:n]


def main():
    adata = load_k562()
    kds, lfc = pseudobulk_lfc(adata.X, adata.obs["perturbation"].values)
    gene_names = np.asarray(adata.var_names)
    keep = on_target_ok(adata, kds, lfc, gene_names)
    kds, lfc = kds[keep], lfc[keep]
    cols = pick_genes(lfc)           # NOTE: in the real splits, pick genes on training rows only
    Y = lfc[:, cols]
    np.save(OUT / "k562_lfc.npy", Y)
    np.save(OUT / "k562_knockdowns.npy", kds.astype(str))

    emb = esm2_embed(fetch_sequences(kds))
    has = np.array([k in emb for k in kds])
    kds, Y = kds[has], Y[has]
    X = np.vstack([emb[k] for k in kds])
    np.save(OUT / "k562_esm2_8M.npy", X)

    lines = np.array(["K562"] * len(kds), dtype=object)
    groups = kds  # week 1: one group per knockdown; response clusters come in module 2
    s = splits.unseen_knockdown(lines, kds, groups, line="K562")
    splits.assert_no_leakage(s, lines, kds, "unseen_knockdown")

    Ytr, Yte = Y[s.train], Y[s.test]
    preds = {
        "zero_change": baselines.zero_change(len(Yte), Y.shape[1]),
        "mean_response": baselines.mean_response(Ytr, len(Yte)),
        "ridge_esm2_8M": baselines.Ridge(alpha=10).fit(X[s.train], Ytr).predict(X[s.test]),
    }
    results = {}
    for name, P in preds.items():
        results[name] = {m: metrics.bootstrap_ci(f(P, Yte)) for m, f in metrics.METRICS.items()}
        print(name, {m: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for m, v in results[name].items()})
    d = metrics.paired_bootstrap_diff(metrics.pearson_top_de(preds["ridge_esm2_8M"], Yte),
                                      metrics.pearson_top_de(preds["mean_response"], Yte))
    print(f"ridge minus mean_response, Pearson: {d[0]:.3f} [{d[1]:.3f}, {d[2]:.3f}]")
    (OUT / "week1_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
