"""Module 1 - data: 01_build_pseudobulk.

Build y: one row per (cell line, knockdown), one column per measured gene.

Tasks
1. Load each screen (cellshift.data.load_replogle / load_nadig).
2. Normalize to counts per 10k, then pseudobulk_lfc() per line.
3. On-target QC: keep knockdowns whose own gene dropped (see week1_k562.on_target_ok).
4. Keep knockdowns and genes present in all four lines; harmonize gene symbols.
Writes data/processed/{line}_lfc.npy, {line}_knockdowns.npy, genes.npy, and a QC table
(cells per knockdown, knockdowns kept per line) to reports/qc_counts.csv.
NOTE: the 2000 "most variable" genes are picked later, on training rows only.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"01_build_pseudobulk: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
