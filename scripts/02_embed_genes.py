"""Module 1 - features: 02_embed_genes.

Build X: one vector per knocked-down gene.

Tasks
1. Union of knockdowns across lines -> cellshift.features.fetch_sequences (cache to data/raw/uniprot.fasta).
2. cellshift.features.esm2_embed for the 8M model first; 35M/150M/650M only if time allows.
3. cellshift.features.random_features with the same width (control).
Writes data/processed/X_esm2_{size}.npz and X_random.npz (keys = gene symbols).

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"02_embed_genes: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
