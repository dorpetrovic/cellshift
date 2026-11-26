"""Module 2 - in-distribution: 04_train_models.

Ridge, kNN and an MLP ensemble (5 seeds) on ESM-2 features, unseen-knockdown split.

Pick hyperparameters on a validation fold inside train only.
Writes reports/m2_models.csv and saves predictions (incl. each ensemble member) for module 3.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"04_train_models: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
