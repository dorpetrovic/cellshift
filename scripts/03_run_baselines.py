"""Module 2 - in-distribution: 03_run_baselines.

Baselines on the unseen-knockdown split, every line.

zero_change, mean_response, kNN and ridge on random features.
Metrics from cellshift.metrics with bootstrap CIs; writes reports/m2_baselines.csv.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"03_run_baselines: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
