"""Module 3 - Q1: 05_accuracy_drop.

How much worse is each model on a cell line it never saw?

For each held-out line L (leave-one-cell-line-out): train on the other three, test on L
(splits.unseen_line and unseen_both). Compare with the same-line score from module 2 using
cellshift.shift_eval.accuracy_drop. Writes reports/q1_accuracy_drop.csv.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"05_accuracy_drop: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
