"""Module 4 - packaging: 09_make_report.

Turn reports/*.csv into figures and reports/cellshift_report.md.

Figures: accuracy drop per line (Q1), coverage bars same vs new line (Q2),
risk-coverage curves (Q3), coverage/accuracy vs n labelled (Q4).

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"09_make_report: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
