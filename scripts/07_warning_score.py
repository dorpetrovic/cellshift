"""Module 3 - Q3: 07_warning_score.

Can a label-free score warn us before the errors happen?

Scores: shift_eval.knn_distance (embedding distance to training genes) and
shift_eval.ensemble_disagreement. Validate against the true per-knockdown error with
shift_eval.warning_quality and risk_coverage. Writes reports/q3_warning.csv.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"07_warning_score: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
