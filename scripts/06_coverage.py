"""Module 3 - Q2: 06_coverage.

Do 90% intervals still cover 90% in a new line?

Intervals: split conformal (conformal.calibrate/intervals) and ensemble mean +- z*std.
Calibrate on same-line held-out rows, test on (a) same line, (b) new line.
shift_eval.coverage_report for each. Writes reports/q2_coverage.csv.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"06_coverage: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
