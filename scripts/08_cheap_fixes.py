"""Module 3 - Q4: 08_cheap_fixes.

Which cheap fix recovers the most, for the least labelled data?

1. Few-shot: label n = 20, 50, 100 knockdowns in the new line (shift_eval.few_shot_indices);
   recalibrate conformal on them and apply shift_eval.few_shot_affine.
2. Weighted conformal (shift_eval.weighted_conformal_quantile), weights from a
   classifier that tells training-line rows from new-line rows.
3. Cell-context features: append a line descriptor (e.g. mean control expression, PCA'd) to X.
Writes reports/q4_fixes.csv.

Stub: not implemented yet.
"""
import argparse

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    with open(ap.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    raise SystemExit(f"08_cheap_fixes: not implemented yet (seed={cfg['seed']})")


if __name__ == "__main__":
    main()
