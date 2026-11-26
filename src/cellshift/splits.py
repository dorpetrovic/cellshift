"""Train/test splits for knockdown-effect prediction.

Every split is a pair of boolean masks over rows of a long table where each row
is one (cell_line, knockdown) pair. The three regimes:

1. ``unseen_knockdown``  same line, knockdowns held out by *response group*
2. ``unseen_line``       train on other lines, test on one line, knockdowns seen in training
3. ``unseen_both``       test line and test knockdowns both absent from training

``assert_no_leakage`` is the check the tests call; run it on every split you make.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray  # bool mask over rows
    test: np.ndarray   # bool mask over rows


def _as_array(x) -> np.ndarray:
    return np.asarray(list(x), dtype=object)


def unseen_knockdown(lines, knockdowns, groups, line: str, test_frac: float = 0.2, seed: int = 0) -> Split:
    """Hold out whole response groups within one cell line.

    ``groups`` assigns each row a cluster id (e.g. from clustering responses in the
    *training* lines, or protein-complex membership), so knockdowns with near-identical
    responses never straddle the split.
    """
    lines, knockdowns, groups = map(_as_array, (lines, knockdowns, groups))
    in_line = lines == line
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups[in_line])), dtype=object)
    rng.shuffle(uniq)
    n_test = max(1, round(test_frac * len(uniq)))
    test_groups = set(uniq[:n_test])
    is_test_group = np.array([g in test_groups for g in groups])
    test = in_line & is_test_group
    # A knockdown in a test group may not appear in train under another group label.
    test_kd = set(knockdowns[test])
    train = in_line & ~is_test_group & np.array([k not in test_kd for k in knockdowns])
    return Split(f"unseen_knockdown[{line}]", train, test)


def unseen_line(lines, knockdowns, test_line: str) -> Split:
    """Train on all other lines; test on ``test_line`` for knockdowns seen in training."""
    lines, knockdowns = map(_as_array, (lines, knockdowns))
    train = lines != test_line
    train_kd = set(knockdowns[train])
    test = (lines == test_line) & np.array([k in train_kd for k in knockdowns])
    return Split(f"unseen_line[{test_line}]", train, test)


def unseen_both(lines, knockdowns, test_line: str, test_frac: float = 0.2, seed: int = 0) -> Split:
    """Test line and a random subset of knockdowns are both excluded from training."""
    lines, knockdowns = map(_as_array, (lines, knockdowns))
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(knockdowns)), dtype=object)
    rng.shuffle(uniq)
    held = set(uniq[: max(1, round(test_frac * len(uniq)))])
    is_held = np.array([k in held for k in knockdowns])
    train = (lines != test_line) & ~is_held
    test = (lines == test_line) & is_held
    return Split(f"unseen_both[{test_line}]", train, test)


def assert_no_leakage(split: Split, lines, knockdowns, regime: str) -> None:
    """Raise AssertionError if the split leaks information its regime forbids."""
    lines, knockdowns = map(_as_array, (lines, knockdowns))
    tr, te = split.train, split.test
    assert not np.any(tr & te), "a row is in both train and test"
    assert te.any() and tr.any(), "empty train or test set"
    tr_kd, te_kd = set(knockdowns[tr]), set(knockdowns[te])
    tr_ln, te_ln = set(lines[tr]), set(lines[te])
    if regime == "unseen_knockdown":
        assert not (tr_kd & te_kd), f"knockdowns in both sets: {sorted(tr_kd & te_kd)[:5]}"
        assert tr_ln == te_ln and len(te_ln) == 1, "unseen_knockdown must stay within one line"
    elif regime == "unseen_line":
        assert not (tr_ln & te_ln), "test line appears in training"
        assert te_kd <= tr_kd, "a test knockdown was never seen in training"
    elif regime == "unseen_both":
        assert not (tr_ln & te_ln), "test line appears in training"
        assert not (tr_kd & te_kd), "test knockdown appears in training"
    else:
        raise ValueError(f"unknown regime {regime!r}")
