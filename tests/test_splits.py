import numpy as np
import pytest

from cellshift import splits


@pytest.fixture
def table():
    lines, kds, groups = [], [], []
    for line in ["K562", "RPE1", "HepG2", "Jurkat"]:
        for i in range(60):
            lines.append(line)
            kds.append(f"G{i}")
            groups.append(f"cluster{i // 3}")  # 3 knockdowns per response group
    return np.array(lines, object), np.array(kds, object), np.array(groups, object)


def test_unseen_knockdown_has_no_overlap(table):
    lines, kds, groups = table
    s = splits.unseen_knockdown(lines, kds, groups, line="K562")
    splits.assert_no_leakage(s, lines, kds, "unseen_knockdown")
    # whole groups are held out together
    assert set(groups[s.train]).isdisjoint(set(groups[s.test]))


def test_unseen_line(table):
    lines, kds, _ = table
    s = splits.unseen_line(lines, kds, test_line="Jurkat")
    splits.assert_no_leakage(s, lines, kds, "unseen_line")


def test_unseen_both(table):
    lines, kds, _ = table
    s = splits.unseen_both(lines, kds, test_line="HepG2")
    splits.assert_no_leakage(s, lines, kds, "unseen_both")


def test_leak_is_caught(table):
    lines, kds, _ = table
    s = splits.unseen_line(lines, kds, test_line="Jurkat")
    leaky = splits.Split("leaky", s.train | (lines == "Jurkat") & (kds == "G0"), s.test & ~((lines == "Jurkat") & (kds == "G0")))
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(leaky, lines, kds, "unseen_line")
