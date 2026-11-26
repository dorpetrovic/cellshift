import numpy as np

from cellshift import conformal, shift_eval


def test_accuracy_drop_sign_and_interval():
    rng = np.random.default_rng(0)
    score_in, score_out = rng.normal(0.6, 0.1, 200), rng.normal(0.3, 0.1, 150)
    d, lo, hi = shift_eval.accuracy_drop(score_in, score_out)
    assert lo < d < hi and 0.25 < d < 0.35


def test_coverage_report_flags_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.normal(scale=2.0, size=(100, 50))
    rep = shift_eval.coverage_report(-np.ones_like(y), np.ones_like(y), y, target=0.9)
    assert rep["gap"] < -0.3  # +-1 covers ~38% of N(0, 2^2)


def test_warning_quality_perfect_and_useless():
    rng = np.random.default_rng(2)
    err = rng.exponential(size=400)
    perfect = shift_eval.warning_quality(err, err)
    assert perfect["spearman"] > 0.999 and perfect["auroc"] > 0.999
    useless = shift_eval.warning_quality(rng.normal(size=400), err)
    assert abs(useless["auroc"] - 0.5) < 0.1


def test_knn_distance_higher_for_far_points():
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(100, 8)) + np.array([5] + [0] * 7)
    near = X_train[:5] + 0.01
    far = -X_train[:5]
    d = shift_eval.knn_distance(X_train, np.vstack([near, far]))
    assert d[:5].max() < d[5:].min()


def test_risk_coverage_rises_for_good_score():
    err = np.arange(100, dtype=float)
    _, risk = shift_eval.risk_coverage(err, err)
    assert np.all(np.diff(risk) > 0)


def test_few_shot_affine_recovers_scale():
    rng = np.random.default_rng(4)
    pred = rng.normal(size=(20, 30))
    y = 2.0 * pred + 0.5
    fixed, a, b = shift_eval.few_shot_affine(y, pred, pred)
    assert abs(a - 2.0) < 1e-8 and abs(b - 0.5) < 1e-8 and np.allclose(fixed, y)


def test_weighted_conformal_equals_standard_with_equal_weights():
    s = np.random.default_rng(5).exponential(size=199)
    q_w = shift_eval.weighted_conformal_quantile(s, np.ones_like(s), 1.0, alpha=0.1)
    assert q_w == conformal.conformal_quantile(s, alpha=0.1)
