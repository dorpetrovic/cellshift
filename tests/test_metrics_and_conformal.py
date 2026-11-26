import numpy as np

from cellshift import baselines, conformal, metrics


def test_perfect_prediction_scores():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 500))
    assert np.allclose(metrics.pearson_top_de(y, y), 1.0)
    assert np.allclose(metrics.mae_top_de(y, y), 0.0)
    assert np.allclose(metrics.topk_recall(y, y), 1.0)


def test_bootstrap_interval_contains_mean():
    x = np.random.default_rng(1).normal(size=200)
    m, lo, hi = metrics.bootstrap_ci(x)
    assert lo < m < hi


def test_conformal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(2)
    y_cal, y_test = rng.normal(size=(200, 100)), rng.normal(size=(200, 100))
    pred_cal, pred_test = np.zeros_like(y_cal), np.zeros_like(y_test)
    q = conformal.calibrate(y_cal, pred_cal, alpha=0.1)
    lo, hi = conformal.intervals(pred_test, q)
    cov = metrics.coverage(lo, hi, y_test).mean()
    assert 0.88 < cov < 0.92


def test_conformal_undercovers_under_shift():
    """The failure mode CellShift measures: a noisier test domain breaks coverage."""
    rng = np.random.default_rng(3)
    y_cal = rng.normal(scale=1.0, size=(200, 100))
    y_test = rng.normal(scale=2.0, size=(200, 100))
    q = conformal.calibrate(y_cal, np.zeros_like(y_cal), alpha=0.1)
    lo, hi = conformal.intervals(np.zeros_like(y_test), q)
    assert metrics.coverage(lo, hi, y_test).mean() < 0.75


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 8))
    W = rng.normal(size=(8, 20))
    Y = X @ W
    pred = baselines.Ridge(alpha=1e-6).fit(X, Y).predict(X)
    assert np.allclose(pred, Y, atol=1e-4)
