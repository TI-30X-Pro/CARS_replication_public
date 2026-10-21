"""Tests for classifier portability, ranking, and adversary confirmation."""

import numpy as np
import pytest

from cars.selection.libm import atan2f
from cars.selection.model import Trees, sample_weights
from cars.selection.selector import ConfirmationGate, rank


def gate_run(preds, k=5):
    g = ConfirmationGate(k)
    return [g(p) for p in preds]


def test_gate_keeps_first_prediction_until_confirmed():
    assert gate_run([3, 7, 7, 7, 7, 7, 7]) == [3, 3, 3, 3, 3, 7, 7]


def test_gate_restarts_when_the_run_breaks():
    # back to the active one, or another candidate: the count starts again
    assert gate_run([3, 7, 7, 7, 7, 3, 7, 7, 7, 7, 7]) == [3] * 10 + [7]
    assert gate_run([3, 7, 7, 7, 7, 8, 7, 7, 7, 7, 7]) == [3] * 10 + [7]


def test_gate_k1_follows_every_prediction():
    assert gate_run([3, 7, 8, 8], k=1) == [3, 7, 8, 8]


def test_rank_puts_target_last_and_breaks_ties_by_higher_index():
    order = rank([0.2, 0.9, 0.5, 0.5, 0.95], target=4)
    assert list(order) == [1, 3, 2, 0, 4]


def test_weights_balance_the_classes():
    w = sample_weights([0, 0, 0, 0, 1])
    assert w[-1] == pytest.approx(2.0) and (w[:-1] == 1).all()


def test_trees_predict_like_scikit_learn():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) > 0.3).astype(int)
    m = HistGradientBoostingClassifier(max_iter=30, max_depth=3).fit(X, y)
    t = Trees.from_sklearn(m)
    Xt = rng.normal(size=(200, 4)).astype(np.float32)
    Xt[rng.random(Xt.shape) < 0.1] = np.nan
    np.testing.assert_array_equal(t.predict_proba(Xt), m.predict_proba(Xt)[:, 1])


def test_atan2f_within_one_ulp():
    rng = np.random.default_rng(1)
    y = (rng.normal(size=100_000) * rng.choice([1e-3, 1, 100], 100_000)).astype(np.float32)
    x = rng.normal(size=100_000).astype(np.float32)
    exact = np.arctan2(y.astype(np.float64), x.astype(np.float64)).astype(np.float32)
    ulps = np.abs(atan2f(y, x).view(np.int32).astype(np.int64) - exact.view(np.int32).astype(np.int64))
    assert ulps.max() <= 1


def test_atan2f_special_cases():
    pi = np.float32(np.pi)
    y = np.array([0.0, -0.0, 0.0, 2.0, -2.0, 0.5, np.nan], dtype=np.float32)
    x = np.array([1.0, 3.0, -1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    out = atan2f(y, x)
    assert out[0] == 0 and out[1] == 0 and np.signbit(out[1])
    assert out[2] == pi and out[3] == np.float32(np.pi / 2) and out[4] == -np.float32(np.pi / 2)
    assert abs(out[5] - np.arctan(0.5)) < 1e-7 and np.isnan(out[6])


def test_shipped_classifier_read_without_old_scikit_learn():
    t = Trees.from_pickle()
    assert t.n_trees == 800 and t.baseline == pytest.approx(-1.32741, abs=1e-5)
