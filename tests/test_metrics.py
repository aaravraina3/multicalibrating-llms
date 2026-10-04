import numpy as np

from calib import metrics as M
from calib.bootstrap import paired_diff

rng = np.random.default_rng(0)


def test_perfect_brier_zero():
    y = rng.integers(0, 2, 1000)
    assert M.brier(y.astype(float), y) == 0


def test_base_rate_bss_zero():
    y = rng.integers(0, 2, 1000)
    assert abs(M.brier_skill(np.full(1000, y.mean()), y)) < 1e-12


def test_calibrated_ece_near_zero():
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert M.ece(p, y) < 0.01
    assert M.gasce(p, y, p < 0.5) < 0.001


def test_constant_prediction_can_have_zero_ece():
    y = rng.integers(0, 2, 1000)
    assert M.ece(np.full(1000, y.mean()), y) < 1e-12


def test_bootstrap_self_is_zero():
    p = rng.random(500)
    y = rng.integers(0, 2, 500)
    tasks = np.repeat(np.arange(50), 10)
    res = paired_diff(M.brier, p, p, y, tasks, n_boot=200)
    assert res["lo"] == 0 and res["hi"] == 0


def test_accuracy_is_strict():
    assert M.accuracy(np.array([0.5]), np.array([1])) == 0


def test_round_to_grid_matches_hand_example():
    # 0.52 rounds to 0.50, 0.53 to 0.55, 0.97 to 0.95, 0.98 to 1.0.
    p = np.array([0.52, 0.53, 0.97, 0.98])
    y = np.array([1, 0, 1, 1])
    # bins: 0.50 -> [1] (gap 0.5), 0.55 -> [0] (gap 0.55), 0.95 -> [1] (gap 0.05), 1.0 -> [1] (gap 0)
    expected = 0.25 * (0.5 + 0.55 + 0.05 + 0.0)
    assert abs(M.ece(p, y, round_to_grid=True) - expected) < 1e-12


def test_grid_ties_go_low():
    assert M.nearest_grid_index(np.array([0.375]))[0] == 7
