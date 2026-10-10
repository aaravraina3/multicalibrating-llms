import numpy as np

from calib.stats import holm, murphy, two_sided_p


def test_murphy_identity_exact_when_constant_within_bins():
    rng = np.random.default_rng(0)
    # predictions take one value per bin, so the binned decomposition is exact
    p = rng.choice([0.125, 0.375, 0.625, 0.875], 5000)
    y = (rng.random(5000) < p).astype(int)
    d = murphy(p, y, n_bins=4)
    assert abs(d["gap"]) < 1e-12
    assert abs(d["brier"] - (d["reliability"] - d["resolution"] + d["uncertainty"])) < 1e-12


def test_murphy_toy_by_hand():
    p = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0, 1, 1, 1])
    d = murphy(p, y, n_bins=2)
    # bins: {0.2: y mean 0.5}, {0.8: y mean 1.0}; ybar = 0.75
    assert abs(d["reliability"] - (0.5 * 0.3 ** 2 + 0.5 * 0.2 ** 2)) < 1e-12
    assert abs(d["resolution"] - (0.5 * 0.25 ** 2 + 0.5 * 0.25 ** 2)) < 1e-12
    assert abs(d["uncertainty"] - 0.1875) < 1e-12


def test_holm_textbook_example():
    # p = 0.01, 0.04, 0.03, 0.005 -> sorted 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, max(0.04, 0.06)
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one():
    assert np.allclose(holm([0.5, 0.6]), [1.0, 1.0])


def test_two_sided_p():
    assert two_sided_p(np.array([1.0, 2.0, 3.0, -1.0])) == 0.5
    assert two_sided_p(np.array([1.0, 2.0])) == 0.0
