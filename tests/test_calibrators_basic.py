import numpy as np

from calib.calibrators.histogram import HistogramBinning
from calib.calibrators.platt import Platt
from calib.metrics import ece

rng = np.random.default_rng(0)


def calibrated(n=50_000):
    p = rng.uniform(0.02, 0.98, n)
    return p, (rng.random(n) < p).astype(int)


def test_logit_platt_is_identity_on_calibrated_input():
    p, y = calibrated()
    platt = Platt("logit").fit(p, y)
    assert abs(platt.a - 1) < 0.05 and abs(platt.b) < 0.05
    assert np.max(np.abs(platt.predict(p) - p)) < 0.02


def test_platt_keeps_ranking():
    p, y = calibrated()
    for version in ["code", "paper", "logit"]:
        q = Platt(version).fit(p, y).predict(p)
        assert np.all(np.diff(q[np.argsort(p)]) >= 0)


def test_histogram_reduces_train_ece():
    p, y = calibrated()
    p_bad = p ** 0.3  # overconfident
    q = HistogramBinning().fit(p_bad, y).predict(p_bad)
    assert ece(q, y) < ece(p_bad, y) / 5
