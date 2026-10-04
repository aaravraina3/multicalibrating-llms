import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from calib.calibrators.ighb import IGHB
from calib.calibrators.iglb import IGLB
from calib.calibrators.linr import LINR
from calib.calibrators.logr import LOGR
from calib.metrics import max_gasce


def population(n, seed):
    """Phase 2 population: an interaction the main-effects predictor misses."""
    rng = np.random.default_rng(seed)
    long, nested, imports = rng.random(n) < 0.4, rng.random(n) < 0.3, rng.random(n) < 0.5
    p_true = expit(1.0 - 0.8 * long - 0.5 * nested - 1.2 * (long & imports))
    y = (rng.random(n) < p_true).astype(int)
    X = np.column_stack([long, nested, imports]).astype(float)
    groups = np.column_stack([np.ones(n, bool), long, nested, imports, long & imports])
    return X, groups, y


X0, _, y0 = population(50_000, 0)
base = LogisticRegression(C=1e6).fit(X0, y0)
predictor = lambda X: expit(2 * base.decision_function(X))
Xtr, Gtr, ytr = population(20_000, 1)
Xva, Gva, yva = population(10_000, 2)
Xte, Gte, yte = population(50_000, 3)
ptr, pva, pte = predictor(Xtr), predictor(Xva), predictor(Xte)


@pytest.mark.parametrize("version", ["code", "paper"])
def test_ighb_reduces_fresh_max_gasce(version):
    m = IGHB(version, alpha=0.001).fit(ptr, ytr, Gtr)
    assert max_gasce(m.predict(pte, Gte), yte, Gte) < max_gasce(pte, yte, Gte) / 2


@pytest.mark.parametrize("version", ["code", "paper"])
def test_iglb_reduces_fresh_max_gasce(version):
    m = IGLB(version).fit(ptr, ytr, Gtr, pva, yva, Gva)
    assert max_gasce(m.predict(pte, Gte), yte, Gte) < max_gasce(pte, yte, Gte) / 2


@pytest.mark.parametrize("version", ["code", "paper"])
def test_ighb_replay_reproduces_train(version):
    m = IGHB(version, alpha=0.001).fit(ptr, ytr, Gtr)
    assert len(m.rules) > 0
    assert np.array_equal(m.predict(ptr, Gtr), m.fitted_)


@pytest.mark.parametrize("version", ["code", "paper"])
def test_iglb_replay_reproduces_train(version):
    m = IGLB(version).fit(ptr, ytr, Gtr, pva, yva, Gva)
    assert len(m.rules) > 0
    assert np.array_equal(m.predict(ptr, Gtr), m.fitted_)


def test_linr_zeroes_group_mean_residuals_on_train():
    for version in ["code", "paper"]:
        q = LINR(version).fit(ptr, ytr, Gtr).predict(ptr, Gtr)
        for g in range(Gtr.shape[1]):
            assert abs(np.mean(ytr[Gtr[:, g]] - q[Gtr[:, g]])) < 0.01


def test_logr_code_outputs_hard_labels():
    q = LOGR("code").fit(ptr, ytr, Gtr).predict(pte, Gte)
    assert set(np.unique(q)) <= {0.0, 1.0}
    q = LOGR("paper").fit(ptr, ytr, Gtr).predict(pte, Gte)
    assert len(np.unique(q)) > 2


def test_ighb_campos_alpha_is_loose():
    # max P(g) * gASCE(g) of this predictor is below 1/20, so the Campos stopping rule never starts.
    assert len(IGHB("code").fit(ptr, ytr, Gtr).rules) == 0
