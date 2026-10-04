"""Routing between a primary model A and a fallback model B. Each row is one (problem, sample k) pair.

A policy returns use_b: True where the final answer comes from B."""

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

# Assumed costs, not measured: A costs 1 call, B costs C_B in {1, 2, 5}, a wrong shipped answer costs L.
C_A = 1.0
C_B_SCENARIOS = [1.0, 2.0, 5.0]
LOSS_WRONG = 10.0
RATES = [0.1, 0.2, 0.3, 0.4]


def always_primary(p_a):
    return np.zeros(len(p_a), bool)


def always_fallback(p_a):
    return np.ones(len(p_a), bool)


def oracle(y_a, y_b):
    """Escalate only when A fails and B passes. Uses labels, so it's a reference, not a policy."""
    return (y_a == 0) & (y_b == 1)


def lowest_fraction(p_a, cutoff):
    """Rank based: escalate rows whose primary confidence is below a cutoff set on other data."""
    return p_a < cutoff


def cost_threshold(p_a, q_b, c_b, loss=LOSS_WRONG):
    """Scale based: escalate when L * (q_B - p_A) > c_B, i.e. p_A < q_B - c_B / L."""
    return p_a < q_b - c_b / loss


def fallback_given_primary(p_a_calib, y_b_calib):
    """q_B as a function of A's confidence: logistic fit of B's outcome on logit p_A, over calib pairs.
    The two models fail on the same hard problems, so B's overall pass rate overstates its chances exactly
    where A looks worst."""
    x = logit(np.clip(p_a_calib, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model = LogisticRegression(C=1e6).fit(x, y_b_calib)
    return lambda p_a: model.predict_proba(logit(np.clip(p_a, 1e-6, 1 - 1e-6)).reshape(-1, 1))[:, 1]


def pick_higher(p_a, p_b):
    """Run both, keep B's answer when its calibrated probability is higher. Ties keep A."""
    return p_b > p_a


def evaluate(use_b, y_a, y_b, c_b, both_run=False, loss=LOSS_WRONG):
    final = np.where(use_b, y_b, y_a)
    kept_a = ~use_b
    esc = use_b.mean()
    compute = C_A + c_b if both_run else C_A + c_b * esc
    return {"pass_rate": final.mean(), "escalation": esc,
            "accepted_error": float((y_a[kept_a] == 0).mean()) if kept_a.any() else np.nan,
            "compute_cost": compute, "total_cost": compute + loss * (1 - final.mean())}


def oracle_pass_at(y_a, y_b, rate):
    """Best pass rate reachable when escalating a fraction `rate` of rows."""
    return y_a.mean() + min(rate, ((y_a == 0) & (y_b == 1)).mean())
