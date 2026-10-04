"""IGLB, Campos Algorithm 3: overlapping cells (p <= t or p >= t, within a group) patched with
q = sigmoid(a + b * logit p), fit to minimize Brier on the cell. Early stops on a validation set."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

from calib.metrics import GRID_STEPS, grid, nearest_grid_index

LOGIT_EPS = 1e-10  # their clip before logit


def _patch(p, a, b):
    return expit(a + b * logit(np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)))


def fit_patch(p, y):
    """a, b minimizing mean (sigmoid(a + b logit p) - y)^2, BFGS from (0, 1) as in their code."""
    if len(p) == 0:
        return 0.0, 1.0
    res = minimize(lambda ab: np.mean((_patch(p, ab[0], ab[1]) - y) ** 2), x0=[0.0, 1.0], method="BFGS")
    return float(res.x[0]), float(res.x[1])


class IGLB:
    """version:
    'code'  - Campos code: no rounding.
    'paper' - Algorithm 3: start from rounded p and round after every accepted patch.
    Stops when the chosen cell holds less than epsilon of the rows, or validation Brier does not drop.
    """

    def __init__(self, version="code", n_bins=GRID_STEPS, epsilon=0.01, max_iter=1000):
        self.version = version
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.thresholds = grid(n_bins)

    def _round(self, p):
        return grid(self.n_bins)[nearest_grid_index(p, self.n_bins)] if self.version == "paper" else p

    def _cell(self, p, groups, tau, t, g):
        side = p <= self.thresholds[t] if tau == 0 else p >= self.thresholds[t]
        return side & groups[:, g]

    def _apply(self, p, groups, rule):
        tau, t, g, a, b = rule
        mask = self._cell(p, groups, tau, t, g)
        p = p.copy()
        p[mask] = _patch(p[mask], a, b)
        return p

    def _scores(self, p, y, groups):
        """P[tau, t, g] = share of all rows in the cell; delta = mean(y - p) in the cell."""
        n, G = len(p), groups.astype(float)
        P = np.zeros((2, len(self.thresholds), groups.shape[1]))
        delta = np.zeros_like(P)
        for tau in (0, 1):
            for t, thr in enumerate(self.thresholds):
                side = (p <= thr) if tau == 0 else (p >= thr)
                counts = side @ G
                resid = (side * (y - p)) @ G
                P[tau, t] = counts / n
                delta[tau, t] = np.divide(resid, counts, out=np.zeros_like(resid), where=counts > 0)
        return P, delta

    def fit(self, p, y, groups, p_val, y_val, groups_val):
        p, p_val = self._round(p.astype(float)), self._round(p_val.astype(float))
        self.rules, self.val_brier = [], [float(np.mean((p_val - y_val) ** 2))]
        self.stop_reason = "max_iter"
        for _ in range(self.max_iter):
            P, delta = self._scores(p, y, groups)
            tau, t, g = np.unravel_index((P * delta ** 2).argmax(), P.shape)
            if P[tau, t, g] < self.epsilon:
                self.stop_reason = "epsilon"
                break
            mask = self._cell(p, groups, tau, t, g)
            a, b = fit_patch(p[mask], y[mask])
            rule = (int(tau), int(t), int(g), a, b)
            new_val = self._apply(p_val, groups_val, rule)
            # Paper version: judge the rounded result. Judging the unrounded one can loop forever when
            # rounding undoes a patch that gained ~1e-10 (seen on Qwen3 at the p = 1.0 cell).
            brier_after = float(np.mean((self._round(new_val) - y_val) ** 2))
            if brier_after >= self.val_brier[-1]:
                self.stop_reason = "validation"
                break
            self.rules.append(rule)
            p = self._round(self._apply(p, groups, rule))
            p_val = self._round(new_val)
            self.val_brier.append(float(np.mean((p_val - y_val) ** 2)))
        self.fitted_ = p
        return self

    def predict(self, p, groups):
        p = self._round(p.astype(float))
        for rule in self.rules:
            p = self._round(self._apply(p, groups, rule))
        return p
