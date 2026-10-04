"""Two IGHB variants that guard against fixing noise (RQ4). Both use the code version's cells, no rounding."""

import numpy as np

from calib.calibrators.ighb import IGHB


def cluster_se(resid, task, mask, min_problems=5):
    """Standard error of the mean residual in a cell, treating each problem as one unit.
    Infinite when the cell spans fewer than min_problems problems: a cluster SE from 2 or 3 problems is unreliable."""
    r, t = resid[mask], task[mask]
    if len(r) < 2:
        return np.inf
    _, inv = np.unique(t, return_inverse=True)
    sums, counts = np.bincount(inv, r), np.bincount(inv)
    mean = r.mean()
    if len(counts) < min_problems:
        return np.inf
    return float(np.sqrt(np.sum((sums - counts * mean) ** 2)) / len(r))


class NoiseAwareIGHB(IGHB):
    """Only update a cell when its gap exceeds z standard errors, computed by problem."""

    def __init__(self, z=2.0, **kw):
        super().__init__("code", **kw)
        self.z = z

    def fit(self, p, y, groups, task):
        p = p.astype(float).copy()
        n = len(p)
        self.rules = []
        for _ in range(self.max_iter):
            counts, delta = self._cell_stats(p, y, groups)
            weighted = counts / n * delta ** 2
            if weighted.sum(axis=0).max() <= self.alpha:
                break
            idx, resid = self._cell_index(p), y - p
            picked = None
            for flat in np.argsort(-weighted, axis=None):
                b, g = np.unravel_index(flat, weighted.shape)
                if weighted[b, g] == 0:
                    break
                mask = (idx == b) & groups[:, g]
                if abs(delta[b, g]) > self.z * cluster_se(resid, task, mask):
                    picked = (b, g)
                    break
            if picked is None:
                break
            b, g = picked
            p = self._apply(p, groups, b, g, delta[b, g])
            self.rules.append((int(b), int(g), float(delta[b, g])))
        self.fitted_ = p
        return self


class HoldoutIGHB(IGHB):
    """Accept each proposed update only if it also lowers Brier on a holdout, judged with Laplace noise
    (Thresholdout style, Dwork et al. 2015). Rejected cells are skipped until the next accepted update."""

    def __init__(self, noise=2e-4, seed=0, **kw):
        super().__init__("code", **kw)
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def fit(self, p, y, groups, p_hold, y_hold, groups_hold):
        p, p_hold = p.astype(float).copy(), p_hold.astype(float).copy()
        n = len(p)
        self.rules, blocked = [], set()
        for _ in range(self.max_iter):
            counts, delta = self._cell_stats(p, y, groups)
            weighted = counts / n * delta ** 2
            if weighted.sum(axis=0).max() <= self.alpha:
                break
            candidates = [np.unravel_index(f, weighted.shape) for f in np.argsort(-weighted, axis=None)]
            candidates = [(b, g) for b, g in candidates if weighted[b, g] > 0 and (b, g) not in blocked]
            if not candidates:
                break
            b, g = candidates[0]
            new_hold = self._apply(p_hold, groups_hold, b, g, delta[b, g])
            gain = np.mean((p_hold - y_hold) ** 2) - np.mean((new_hold - y_hold) ** 2)
            if gain + self.rng.laplace(0, self.noise) > 0:
                p = self._apply(p, groups, b, g, delta[b, g])
                p_hold = new_hold
                self.rules.append((int(b), int(g), float(delta[b, g])))
                blocked = set()
            else:
                blocked.add((b, g))
        self.fitted_ = p
        return self
