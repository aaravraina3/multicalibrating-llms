"""IGHB, Campos Algorithm 2: repeatedly shift the group x bin cell with the largest P(g, m) * delta^2."""

import numpy as np

from calib.metrics import GRID_STEPS, grid, nearest_grid_index


class IGHB:
    """version:
    'code'  - Campos code: no rounding. Cells are the 20 intervals between grid points (np.digitize),
              so p = 1.0 is in no cell. Clip to [0, 1] after each update.
    'paper' - Algorithm 2: round p to the nearest grid point at the start and after every update.
              Cells are 20 equal width bins, the last one includes 1.0.
    Stops when max over groups of P(g) * gASCE(g) <= alpha (default 1/M, as in Campos), or at max_iter.
    """

    def __init__(self, version="code", n_bins=GRID_STEPS, max_iter=1000, alpha=None):
        self.version = version
        self.n_bins = n_bins
        self.alpha = 1 / n_bins if alpha is None else alpha
        self.max_iter = max_iter

    def _cell_index(self, p):
        if self.version == "code":
            return np.digitize(p, grid(self.n_bins)) - 1  # 0..19, and 20 for p = 1.0
        return np.minimum((p * self.n_bins).astype(int), self.n_bins - 1)

    def _round(self, p):
        return grid(self.n_bins)[nearest_grid_index(p, self.n_bins)] if self.version == "paper" else p

    def _apply(self, p, groups, b, g, delta):
        mask = (self._cell_index(p) == b) & groups[:, g]
        p = p.copy()
        p[mask] = np.clip(p[mask] + delta, 0, 1)
        return self._round(p)

    def _cell_stats(self, p, y, groups):
        """counts[b, g] and delta[b, g] = mean(y - p) in cell (0 when empty)."""
        idx = self._cell_index(p)
        onehot = np.stack([idx == b for b in range(self.n_bins)], axis=1).astype(float)  # n x M
        G = groups.astype(float)
        counts = onehot.T @ G
        resid = onehot.T @ (G * (y - p)[:, None])
        delta = np.divide(resid, counts, out=np.zeros_like(resid), where=counts > 0)
        return counts, delta

    def fit(self, p, y, groups):
        p = self._round(p.astype(float))
        n = len(p)
        self.rules, self.history = [], []
        for _ in range(self.max_iter):
            counts, delta = self._cell_stats(p, y, groups)
            # gASCE(g) = sum over bins of P(bin | g) * delta^2; P(g) * gASCE(g) = sum over bins of P(g, bin) * delta^2
            weighted = counts / n * delta ** 2
            stop_stat = weighted.sum(axis=0).max()
            self.history.append(float(stop_stat))
            if stop_stat <= self.alpha:
                break
            b, g = np.unravel_index(weighted.argmax(), weighted.shape)
            p = self._apply(p, groups, b, g, delta[b, g])
            self.rules.append((int(b), int(g), float(delta[b, g])))
        self.converged = stop_stat <= self.alpha
        self.fitted_ = p
        return self

    def predict(self, p, groups):
        """Replay the saved rules in order, without labels."""
        p = self._round(p.astype(float))
        for b, g, delta in self.rules:
            p = self._apply(p, groups, b, g, delta)
        return p
