"""Histogram binning and isotonic regression. Both ignore groups."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from calib.metrics import GRID_STEPS, grid, nearest_grid_index


class HistogramBinning:
    """Round p to the nearest of 21 grid points, then add that point's average train residual."""

    def __init__(self, n_bins=GRID_STEPS):
        self.n_bins = n_bins

    def fit(self, p, y, groups=None):
        g = grid(self.n_bins)
        idx = nearest_grid_index(p, self.n_bins)
        self.shift = np.zeros(len(g))
        for b in range(len(g)):
            m = idx == b
            if m.any():
                self.shift[b] = y[m].mean() - g[b]
        return self

    def predict(self, p, groups=None):
        idx = nearest_grid_index(p, self.n_bins)
        return np.clip(grid(self.n_bins)[idx] + self.shift[idx], 0, 1)


class Isotonic:
    """Best non-decreasing staircase from p to y. Our addition, not in Campos's table."""

    def fit(self, p, y, groups=None):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def predict(self, p, groups=None):
        return self.model.predict(p)
