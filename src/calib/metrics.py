"""Evaluation metrics. All take predictions p in [0, 1] and 0/1 labels y."""

import numpy as np


def ece(p, y, n_bins=20):
    """Expected calibration error: bin-size weighted mean of |pass rate - mean prediction|."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total
