"""Simplified HJKRR 2018 Algorithm 3.2: fix any big enough group x bin slice whose gap exceeds alpha."""

import numpy as np


def bin_index(p, n_bins):
    """Equal width bins on [0, 1]; the last bin includes 1.0."""
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


class HJKRR:
    def __init__(self, n_bins=10, alpha=0.02, min_size=100, max_updates=1000):
        self.n_bins = n_bins
        self.alpha = alpha
        self.min_size = min_size
        self.max_updates = max_updates
        self.rules = []
        self.distances = []

    def fit(self, p, y, groups, p_true=None):
        """groups: bool matrix n x G. p_true: only for synthetic data, to track distance to truth."""
        p = p.astype(float).copy()
        self.rules = []
        self.distances = [np.mean((p - p_true) ** 2)] if p_true is not None else []
        while len(self.rules) < self.max_updates:
            updated = False
            for g in range(groups.shape[1]):
                for b in range(self.n_bins):
                    # Bins are recomputed every time because earlier updates move rows between bins.
                    mask = groups[:, g] & (bin_index(p, self.n_bins) == b)
                    if mask.sum() < self.min_size:
                        continue
                    gap = y[mask].mean() - p[mask].mean()
                    if abs(gap) > self.alpha:
                        p[mask] = np.clip(p[mask] + gap, 0, 1)
                        self.rules.append((g, b, gap))
                        if p_true is not None:
                            self.distances.append(np.mean((p - p_true) ** 2))
                        updated = True
                        if len(self.rules) >= self.max_updates:
                            break
                if len(self.rules) >= self.max_updates:
                    break
            if not updated:
                break
        self.fitted_ = p
        return self

    def predict(self, p, groups):
        """Replay the saved updates in order. Uses no labels."""
        p = p.astype(float).copy()
        for g, b, gap in self.rules:
            mask = groups[:, g] & (bin_index(p, self.n_bins) == b)
            p[mask] = np.clip(p[mask] + gap, 0, 1)
        return p
