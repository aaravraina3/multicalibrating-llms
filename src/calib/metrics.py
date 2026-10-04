"""Evaluation metrics. All take predictions p in [0, 1] and 0/1 labels y."""

import numpy as np
from sklearn.metrics import roc_auc_score

GRID_STEPS = 20  # Campos grid: 21 points 0, 0.05, ..., 1.0


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def brier_skill(p, y):
    """(ref - brier) / ref, ref = r(1 - r) with r the pass rate of the evaluated rows (Campos convention)."""
    r = np.mean(y)
    ref = r * (1 - r)
    if ref == 0:
        return 1.0 if brier(p, y) == 0 else -np.inf
    return float((ref - brier(p, y)) / ref)


def log_loss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))


def accuracy(p, y):
    """Correct when p > 0.5, strictly greater as in Campos code."""
    return float(np.mean((p > 0.5) == y))


def auroc(p, y):
    return float(roc_auc_score(y, p))


def grid(n_bins=GRID_STEPS):
    """Campos grid, rounded to avoid float drift: 0, 0.05, ..., 1.0."""
    return np.round(np.arange(n_bins + 1) / n_bins, 10)


def nearest_grid_index(p, n_bins=GRID_STEPS):
    """Index of the nearest grid point. Exact ties go to the lower point, as argmin does in their code."""
    return np.abs(np.asarray(p)[:, None] - grid(n_bins)[None, :]).argmin(axis=1)


def _bins(p, n_bins, round_to_grid):
    """Bin index per row and the confidence used for each row.

    round_to_grid=False: equal width bins, last one includes 1.0, confidence is the row's own p.
    round_to_grid=True: Campos code. Round p to the nearest of n_bins + 1 grid points; each
    grid point is a bin and its value is the confidence.
    """
    if round_to_grid:
        idx = nearest_grid_index(p, n_bins)
        return idx, idx / n_bins, n_bins + 1
    return np.minimum((np.clip(p, 0, 1) * n_bins).astype(int), n_bins - 1), p, n_bins


def ece(p, y, n_bins=GRID_STEPS, round_to_grid=False):
    """Bin-size weighted mean of |pass rate - mean confidence| over bins."""
    idx, conf, k = _bins(p, n_bins, round_to_grid)
    total = 0.0
    for b in range(k):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - conf[m].mean())
    return float(total)


def gasce(p, y, mask, n_bins=GRID_STEPS, round_to_grid=False):
    """Calibration error inside one group: sum over bins of P(bin | group) * mean(y - p in bin)^2."""
    idx, conf, k = _bins(p, n_bins, round_to_grid)
    idx, conf, yg = idx[mask], conf[mask], y[mask]
    if len(yg) == 0:
        return float("nan")
    total = 0.0
    for b in range(k):
        m = idx == b
        if m.any():
            total += m.mean() * (yg[m] - conf[m]).mean() ** 2
    return float(total)


def gasce_all(p, y, groups, n_bins=GRID_STEPS, round_to_grid=False):
    """gASCE for every column of the n x G boolean group matrix."""
    return np.array([gasce(p, y, groups[:, g], n_bins, round_to_grid) for g in range(groups.shape[1])])


def max_gasce(p, y, groups, n_bins=GRID_STEPS, round_to_grid=False):
    return float(np.nanmax(gasce_all(p, y, groups, n_bins, round_to_grid)))


def summary(p, y, groups=None, round_to_grid=False):
    """All headline metrics in one dict."""
    out = {"brier": brier(p, y), "bss": brier_skill(p, y), "log_loss": log_loss(p, y),
           "ece": ece(p, y, round_to_grid=round_to_grid), "acc": accuracy(p, y), "auroc": auroc(p, y)}
    if groups is not None:
        out["max_gasce"] = max_gasce(p, y, groups, round_to_grid=round_to_grid)
    return out
