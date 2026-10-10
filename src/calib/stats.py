"""Murphy decomposition of the Brier score and Holm's multiple comparison correction."""

import numpy as np

from calib.bootstrap import resample_indices


def murphy(p, y, n_bins=20):
    """brier ~= reliability - resolution + uncertainty, with 20 equal width bins.
    Exact only when predictions are constant within each bin; `gap` reports the leftover."""
    idx = np.minimum((np.clip(p, 0, 1) * n_bins).astype(int), n_bins - 1)
    ybar = y.mean()
    rel = res = 0.0
    for k in range(n_bins):
        m = idx == k
        if m.any():
            w = m.mean()
            rel += w * (p[m].mean() - y[m].mean()) ** 2
            res += w * (y[m].mean() - ybar) ** 2
    unc = ybar * (1 - ybar)
    brier = float(np.mean((p - y) ** 2))
    return {"brier": brier, "reliability": rel, "resolution": res, "uncertainty": unc,
            "gap": brier - (rel - res + unc)}


def bootstrap_diffs(stat, p_a, p_b, y, task_ids, n_boot=2000, seed=0):
    """stat(A) - stat(B) on each task clustered resample."""
    return np.array([stat(p_a[i], y[i]) - stat(p_b[i], y[i]) for i in resample_indices(task_ids, n_boot, seed)])


def two_sided_p(diffs):
    """2 * min(share <= 0, share >= 0), capped at 1."""
    return float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))


def holm(pvals):
    """Holm adjusted p values in the original order: sort ascending, multiply rank i by (m - i + 1),
    make non decreasing down the sorted list, cap at 1."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj_sorted
    return out
