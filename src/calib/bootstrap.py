"""Task clustered paired bootstrap: resample problems, not rows."""

import numpy as np


def _rows_by_task(task_ids):
    tasks, inverse = np.unique(task_ids, return_inverse=True)
    return [np.flatnonzero(inverse == t) for t in range(len(tasks))]


def resample_indices(task_ids, n_boot=2000, seed=0):
    """Yield row indices for each bootstrap resample. A problem drawn twice contributes its rows twice."""
    groups = _rows_by_task(task_ids)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[t] for t in picks])


def paired_diff(metric, p_a, p_b, y, task_ids, n_boot=2000, seed=0):
    """metric(A) - metric(B) on the same resampled problems. Returns point estimate and 95% interval."""
    diffs = np.array([metric(p_a[i], y[i]) - metric(p_b[i], y[i])
                      for i in resample_indices(task_ids, n_boot, seed)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": metric(p_a, y) - metric(p_b, y), "lo": float(lo), "hi": float(hi)}


def interval(metric, p, y, task_ids, n_boot=2000, seed=0):
    """95% interval for a single method's metric."""
    vals = np.array([metric(p[i], y[i]) for i in resample_indices(task_ids, n_boot, seed)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"value": metric(p, y), "lo": float(lo), "hi": float(hi)}
