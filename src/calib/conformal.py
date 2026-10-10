"""Conformal risk control (Angelopoulos et al. 2022) for accepting primary answers.

Loss of one answer at threshold t: 1 if it is accepted (p >= t) and fails, else 0. It can only go down as t
rises. Losses are averaged within each problem; problems are the exchangeable units."""

import numpy as np


def problem_losses(p, y, task, t):
    """Per problem: share of its answers that are accepted and fail."""
    bad = ((p >= t) & (y == 0)).astype(float)
    _, inv = np.unique(task, return_inverse=True)
    return np.bincount(inv, bad) / np.bincount(inv)


def choose_threshold(p, y, task, target):
    """Smallest t with (n / (n + 1)) * mean_loss(t) + 1 / (n + 1) <= target, n = number of problems.
    Candidates are the observed scores plus +inf (accept nothing, loss 0)."""
    n = len(np.unique(task))
    for t in np.append(np.unique(p), np.inf):
        if n / (n + 1) * problem_losses(p, y, task, t).mean() + 1 / (n + 1) <= target:
            return float(t)
    return float("inf")


def realized_risk(p, y, task, t):
    return float(problem_losses(p, y, task, t).mean())
