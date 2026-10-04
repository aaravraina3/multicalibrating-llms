"""Reliability diagrams and other figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reliability_diagram(p, y, ax=None, n_bins=20, title=None):
    """Mean label vs mean prediction per equal width bin. Marker area shows bin size; counts are printed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(p[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    ns = np.array(ns)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    ax.plot(xs, ys, color="C0", lw=1)
    ax.scatter(xs, ys, s=20 + 300 * ns / ns.max(), color="C0", alpha=0.7, zorder=3)
    for x, yy, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yy), fontsize=6, xytext=(3, -8), textcoords="offset points")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed pass rate")
    if title:
        ax.set_title(title, fontsize=9)
    return ax
