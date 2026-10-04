"""Phase 2: synthetic sandbox where the true pass probability p* is known."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from calib.calibrators.hjkrr import HJKRR
from calib.metrics import ece
from calib.plots import reliability_diagram

OUT = Path("runs/phase2")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
results = {}


def population(n, rng):
    long = rng.random(n) < 0.4
    nested = rng.random(n) < 0.3
    imports = rng.random(n) < 0.5
    p_true = expit(1.0 - 0.8 * long - 0.5 * nested - 1.2 * (long & imports))
    y = (rng.random(n) < p_true).astype(int)
    X = np.column_stack([long, nested, imports]).astype(float)
    groups = np.column_stack([np.ones(n, bool), long, nested, imports, long & imports])
    return X, groups, p_true, y


# 1-2. Population and an overconfident main-effects-only predictor.
X, groups, p_true, y = population(100_000, rng)
lr = LogisticRegression(C=1e6).fit(X, y)
predictor = lambda X: expit(2 * lr.decision_function(X))  # doubled logit = overconfident
p0 = predictor(X)
results["start"] = {"ece": ece(p0, y), "dist_to_truth": float(np.mean((p0 - p_true) ** 2))}

# 3. Reliability diagram.
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
reliability_diagram(p0, y, axes[0], n_bins=10, title=f"overconfident predictor, ECE {ece(p0, y):.3f}")
reliability_diagram(p_true, y, axes[1], n_bins=10, title=f"true p*, ECE {ece(p_true, y):.3f}")
fig.tight_layout()
fig.savefig(OUT / "reliability.png", dpi=150)

# 4-5. HJKRR loop, distance to truth after every update. Two cases.
big = HJKRR(n_bins=10, alpha=0.02, min_size=500).fit(p0, y, groups, p_true)
Xs, gs, pts, ys = population(1_000, np.random.default_rng(1))
small = HJKRR(n_bins=10, alpha=0.005, min_size=10).fit(predictor(Xs), ys, gs, pts)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, run, name in [(axes[0], big, "large slices: n=100,000, min_size=500, alpha=0.02"),
                      (axes[1], small, "small slices: n=1,000, min_size=10, alpha=0.005")]:
    d = np.array(run.distances)
    ups = np.diff(d) > 0
    ax.plot(d, marker="o", ms=3)
    ax.plot(np.where(ups)[0] + 1, d[1:][ups], "rx", label="update moved away from truth")
    ax.set_xlabel("update")
    ax.set_ylabel("mean (p - p*)^2")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "distance_per_update.png", dpi=150)
for key, run in [("large_slices", big), ("small_slices", small)]:
    d = np.array(run.distances)
    results[key] = {"updates": len(run.rules), "updates_moving_away": int((np.diff(d) > 0).sum()),
                    "dist_start": float(d[0]), "dist_end": float(d[-1])}
p_big = big.fitted_
results["large_slices"]["ece_after"] = ece(p_big, y)

# 6. Winner's curse: perfectly calibrated predictor, 30 slices of 200, fix the biggest gap.
reps, before, after_fresh, picked = 2000, [], [], []
for r in range(reps):
    pt = rng.uniform(0.2, 0.8, 6000)
    yy = (rng.random(6000) < pt).astype(int)
    slices = rng.permutation(6000).reshape(30, 200)
    gaps = np.array([yy[s].mean() - pt[s].mean() for s in slices])
    worst = slices[np.argmax(np.abs(gaps))]
    fixed = pt[worst] + gaps[np.argmax(np.abs(gaps))]
    fresh = (rng.random(200) < pt[worst]).astype(int)
    picked.append(abs(gaps).max())
    before.append(abs(fresh.mean() - pt[worst].mean()))
    after_fresh.append(abs(fresh.mean() - fixed.mean()))
results["winners_curse"] = {"mean_biggest_gap": float(np.mean(picked)),
                            "fresh_gap_unfixed": float(np.mean(before)),
                            "fresh_gap_after_fix": float(np.mean(after_fresh)),
                            "share_reps_worse": float(np.mean(np.array(after_fresh) > np.array(before)))}
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(before, bins=40, alpha=0.6, label="fresh gap, left alone")
ax.hist(after_fresh, bins=40, alpha=0.6, label="fresh gap, after 'fixing'")
ax.set_xlabel("|pass rate - mean prediction| on fresh labels")
ax.set_title("winner's curse: fixing the biggest of 30 noisy gaps", fontsize=9)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "winners_curse.png", dpi=150)

# 7. Overfitting vs sample size: fit on n rows, measure distance to truth on a fresh 100,000.
Xf, gf, ptf, _ = population(100_000, np.random.default_rng(2))
pf = predictor(Xf)
start = np.mean((pf - ptf) ** 2)
sizes, curve, curve_true = [200, 500, 2000, 10000], {}, {}
for n in sizes:
    vals, vals_true = [], []
    for r in range(20):
        Xn, gn, ptn, yn = population(n, np.random.default_rng(100 + 1000 * n + r))
        run = HJKRR(n_bins=10, alpha=0.01, min_size=10).fit(predictor(Xn), yn, gn)
        vals.append(np.mean((run.predict(pf, gf) - ptf) ** 2))
        # Starting from the truth there is nothing to fix, so any change is fitted noise.
        run_true = HJKRR(n_bins=10, alpha=0.01, min_size=10).fit(ptn, yn, gn)
        vals_true.append(np.mean((run_true.predict(ptf, gf) - ptf) ** 2))
    curve[n], curve_true[n] = vals, vals_true
summ = lambda v: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
results["overfitting"] = {"start": float(start),
                          "from_overconfident": {str(n): summ(v) for n, v in curve.items()},
                          "from_truth": {str(n): summ(v) for n, v in curve_true.items()}}
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, c, title, base in [(axes[0], curve, "start: overconfident predictor", start),
                           (axes[1], curve_true, "start: true p* (nothing to fix)", 0.0)]:
    ax.plot(sizes, [np.mean(c[n]) for n in sizes], marker="o", label="after HJKRR (mean of 20)")
    for n in sizes:
        ax.scatter([n] * 20, c[n], color="C0", alpha=0.25, s=10)
    ax.axhline(base, color="gray", ls="--", label="before calibration")
    ax.set_xscale("log")
    ax.set_xlabel("rows used to fit")
    ax.set_ylabel("mean (p - p*)^2 on fresh 100,000")
    ax.set_title(f"{title}; min_size=10, alpha=0.01", fontsize=9)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "overfitting_vs_n.png", dpi=150)

(OUT / "summary.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
