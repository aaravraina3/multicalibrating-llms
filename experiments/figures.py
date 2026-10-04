"""Phase 14 figures from the final run's settings. Recomputes test predictions deterministically; no tuning."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calib import metrics as M
from calib.data import load_rows
from calib.groups import EXTENDED
from calib.pipeline import calibrate
from calib.plots import reliability_diagram
from calib.rq_setup import arrays, base_preds_with_test, group_sets, prepare, sel

RUN = Path("runs/8f1de29")
OUT = RUN / "figures"
OUT.mkdir(parents=True, exist_ok=True)
chosen = json.loads(Path("runs/phase10/chosen.json").read_text())

# Rebuild rows with test base predictions exactly as final.py did.


rows, G = prepare(load_rows(include_test=True), base_preds_with_test())
NAMES = {"qwen3": "Qwen3 Coder", "gpt-oss": "GPT OSS"}


def preds_for(model, start):
    cols = group_sets(rows, G, model)[0]["no_difficulty"]
    fit = arrays(rows, G, sel(rows, model, ["calib_fit", "calib_stop"]), start, cols)
    fit_only = arrays(rows, G, sel(rows, model, "calib_fit"), start, cols)
    stop = arrays(rows, G, sel(rows, model, "calib_stop"), start, cols)
    te_mask = sel(rows, model, "test")
    te = arrays(rows, G, te_mask, start, cols)
    out = {"uncalibrated": te["p"],
           "Platt": calibrate("platt", chosen["platt_version"][f"{model}/{start}"], fit, None, te)[0],
           "IGLB": calibrate("iglb", "code", fit_only, stop, te, epsilon=chosen["iglb_epsilon"])[0]}
    return {k: np.clip(v, 0, 1) for k, v in out.items()}, te["y"], G[te_mask]


cache = {(m, s): preds_for(m, s) for m in NAMES for s in ["avg_prob", "B2"]}

# Figure: reliability, avg_prob uncalibrated / avg_prob IGLB / B2 Platt.
fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
for i, model in enumerate(NAMES):
    for j, (start, method) in enumerate([("avg_prob", "uncalibrated"), ("avg_prob", "IGLB"), ("B2", "Platt")]):
        preds, y, _ = cache[(model, start)]
        q = preds[method]
        reliability_diagram(q, y, axes[i, j], title=f"{NAMES[model]}: {start}, {method}\n"
                                                    f"BSS {M.brier_skill(q, y):.3f}, ECE {M.ece(q, y):.3f}")
fig.tight_layout()
fig.savefig(OUT / "reliability.png", dpi=140)

# Figure: group calibration, mean prediction vs pass rate per group, Platt vs IGLB.
fig, axes = plt.subplots(2, 2, figsize=(9, 8.6))
for i, model in enumerate(NAMES):
    for j, start in enumerate(["avg_prob", "B2"]):
        ax = axes[i, j]
        preds, y, Gt = cache[(model, start)]
        for method, marker in [("Platt", "o"), ("IGLB", "s")]:
            xs, ys = [], []
            for g, name in enumerate(EXTENDED):
                m = Gt[:, g]
                if m.sum() >= 50 and name != "all":
                    xs.append(preds[method][m].mean())
                    ys.append(y[m].mean())
            err = np.mean(np.abs(np.array(xs) - np.array(ys)))
            ax.scatter(xs, ys, marker=marker, s=40, alpha=0.75, label=f"{method} (mean |gap| {err:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability in group")
        ax.set_ylabel("pass rate in group")
        ax.set_title(f"{NAMES[model]}, start {start}", fontsize=9)
        ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "group_calibration.png", dpi=140)

# Figure: IGHB stopping rule (validation, Phase 7).
sweep = pd.read_csv("runs/phase7/ighb_alpha_sweep.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for k, model in enumerate(NAMES):
    s = sweep[sweep.model == model]
    ax.plot(s.alpha, s.train_bss, marker="o", color=f"C{k}", ls="--", label=f"{NAMES[model]} train")
    ax.plot(s.alpha, s.val_bss, marker="o", color=f"C{k}", label=f"{NAMES[model]} validation")
ax.axvline(0.05, color="gray", lw=1, ls=":")
ax.text(0.05, ax.get_ylim()[0] + 0.02, " Campos alpha = 1/M", fontsize=8)
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("IGHB stopping threshold alpha (smaller = more steps)")
ax.set_ylabel("BSS")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ighb_alpha.png", dpi=140)
print("wrote", sorted(p.name for p in OUT.iterdir()))
