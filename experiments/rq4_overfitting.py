"""Phase 12 / RQ4: as the calibration set shrinks, does IGHB overfit, and do the guarded variants help?
Subsample calib problems, fit, measure held out Brier change relative to the starting score."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calib import metrics as M
from calib.calibrators.ighb import IGHB
from calib.calibrators.ighb_protected import HoldoutIGHB, NoiseAwareIGHB
from calib.calibrators.iglb import IGLB
from calib.calibrators.platt import Platt
from calib.data import load_rows
from calib.rq_setup import arrays, group_sets, load_base_preds, prepare, sel

SIZES, REPEATS, LONG_ALPHA = [25, 50, 100, 200, 211], 20, 1e-4


def split_problems(ids, rng, frac=0.7):
    ids = rng.permutation(ids)
    cut = int(round(frac * len(ids)))
    return set(ids[:cut]), set(ids[cut:])


def run(rows, G, chosen, role, out):
    """Subsampling study scored on `role` rows. Writes csv and plot to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for model in ["qwen3", "gpt-oss"]:
        cols = group_sets(rows, G, model)[0]["no_difficulty"]
        calib_mask = sel(rows, model, ["calib_fit", "calib_stop"])
        calib_ids = np.array(sorted(rows.task_id[calib_mask].unique()))
        for start in ["avg_prob", "B2"]:
            ev = arrays(rows, G, sel(rows, model, role), start, cols)
            base_brier = M.brier(ev["p"], ev["y"])
            for size in SIZES:
                for rep in range(REPEATS if size < len(calib_ids) else 1):
                    rng = np.random.default_rng(1000 * size + rep)
                    ids = rng.choice(calib_ids, size, replace=False)
                    fit_ids, hold_ids = split_problems(ids, rng)
                    in_ids = calib_mask & rows.task_id.isin(ids).to_numpy()
                    in_fit = calib_mask & rows.task_id.isin(fit_ids).to_numpy()
                    in_hold = calib_mask & rows.task_id.isin(hold_ids).to_numpy()
                    a, f, h = (arrays(rows, G, m, start, cols) for m in [in_ids, in_fit, in_hold])
                    platt_version = chosen["platt_version"][f"{model}/{start}"]
                    fits = {
                        "Platt": Platt(platt_version).fit(a["p"], a["y"]),
                        "IGHB (alpha 0.003)": IGHB("code", alpha=chosen["ighb_alpha"]).fit(a["p"], a["y"], a["G"]),
                        "IGHB long (alpha 1e-4)": IGHB("code", alpha=LONG_ALPHA, max_iter=300).fit(a["p"], a["y"], a["G"]),
                        "noise aware IGHB": NoiseAwareIGHB(alpha=LONG_ALPHA, max_iter=300).fit(a["p"], a["y"], a["G"], a["task"]),
                        "holdout IGHB": HoldoutIGHB(alpha=LONG_ALPHA, max_iter=300, seed=rep).fit(
                            f["p"], f["y"], f["G"], h["p"], h["y"], h["G"]),
                        "IGLB": IGLB("code", epsilon=chosen["iglb_epsilon"]).fit(f["p"], f["y"], f["G"], h["p"], h["y"], h["G"]),
                    }
                    for name, cal in fits.items():
                        q = cal.predict(ev["p"], ev["G"]) if name != "Platt" else cal.predict(ev["p"])
                        results.append({"model": model, "start": start, "size": size, "rep": rep, "method": name,
                                        "steps": len(getattr(cal, "rules", [])),
                                        "brier_change": M.brier(np.clip(q, 0, 1), ev["y"]) - base_brier})

    results = pd.DataFrame(results)
    results.to_csv(out / "rq4_subsample.csv", index=False)
    summary = results.groupby(["model", "start", "method", "size"]).agg(
        mean=("brier_change", "mean"), sd=("brier_change", "std"), steps=("steps", "mean")).reset_index()
    summary.to_csv(out / "rq4_summary.csv", index=False)
    print(summary.pivot_table(index=["model", "start", "method"], columns="size", values="mean").round(4).to_string())
    print(summary.pivot_table(index=["model", "start", "method"], columns="size", values="steps").round(1).to_string())

    methods = results.method.unique()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for i, model in enumerate(["qwen3", "gpt-oss"]):
        for j, start in enumerate(["avg_prob", "B2"]):
            ax = axes[i, j]
            s = summary[(summary.model == model) & (summary.start == start)]
            for k, name in enumerate(methods):
                d = s[s.method == name]
                ax.errorbar(d["size"] * (1 + 0.03 * (k - 2)), d["mean"], yerr=d["sd"], marker="o", ms=4, capsize=2,
                            label=name)
            ax.axhline(0, color="gray", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("calibration problems")
            ax.set_ylabel(f"{role} Brier change vs starting score")
            ax.set_title(f"{model}, start {start} (mean +- sd over 20 subsamples)", fontsize=9)
            if start == "avg_prob":
                ax.set_ylim(s["mean"].min() - 0.02, 0.02)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "brier_change_vs_size.png", dpi=130)
    return summary


if __name__ == "__main__":
    rows, G = prepare(load_rows(), load_base_preds())
    chosen = json.loads(Path("runs/phase10/chosen.json").read_text())
    run(rows, G, chosen, "validation", Path("runs/phase12"))
