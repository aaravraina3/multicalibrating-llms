"""Phase 11 / RQ3: does multicalibration change routing decisions beyond Platt?
Calibrators fit on calib. P4 cutoffs come from `cutoff_role` predictions; q_B from the fallback's calib pass rate."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calib import routing as R
from calib.bootstrap import resample_indices
from calib.data import load_rows
from calib.groups import EXTENDED
from calib.rq_setup import calibrated, group_sets, load_base_preds, prepare, sel

STARTS = ["avg_prob", "B2"]
CALIBRATORS = ["uncalibrated", "platt", "iglb"]


def policy_rows(pair, q_b, cutoffs, q_b_given_a):
    """Every policy's use_b decisions for one calibrated pair. Returns {name: (use_b, both_run)}."""
    p_a, p_b = pair.q_a.to_numpy(), pair.q_b.to_numpy()
    y_a, y_b = pair.y_a.to_numpy(), pair.y_b.to_numpy()
    out = {"P0 always primary": (R.always_primary(p_a), False),
           "P1 always fallback": (R.always_fallback(p_a), False),
           "P2 oracle": (R.oracle(y_a, y_b), False)}
    for r in R.RATES:
        out[f"P4 lowest {int(r * 100)}%"] = (R.lowest_fraction(p_a, cutoffs[r]), False)
    for c_b in R.C_B_SCENARIOS:
        out[f"P5 threshold c_B={c_b:g}"] = (R.cost_threshold(p_a, q_b, c_b), False)
        out[f"P5c conditional c_B={c_b:g}"] = (R.cost_threshold(p_a, q_b_given_a(p_a), c_b), False)
    out["P6 pick higher"] = (R.pick_higher(p_a, p_b), True)
    return out


def scenario_cost(name):
    return float(name.split("c_B=")[1]) if "c_B=" in name else None


def run(rows, G, chosen, eval_role, cutoff_role, out_dir, n_boot=2000):
    out_dir.mkdir(parents=True, exist_ok=True)
    table, boot, sanity, curves = [], [], [], {}
    decisions = {}
    for primary, fallback in [("qwen3", "gpt-oss"), ("gpt-oss", "qwen3")]:
        cols_a = group_sets(rows, G, primary)[0]["no_difficulty"]
        cols_b = group_sets(rows, G, fallback)[0]["no_difficulty"]
        fb_calib = sel(rows, fallback, "calib_fit") | sel(rows, fallback, "calib_stop")
        q_b = rows.y.to_numpy()[fb_calib].mean()
        for start in STARTS:
            for method in CALIBRATORS:
                a = calibrated(rows, G, primary, start, method, eval_role, cols_a, chosen)
                b = calibrated(rows, G, fallback, start, method, eval_role, cols_b, chosen)
                a_cut = calibrated(rows, G, primary, start, method, cutoff_role, cols_a, chosen)
                pair = a.merge(b, on=["task_id", "sample_idx"], suffixes=("_a", "_b"), validate="1:1")
                # "lower" keeps the cutoff an actual data value, so any order preserving map gives identical decisions.
                cutoffs = {r: np.quantile(a_cut.q.to_numpy(), r, method="lower") for r in R.RATES}
                y_a, y_b = pair.y_a.to_numpy(), pair.y_b.to_numpy()
                # q_B given A's confidence, fit on calib pairs (A's calibrated score is in sample there).
                a_cal = calibrated(rows, G, primary, start, method, ["calib_fit", "calib_stop"], cols_a, chosen)
                b_cal = rows.loc[fb_calib, ["task_id", "sample_idx", "y"]]
                cal_pair = a_cal.merge(b_cal, on=["task_id", "sample_idx"], suffixes=("_a", "_b"), validate="1:1")
                q_b_given_a = R.fallback_given_primary(cal_pair.q.to_numpy(), cal_pair.y_b.to_numpy())
                pols = policy_rows(pair, q_b, cutoffs, q_b_given_a)
                key = (primary, start, method)
                decisions[key] = (pair, pols)
                for name, (use_b, both) in pols.items():
                    for c_b in ([scenario_cost(name)] if scenario_cost(name) else R.C_B_SCENARIOS):
                        res = R.evaluate(use_b, y_a, y_b, c_b, both_run=both)
                        res["regret"] = R.oracle_pass_at(y_a, y_b, res["escalation"]) - res["pass_rate"]
                        table.append({"primary": primary, "start": start, "calibrator": method, "policy": name,
                                      "c_b": c_b, **res})
                # random escalation, in expectation
                for r in R.RATES:
                    for c_b in R.C_B_SCENARIOS:
                        pr = (1 - r) * y_a.mean() + r * y_b.mean()
                        table.append({"primary": primary, "start": start, "calibrator": method,
                                      "policy": f"P3 random {int(r * 100)}%", "c_b": c_b, "pass_rate": pr,
                                      "escalation": r, "accepted_error": 1 - y_a.mean(),
                                      "compute_cost": R.C_A + c_b * r,
                                      "total_cost": R.C_A + c_b * r + R.LOSS_WRONG * (1 - pr),
                                      "regret": R.oracle_pass_at(y_a, y_b, r) - pr})
                # tradeoff curve for the rank based policy
                p_a = pair.q_a.to_numpy()
                order = np.argsort(p_a, kind="stable")
                xs, ys = [], []
                for k in range(0, len(p_a) + 1, max(1, len(p_a) // 50)):
                    use_b = np.zeros(len(p_a), bool)
                    use_b[order[:k]] = True
                    xs.append(use_b.mean())
                    ys.append(np.where(use_b, y_b, y_a).mean())
                curves[key] = (xs, ys, [R.oracle_pass_at(y_a, y_b, x) for x in xs], y_a.mean(), y_b.mean())
            # Sanity: Platt must give the same rank based decisions as uncalibrated.
            for r in R.RATES:
                u = decisions[(primary, start, "uncalibrated")][1][f"P4 lowest {int(r * 100)}%"][0]
                p = decisions[(primary, start, "platt")][1][f"P4 lowest {int(r * 100)}%"][0]
                sanity.append({"primary": primary, "start": start, "rate": r, "decisions_differ": int((u != p).sum())})
            # Bootstrap: IGLB vs Platt, Platt vs uncalibrated, on P4 20% pass, P5 total cost, P6 pass.
            pair = decisions[(primary, start, "platt")][0]
            tasks = pair.task_id.to_numpy()
            comparisons = [("iglb", "platt"), ("platt", "uncalibrated"), ("iglb", "uncalibrated")]
            targets = [("P4 lowest 20%", None, "pass_rate"), ("P6 pick higher", None, "pass_rate")] + \
                      [(f"P5 threshold c_B={c:g}", c, "total_cost") for c in R.C_B_SCENARIOS] + \
                      [(f"P5c conditional c_B={c:g}", c, "total_cost") for c in R.C_B_SCENARIOS]
            for m1, m2 in comparisons:
                (pair1, pols1), (pair2, pols2) = decisions[(primary, start, m1)], decisions[(primary, start, m2)]
                assert (pair1.task_id.to_numpy() == tasks).all() and (pair2.task_id.to_numpy() == tasks).all()
                for pol, c_b, metric in targets:
                    c_b = c_b or 1.0
                    u1, b1 = pols1[pol]
                    u2, b2 = pols2[pol]
                    ya, yb = pair1.y_a.to_numpy(), pair1.y_b.to_numpy()
                    f = lambda idx, u, b: R.evaluate(u[idx], ya[idx], yb[idx], c_b, both_run=b)[metric]
                    full = f(np.arange(len(ya)), u1, b1) - f(np.arange(len(ya)), u2, b2)
                    diffs = np.array([f(i, u1, b1) - f(i, u2, b2) for i in resample_indices(tasks, n_boot)])
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    boot.append({"primary": primary, "start": start, "compare": f"{m1} - {m2}", "policy": pol,
                                 "metric": metric, "diff": full, "lo": lo, "hi": hi})
    table, boot, sanity = pd.DataFrame(table), pd.DataFrame(boot), pd.DataFrame(sanity)
    table.to_csv(out_dir / "routing.csv", index=False)
    boot.to_csv(out_dir / "routing_bootstrap.csv", index=False)
    sanity.to_csv(out_dir / "platt_rank_sanity.csv", index=False)

    # Per group: P4 20% and P6 pass rate by the primary's groups.
    per_group = []
    for (primary, start, method), (pair, pols) in decisions.items():
        m = sel(rows, primary, eval_role)
        gdf = pd.DataFrame(G[m], columns=EXTENDED)
        gdf["task_id"], gdf["sample_idx"] = rows.task_id.to_numpy()[m], rows.sample_idx.to_numpy()[m]
        gp = pair.merge(gdf, on=["task_id", "sample_idx"], how="left")
        for pol in ["P4 lowest 20%", "P6 pick higher"]:
            use_b = pols[pol][0]
            final = np.where(use_b, gp.y_b, gp.y_a)
            for g in EXTENDED:
                mask = gp[g].to_numpy(bool)
                per_group.append({"primary": primary, "start": start, "calibrator": method, "policy": pol,
                                  "group": g, "rows": int(mask.sum()), "pass_rate": final[mask].mean(),
                                  "escalation": use_b[mask].mean()})
    pd.DataFrame(per_group).to_csv(out_dir / "routing_per_group.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for i, primary in enumerate(["qwen3", "gpt-oss"]):
        for j, start in enumerate(STARTS):
            ax = axes[i, j]
            for method in CALIBRATORS:
                xs, ys, oracle, pa, pb = curves[(primary, start, method)]
                ax.plot(xs, ys, label=method, lw=1.5 if method != "platt" else 3, alpha=0.8,
                        ls="--" if method == "platt" else "-")
            ax.plot(xs, oracle, color="black", lw=1, label="oracle")
            ax.plot([0, 1], [pa, pb], color="gray", ls=":", label="random")
            ax.set_xlabel("escalation rate")
            ax.set_ylabel("system pass rate")
            ax.set_title(f"primary {primary}, start {start} (escalate lowest confidence first)", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_curves.png", dpi=130)
    return table, boot, sanity


if __name__ == "__main__":
    rows, G = prepare(load_rows(), load_base_preds())
    chosen = json.loads(Path("runs/phase10/chosen.json").read_text())
    table, boot, sanity = run(rows, G, chosen, "validation", "validation", Path("runs/phase11"))
    print(sanity.to_string(index=False))
    key = table.policy.str.match("P[0-2]") | table.policy.isin(["P4 lowest 20%", "P6 pick higher"]) | \
        table.policy.str.startswith("P5")
    show = table[key & ((table.c_b == 2.0) | table.policy.str.startswith("P5"))]
    print(show[["primary", "start", "calibrator", "policy", "c_b", "pass_rate", "escalation", "total_cost", "regret"]]
          .round(3).to_string(index=False))
    print(boot.round(4).to_string(index=False))
