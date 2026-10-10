"""Phase V3: TreeSHAP interpretation of B4 with the ten safeguards, block ablations, SHAP proposed groups,
a three way group set comparison, and explanations of the worst miscalibrated slices. Development: val_tune."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from calib import metrics as M
from calib.base_model import VARIANTS
from calib.bootstrap import resample_indices
from calib.calibrators.boosted import group_from_conditions
from calib.calibrators.iglb import IGLB
from calib.calibrators.ighb_protected import cluster_se
from calib.data import load_rows
from calib.features import prepare_matrix
from calib.groups import EXTENDED
from calib.rq_setup import load_base_preds
from calib.shap_tools import background, block_index, block_shap, explain_probability, interactions_logodds, top_pairs
from calib.v2_eval import group_cols, mask
from calib.v2_setup import load_features, prepare_v2, v2_preds_path
from calib.xgb_model import grouped_cv_loss, make_xgb, oof_models, search

OUT = Path("runs/v2/v3_shap")
SEEDS = [0, 1, 2]
V1_CHOSEN = json.loads(Path("runs/phase10/chosen.json").read_text())
BEST = json.loads(Path("runs/v2/v2_xgboost/best_config.json").read_text())
DISCOVERED = json.loads(Path("runs/v2/v2_xgboost/calibrators/discovered_groups.json").read_text())


def load():
    rows = load_rows()
    rows, G = prepare_v2(rows, load_base_preds().merge(pd.read_parquet(v2_preds_path()),
                                                      on=["model", "task_id", "sample_idx"]))
    X, xcols = prepare_matrix(load_features(rows), VARIANTS["B2"][1])
    return rows, G, X, xcols


def shap_group_set(X_base, sv_oof, iv_oof_by_seed, xcols, task_base, min_problems=40):
    """Top 4 features by out of fold mean |SHAP| and the top 2 interaction pairs stable across seeds,
    each turned into a group on its higher impact side of the base_train median."""
    imp = np.abs(sv_oof).mean(0)
    med = np.median(X_base, axis=0)

    def split_point(j):
        """The base_train median, unless it leaves one side empty (binary or heavily tied features):
        then split between the median and the next distinct value."""
        x, thr = X_base[:, j], med[j]
        if (x > thr).sum() == 0 or (x <= thr).sum() == 0:
            vals = np.unique(x)
            k = int(np.searchsorted(vals, thr))
            lo_v, hi_v = (vals[k - 1], vals[k]) if k > 0 else (vals[0], vals[1])
            thr = (lo_v + hi_v) / 2
        return thr

    def side(j):
        thr = split_point(j)
        hi = X_base[:, j] > thr
        a = np.abs(sv_oof[hi, j]).mean() if hi.any() else 0
        b = np.abs(sv_oof[~hi, j]).mean() if (~hi).any() else 0
        return (">", float(thr)) if a >= b else ("<=", float(thr))

    tops = [set((a, b) for a, b, _ in top_pairs(iv, xcols, 5)) for iv in iv_oof_by_seed]
    stable = set.intersection(*tops)
    mean_iv = np.mean([np.abs(iv).mean(0) for iv in iv_oof_by_seed], axis=0)
    stable = sorted(stable, key=lambda ab: -mean_iv[xcols.index(ab[0]), xcols.index(ab[1])])
    groups = []
    for j in np.argsort(-imp)[:4]:
        op, thr = side(j)
        groups.append({"name": f"{xcols[j]} {op} {thr:.3g}", "conditions": [[xcols[j], op, thr]]})
    for a, b in stable[:2]:
        conds = [[f, *side(xcols.index(f))] for f in (a, b)]
        groups.append({"name": " & ".join(f"{f} {op} {t:.3g}" for f, op, t in conds), "conditions": conds})
    kept = []
    for g in groups:
        m = group_from_conditions(X_base, xcols, [tuple(c) for c in g["conditions"]])
        g["base_train_problems"] = int(len(np.unique(task_base[m])))
        if g["base_train_problems"] >= min_problems:
            kept.append(g)
    return kept, [list(p) for p in stable]


def membership(X, xcols, groups):
    if not groups:
        return np.zeros((len(X), 0), bool)
    return np.column_stack([group_from_conditions(X, xcols, [tuple(c) for c in g["conditions"]]) for g in groups])


def slices_over(q, y, task, U, names, n_bins=10, min_problems=5):
    """Group x confidence bin slices with at least 5 problems: gap, cluster SE, share, IGHB style weight."""
    idx = np.minimum((q * n_bins).astype(int), n_bins - 1)
    out = []
    for g in range(U.shape[1]):
        for b in range(n_bins):
            m = U[:, g] & (idx == b)
            if len(np.unique(task[m])) >= min_problems:
                gap = float((y[m] - q[m]).mean())
                out.append({"group": names[g], "bin": b, "rows": int(m.sum()), "problems": len(np.unique(task[m])),
                            "mean_p": float(q[m].mean()), "pass_rate": float(y[m].mean()), "gap": gap,
                            "se": cluster_se(y - q, task, m), "weight": m.mean() * gap ** 2, "mask": m})
    return out


def run_model(rows, G, X, xcols, model, out):
    out.mkdir(parents=True, exist_ok=True)
    y = rows.y.to_numpy()
    task = rows.task_id.to_numpy()
    params = BEST[model]["params"]
    bt, ca, vt = mask(rows, model, "base_train"), mask(rows, model, "calib"), mask(rows, model, "val_tune")
    bg = background(X[bt], 200, seed=0)
    blocks = block_index(xcols)
    res = {}

    # Safeguard 5: B4 under 3 seeds; probability scale interventional SHAP on val_tune (safeguards 2 to 4).
    svs, ivs, models = {}, {}, {}
    for s in SEEDS:
        models[s] = make_xgb(params, s).fit(X[bt], y[bt])
        svs[s], base_value = explain_probability(models[s], bg, X[vt])
        ivs[s] = interactions_logodds(models[s], X[vt])
    sv = svs[0]
    imp = pd.DataFrame({f"seed{s}": np.abs(svs[s]).mean(0) for s in SEEDS}, index=xcols)
    imp["mean"] = imp.mean(1)
    imp.sort_values("mean", ascending=False).to_csv(out / "importance_features_probability_scale.csv")
    rho = {f"{a}-{b}": float(spearmanr(imp[f"seed{a}"], imp[f"seed{b}"]).statistic)
           for a in SEEDS for b in SEEDS if a < b}
    top10 = [set(imp[f"seed{s}"].sort_values(ascending=False).index[:10]) for s in SEEDS]
    res["seed_spearman"] = rho
    res["top10_in_all_seeds"] = sorted(set.intersection(*top10), key=lambda f: -imp.loc[f, "mean"])

    # Safeguard 1 and 9: block importance with task clustered bootstrap intervals (1000 draws).
    bs = block_shap(sv, xcols)
    tv = task[vt]
    block_rows = []
    for b, v in bs.items():
        boots = [np.abs(v[i]).mean() for i in resample_indices(tv, 1000, seed=0)]
        block_rows.append({"block": b, "mean_abs_shap_prob": float(np.abs(v).mean()),
                           "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                           "sum_of_feature_importance": float(imp.loc[[xcols[i] for i in blocks[b]], "seed0"].sum())})
    block_df = pd.DataFrame(block_rows).sort_values("mean_abs_shap_prob", ascending=False)
    block_df.to_csv(out / "importance_blocks_probability_scale.csv", index=False)

    # Safeguard 6: interaction pairs in the top 5 for all 3 seeds (log odds scale, path dependent).
    tops = {s: top_pairs(ivs[s], xcols, 5) for s in SEEDS}
    stable = set.intersection(*[set((a, b) for a, b, _ in tops[s]) for s in SEEDS])
    res["interactions_top5_logodds"] = {f"seed{s}": tops[s] for s in SEEDS}
    res["stable_interaction_pairs_val_tune"] = sorted(stable)

    # Safeguard 10: noise feature check.
    rng = np.random.default_rng(123)
    noise = rng.normal(size=(len(X), 1))
    Xn, ncols = np.hstack([X, noise]), xcols + ["NOISE"]
    mn = make_xgb(params, 0).fit(Xn[bt], y[bt])
    svn, _ = explain_probability(mn, background(Xn[bt], 200, seed=0), Xn[vt])
    impn = pd.Series(np.abs(svn).mean(0), index=ncols).sort_values(ascending=False)
    res["noise_rank"] = int(list(impn.index).index("NOISE")) + 1
    res["features_below_noise"] = list(impn.index[impn.index.get_loc("NOISE") + 1:])
    impn.to_csv(out / "noise_check_importance.csv")

    # Plots (seed 0), labeled with the scale.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    imp["seed0"].sort_values().tail(15).plot.barh(ax=axes[0])
    axes[0].set_xlabel("mean |SHAP|, probability scale (points of pass probability)")
    axes[0].set_title(f"{model}: B4 feature importance on val_tune\n(explains B4 before calibration)", fontsize=9)
    bd = block_df.sort_values("mean_abs_shap_prob")
    axes[1].barh(bd.block, bd.mean_abs_shap_prob, xerr=[bd.mean_abs_shap_prob - bd.lo, bd.hi - bd.mean_abs_shap_prob])
    axes[1].set_xlabel("mean |block SHAP|, probability scale, 95% task clustered interval")
    axes[1].set_title(f"{model}: block importance (explains B4 before calibration)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "importance_bars.png", dpi=130)
    plt.close("all")
    shap.summary_plot(sv, X[vt], feature_names=xcols, max_display=15, show=False)
    plt.title(f"{model}: SHAP on probability scale, val_tune (explains B4 before calibration)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "beeswarm_probability_scale.png", dpi=130)
    plt.close("all")

    # Step 3: block ablation with the same config; optional 25 trial retune per ablated model.
    abl = []
    full_q = models[0].predict_proba(X[vt])[:, 1]
    abl.append({"dropped": "(none)", "val_tune_log_loss": M.log_loss(full_q, y[vt]), "val_tune_auroc": M.auroc(full_q, y[vt])})
    for b, idx in blocks.items():
        keep = [i for i in range(len(xcols)) if i not in idx]
        mb = make_xgb(params, 0).fit(X[bt][:, keep], y[bt])
        q = mb.predict_proba(X[vt][:, keep])[:, 1]
        study = search(X[bt][:, keep], y[bt], task[bt], n_trials=25, seed=0)
        qr = make_xgb(study.best_params, 0).fit(X[bt][:, keep], y[bt]).predict_proba(X[vt][:, keep])[:, 1]
        abl.append({"dropped": b, "val_tune_log_loss": M.log_loss(q, y[vt]), "val_tune_auroc": M.auroc(q, y[vt]),
                    "retuned_val_tune_log_loss": M.log_loss(qr, y[vt]), "retuned_val_tune_auroc": M.auroc(qr, y[vt])})
    abl = pd.DataFrame(abl)
    abl["delta_log_loss"] = abl.val_tune_log_loss - abl.val_tune_log_loss.iloc[0]
    abl["delta_auroc"] = abl.val_tune_auroc - abl.val_tune_auroc.iloc[0]
    abl.to_csv(out / "block_ablation.csv", index=False)

    # Step 4 and safeguard 7: SHAP groups from out of fold SHAP on base_train.
    Xb, tb, yb = X[bt], task[bt], y[bt]
    sv_oof = np.zeros_like(Xb)
    iv_oof = {s: np.zeros((len(Xb), len(xcols), len(xcols))) for s in SEEDS}
    for s in SEEDS:
        for va_idx, m_fold in oof_models(params, Xb, yb, tb, seed=s):
            if s == 0:
                tr_idx = np.setdiff1d(np.arange(len(Xb)), va_idx)
                sv_oof[va_idx], _ = explain_probability(m_fold, background(Xb[tr_idx], 200, seed=0), Xb[va_idx])
            iv_oof[s][va_idx] = interactions_logodds(m_fold, Xb[va_idx])
    shap_groups, stable_oof = shap_group_set(Xb, sv_oof, [iv_oof[s] for s in SEEDS], xcols, tb)
    res["stable_interaction_pairs_oof_base_train"] = stable_oof
    res["shap_groups"] = shap_groups
    (out / "shap_groups.json").write_text(json.dumps(shap_groups, indent=1))

    # Step 5: three group sets, IGLB on B4 (fit calib, stop val_tune), measured on the union.
    hand_cols = group_cols(rows, G, model)
    sets = {"hand": [EXTENDED[i] for i in hand_cols], "discovered": DISCOVERED.get(f"{model}/B4", []),
            "shap": shap_groups}
    U_parts, U_names = [G[:, hand_cols]], [EXTENDED[i] for i in hand_cols]
    U_parts.append(membership(X, xcols, sets["discovered"]))
    U_names += ["disc: " + " & ".join(f"{c[0]} {c[1]} {c[2]:.3g}" for c in g["conditions"]) for g in sets["discovered"]]
    U_parts.append(membership(X, xcols, shap_groups))
    U_names += ["shap: " + g["name"] for g in shap_groups]
    U = np.column_stack(U_parts)
    p = rows.B4.to_numpy()
    comp = []
    for name in ["none (uncalibrated B4)", "hand", "discovered", "shap"]:
        if name.startswith("none"):
            q = p[vt]
        else:
            if name == "hand":
                Gs = G[:, hand_cols]
            else:
                Gs = membership(X, xcols, sets[name])
            if Gs.shape[1] == 0:
                comp.append({"group_set": name, "groups": 0, "note": "no groups"})
                continue
            cal = IGLB("code", epsilon=V1_CHOSEN["iglb_epsilon"]).fit(p[ca], y[ca], Gs[ca], p[vt], y[vt], Gs[vt])
            q = np.clip(cal.predict(p[vt], Gs[vt]), 0, 1)
        sl = slices_over(q, y[vt], task[vt], U[vt], U_names)
        comp.append({"group_set": name, "groups": len(sets[name]) if name in sets else 0,
                     "val_tune_brier": M.brier(q, y[vt]), "max_gasce_union": M.max_gasce(q, y[vt], U[vt]),
                     "slices_checked": len(sl), "slices_over_2se": sum(abs(s["gap"]) > 2 * s["se"] for s in sl),
                     "iglb_patches": 0 if name.startswith("none") else len(cal.rules)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "group_set_comparison.csv", index=False)

    # Step 6: explain the 5 worst slices of uncalibrated B4 on val_tune with SHAP profiles.
    sl = sorted(slices_over(p[vt], y[vt], task[vt], U[vt], U_names), key=lambda s: -s["weight"])
    worst, picked = [], []
    for s in sl:
        # At least 30 rows, so a 5 row slice with a big but noisy gap can't win; and skip slices that mostly
        # repeat one already picked (e.g. "long code" and "all" at the same level).
        if s["rows"] < 30 or any((s["mask"] & pm).sum() / (s["mask"] | pm).sum() > 0.5 for pm in picked):
            continue
        picked.append(s["mask"])
        prof = sv[s["mask"]].mean(0) - sv.mean(0)
        top = np.argsort(-np.abs(prof))[:3]
        worst.append({k: v for k, v in s.items() if k != "mask"} |
                     {"block_profile_vs_overall": {b: float(v[s["mask"]].mean() - v.mean()) for b, v in bs.items()},
                      "top_features_vs_overall": [(xcols[j], float(prof[j]),
                                                   float(X[vt][s["mask"], j].mean()), float(X[vt][:, j].mean()))
                                                  for j in top]})
        if len(worst) == 5:
            break
    res["worst_slices"] = worst
    res["block_importance"] = block_df.to_dict("records")
    res["ablation"] = abl.to_dict("records")
    res["group_set_comparison"] = comp.to_dict("records")
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    return res, models[0], bg


if __name__ == "__main__":
    rows, G, X, xcols = load()
    for model in ["qwen3", "gpt-oss"]:
        res, _, _ = run_model(rows, G, X, xcols, model, OUT / model)
        print("==", model)
        print("blocks", [(b["block"], round(b["mean_abs_shap_prob"], 4), round(b["lo"], 4), round(b["hi"], 4))
                         for b in res["block_importance"]])
        print("spearman", res["seed_spearman"], "noise rank", res["noise_rank"], "below noise", res["features_below_noise"])
        print("stable pairs val_tune", res["stable_interaction_pairs_val_tune"], "oof", res["stable_interaction_pairs_oof_base_train"])
        print("shap groups", [(g["name"], g["base_train_problems"]) for g in res["shap_groups"]])
        print(pd.DataFrame(res["ablation"]).round(4).to_string(index=False))
        print(pd.DataFrame(res["group_set_comparison"]).round(4).to_string(index=False))
        for w in res["worst_slices"]:
            print(" worst:", w["group"], "bin", w["bin"], "p", round(w["mean_p"], 3), "pass", round(w["pass_rate"], 3),
                  "rows", w["rows"], {k: round(v, 3) for k, v in w["block_profile_vs_overall"].items()},
                  [(f, round(d, 3), round(a, 3), round(b, 3)) for f, d, a, b in w["top_features_vs_overall"]])
        sys.stdout.flush()


def compare_group_sets(rows, G, X, xcols, model, role, shap_dir=OUT):
    """Step 5 on any split with the frozen group sets: IGLB on B4 (fit calib, early stop val_tune) per set,
    measured on the union of hand, discovered, and SHAP groups."""
    y, task = rows.y.to_numpy(), rows.task_id.to_numpy()
    ca, vt, ev = mask(rows, model, "calib"), mask(rows, model, "val_tune"), mask(rows, model, role)
    hand_cols = group_cols(rows, G, model)
    shap_groups = json.loads((shap_dir / model / "shap_groups.json").read_text())
    disc = DISCOVERED.get(f"{model}/B4", [])
    U = np.column_stack([G[:, hand_cols], membership(X, xcols, disc), membership(X, xcols, shap_groups)])
    names = ([EXTENDED[i] for i in hand_cols] + ["disc: " + str(g["conditions"]) for g in disc]
             + ["shap: " + g["name"] for g in shap_groups])
    p = rows.B4.to_numpy()
    sets = {"hand": G[:, hand_cols], "discovered": membership(X, xcols, disc), "shap": membership(X, xcols, shap_groups)}
    out = []
    for name in ["none (uncalibrated B4)", "hand", "discovered", "shap"]:
        if name in sets and sets[name].shape[1] == 0:
            out.append({"group_set": name, "groups": 0, "note": "no groups"})
            continue
        if name in sets:
            Gs = sets[name]
            cal = IGLB("code", epsilon=V1_CHOSEN["iglb_epsilon"]).fit(p[ca], y[ca], Gs[ca], p[vt], y[vt], Gs[vt])
            q, patches = np.clip(cal.predict(p[ev], Gs[ev]), 0, 1), len(cal.rules)
        else:
            q, patches = p[ev], 0
        sl = slices_over(q, y[ev], task[ev], U[ev], names)
        out.append({"group_set": name, "groups": int(sets[name].shape[1]) if name in sets else 0,
                    "brier": M.brier(q, y[ev]), "max_gasce_union": M.max_gasce(q, y[ev], U[ev]),
                    "slices_checked": len(sl), "slices_over_2se": sum(abs(s["gap"]) > 2 * s["se"] for s in sl),
                    "iglb_patches": patches})
    return pd.DataFrame(out)
