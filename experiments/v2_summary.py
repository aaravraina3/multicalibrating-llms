"""Phase V9: write v2_summary.md for one run folder (dev: runs/v2/dev_val_tune, final: runs/<hash>)."""

import json
import sys
from pathlib import Path

import pandas as pd

SHAP = Path("runs/v2/v3_shap")
MODELS = {"qwen3": "Qwen3 Coder", "gpt-oss": "GPT OSS"}


def fmt(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df, cols, names=None, d=3):
    names = names or cols
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], d) if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build(run, role):
    out = [f"# v2 summary ({role})", "",
           f"Run folder `{run}`. Calibrators fit on calib; IGLB and boosted multicalibration early stop on val_tune; "
           "conformal thresholds chosen on val_conformal. SHAP diagnostics are on val_tune by design (safeguard 4) "
           "and explain B4 before calibration (safeguard 8). Intervals: 95% task clustered bootstrap.", ""]
    out += ["## Split sizes", "", "```", Path("runs/v2/split_sizes.txt").read_text().rstrip(), "```", ""]

    t = pd.read_csv(run / "calibrators" / "calibrators.csv")
    t["bss_ci"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(t.bss, t.bss_lo, t.bss_hi)]
    for model, name in MODELS.items():
        out += [f"## Base predictor x calibrator, {name}", ""]
        out.append(md_table(t[t.model == model], ["start", "method", "brier", "bss_ci", "ece", "max_gasce_hand", "auroc"],
                            ["start", "calibrator", "Brier", "BSS [95%]", "ECE", "max gASCE (hand)", "AUROC"]))
        out.append("")

    disc = json.loads((run / "calibrators" / "discovered_groups.json").read_text())
    out += ["## Boosted multicalibration: discovered groups and worst group error", ""]
    for k, gs in disc.items():
        conds = "; ".join(" & ".join(f"{c[0]} {c[1]} {c[2]:.3g}" for c in g["conditions"]) +
                          f" (leaf {g['leaf_value']:+.2f})" for g in gs) or "none"
        out.append(f"* {k}: {conds}")
    bmc = t[t.method.isin(["uncalibrated", "bmc"])]
    out += ["", md_table(bmc, ["model", "start", "method", "max_gasce_hand", "max_gasce_discovered"],
                         ["model", "start", "method", "max gASCE hand", "max gASCE discovered"]), ""]

    d = pd.read_csv(run / "base_predictor_differences.csv")
    d["ci"] = [f"{a:+.4f} [{lo:+.4f}, {hi:+.4f}]" for a, lo, hi in zip(d["diff"], d.lo, d.hi)]
    out += ["## Base predictor differences (uncalibrated)", "", md_table(d, ["model", "comparison", "metric", "ci"],
                                                                         ["model", "comparison", "metric", "diff [95%]"]), ""]

    out += ["## SHAP (B4, val_tune, probability scale unless noted)", ""]
    for model, name in MODELS.items():
        s = json.loads((SHAP / model / "summary.json").read_text())
        imp = pd.read_csv(SHAP / model / "importance_features_probability_scale.csv", index_col=0)
        out.append(f"### {name}")
        out.append("")
        out.append("Top features (mean |SHAP|, seed 0): " +
                   ", ".join(f"{f} {v:.3f}" for f, v in imp.seed0.sort_values(ascending=False).head(8).items()))
        out.append("")
        out.append("Block importance: " + "; ".join(f"{b['block']} {b['mean_abs_shap_prob']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]"
                                                    for b in s["block_importance"]))
        out.append("")
        out.append(f"Seed stability (Spearman): {', '.join(f'{k} {v:.3f}' for k, v in s['seed_spearman'].items())}. "
                   f"In the top 10 for all 3 seeds: {', '.join(s['top10_in_all_seeds'])}.")
        out.append("")
        out.append(f"Noise check: noise ranks {s['noise_rank']} of {len(imp) + 1}; real features below it: "
                   f"{', '.join(s['features_below_noise']) or 'none'}.")
        out.append("")
        out.append("Stable interaction pairs (top 5 in all 3 seeds, log odds scale): val_tune "
                   f"{s['stable_interaction_pairs_val_tune']}; out of fold base_train {s['stable_interaction_pairs_oof_base_train']}.")
        out.append("")
        out.append("SHAP groups (frozen): " + "; ".join(f"{g['name']} ({g['base_train_problems']} problems)" for g in s["shap_groups"]))
        out.append("")
        abl = pd.DataFrame(s["ablation"])
        out.append(md_table(abl, ["dropped", "val_tune_log_loss", "delta_log_loss", "val_tune_auroc", "retuned_val_tune_log_loss"],
                            ["block dropped", "val_tune log loss", "change", "AUROC", "log loss, retuned"], d=4))
        out.append("")
        out.append("Worst slices of uncalibrated B4 (block SHAP profile vs overall, probability scale):")
        out.append("")
        for w in s["worst_slices"]:
            prof = ", ".join(f"{b} {v:+.3f}" for b, v in w["block_profile_vs_overall"].items())
            feats = ", ".join(f"{f} {dv:+.3f} (slice mean {a:.2f} vs {b:.2f})" for f, dv, a, b in w["top_features_vs_overall"])
            out.append(f"* {w['group']}, bin {w['bin']}: predicted {w['mean_p']:.2f}, passed {w['pass_rate']:.2f}, "
                       f"{w['rows']} rows. Blocks: {prof}. Top features: {feats}.")
        out.append("")
    g = pd.read_csv(run / "group_set_comparison.csv")
    out += [f"### Three group sets, IGLB on B4 ({role})", "",
            md_table(g, ["model", "group_set", "groups", "brier", "max_gasce_union", "slices_checked", "slices_over_2se", "iglb_patches"],
                     ["model", "group set", "groups", "Brier", "max gASCE union", "slices", "slices > 2 SE", "IGLB patches"], d=4), ""]

    m = pd.read_csv(run / "murphy.csv")
    m["d_rel"] = [fmt(a, 4) + (f" [{lo:.4f}, {hi:.4f}]" if pd.notna(lo) else "") for a, lo, hi in
                  zip(m.get("d_reliability_vs_platt"), m.get("d_reliability_lo"), m.get("d_reliability_hi"))]
    m["d_res"] = [fmt(a, 4) + (f" [{lo:.4f}, {hi:.4f}]" if pd.notna(lo) else "") for a, lo, hi in
                  zip(m.get("d_resolution_vs_platt"), m.get("d_resolution_lo"), m.get("d_resolution_hi"))]
    out += ["## Murphy decomposition", "", md_table(m, ["model", "start", "method", "brier", "reliability", "resolution",
                                                         "uncertainty", "gap", "d_rel", "d_res"],
                                                     ["model", "start", "method", "Brier", "reliability", "resolution",
                                                      "uncertainty", "gap", "change in reliability vs Platt", "change in resolution vs Platt"], d=4), ""]
    h = pd.read_csv(run / "holm_family.csv")
    out += ["## Holm family (28 comparisons, Brier differences)", "",
            md_table(h, ["model", "comparison", "brier_diff", "lo", "hi", "p", "p_holm", "significant_holm"],
                     ["model", "comparison", "Brier diff", "lo", "hi", "p", "Holm p", "significant"], d=4), ""]

    c = pd.read_csv(run / "conformal.csv")
    sim = pd.read_csv(run / "conformal_simulation.csv")
    out += ["## Conformal risk control (Qwen3 Coder primary, GPT OSS fallback)", "",
            md_table(c, ["calibrator", "target", "threshold", "realized_risk", "escalation", "system_pass", "oracle_pass_matched",
                         "regret", "p4_raw_regret"],
                     ["calibrator", "target", "threshold", "realized risk", "escalation", "system pass", "oracle at same budget",
                      "regret", "regret, rank based raw P4"]), "",
            "Simulation check (500 fresh calibration draws, known pass probabilities):", "",
            md_table(sim, ["target", "mean_realized_risk", "share_of_draws_above_target"],
                     ["target", "mean realized risk", "share of draws above target"]), ""]
    e = pd.read_csv(run / "routing_explanation.csv")
    top = json.loads((run / "routing_explanation_top_features.json").read_text())
    out += [f"Routing explanation ({top['calibrator']} scores, target 0.10, threshold {top['threshold']:.3f}; "
            f"{top['escalated']} of {top['rows']} val_conformal answers escalated). Mean B4 block SHAP, escalated vs accepted "
            "(explains B4 before calibration):", "",
            md_table(e, ["block", "escalated_mean", "accepted_mean", "difference"], d=4), "",
            "Top features for escalated answers: " + ", ".join(f"{f} {v:+.3f}" for f, v in top["top_features_escalated_mean_shap"]), ""]

    s = pd.read_csv(run / "shift" / "shift.csv")
    cols = ["base", "fit_on", "evaluated_on", "calibrator", "bss", "ece", "risk_at_0.1", "coverage_at_0.1"]
    out += ["## Distribution shift (optional V7)", "", md_table(s, cols, ["base", "fit on", "evaluated on", "calibrator", "BSS",
                                                                          "ECE", "risk at target 0.10", "coverage at 0.10"]), ""]
    return "\n".join(out)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    role = sys.argv[2] if len(sys.argv) > 2 else "val_tune"
    (run / "v2_summary.md").write_text(build(run, role))
    print("wrote", run / "v2_summary.md")
