"""Phase V1: boosted multicalibration with learned groups, next to every other calibrator.
Fit on calib, early stop on val_tune, scored on `eval_role` (val_tune during development, test in V8)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calib import metrics as M
from calib.base_model import VARIANTS
from calib.calibrators.boosted import group_from_conditions
from calib.data import load_rows
from calib.features import prepare_matrix
from calib.groups import EXTENDED
from calib.rq_setup import load_base_preds
from calib.v2_eval import METHODS, choose_platt, fit_all, group_cols, mask, part, predict
from calib.v2_setup import load_features, prepare_v2

V1_CHOSEN = json.loads(Path("runs/phase10/chosen.json").read_text())


def discovered_groups(cal, X, xcols, calib_mask, rows, min_problems=40):
    """Distinct feature conditions of leaves in the first 5 rounds, kept if they cover >= 40 calib problems."""
    seen, out = set(), []
    for r, lv, conds, value, n in cal.leaf_rules(xcols):
        if not conds or conds in seen:
            continue
        seen.add(conds)
        g = group_from_conditions(X, xcols, conds)
        if rows.task_id[calib_mask & g].nunique() >= min_problems:
            out.append({"round": r, "level": lv, "conditions": [list(c) for c in conds], "leaf_value": value,
                        "leaf_rows": n})
    return out


def evaluate_starts(rows, G, X, xcols, starts, eval_role, out_dir, platt_versions=None):
    """Every calibrator on every starting score. Returns metric table, predictions, discovered groups, fits."""
    out_dir.mkdir(parents=True, exist_ok=True)
    platt_versions = dict(platt_versions or {})
    table, preds, discovered, fits = [], {}, {}, {}
    for model in ["qwen3", "gpt-oss"]:
        cols = group_cols(rows, G, model)                       # calibrator groups: no difficulty
        hand_cols = group_cols(rows, G, model, with_difficulty=True)  # measured groups: all kept hand groups
        for start in starts:
            fit = part(rows, G, X, model, "calib", start, cols)
            tune = part(rows, G, X, model, "val_tune", start, cols)
            ev = part(rows, G, X, model, eval_role, start, cols)
            key = f"{model}/{start}"
            if key not in platt_versions:
                platt_versions[key] = V1_CHOSEN["platt_version"].get(key) or choose_platt(fit, tune)
            cals = fit_all(fit, tune, platt_versions[key], V1_CHOSEN["ighb_alpha"], V1_CHOSEN["iglb_epsilon"])
            fits[(model, start)] = cals
            groups = discovered_groups(cals["bmc"], X, xcols, mask(rows, model, "calib"), rows)
            discovered[key] = groups
            D = np.column_stack([group_from_conditions(X[ev["mask"]], xcols, [tuple(c) for c in g["conditions"]])
                                 for g in groups]) if groups else np.zeros((ev["mask"].sum(), 0), bool)
            H = G[ev["mask"]][:, hand_cols]
            for method in METHODS:
                q = predict(cals, method, ev)
                preds[(model, start, method)] = q
                rec = {"model": model, "start": start, "method": method, **M.summary(q, ev["y"]),
                       "max_gasce_hand": M.max_gasce(q, ev["y"], H),
                       "max_gasce_discovered": M.max_gasce(q, ev["y"], D) if D.shape[1] else np.nan,
                       "max_gasce_union": M.max_gasce(q, ev["y"], np.column_stack([H, D]))}
                if method in ["ighb", "iglb"]:
                    rec["steps"] = len(cals[method].rules)
                if method == "bmc":
                    rec["steps"] = len(cals["bmc"].rounds)
                table.append(rec)
    table = pd.DataFrame(table)
    table.to_csv(out_dir / "calibrators.csv", index=False)
    (out_dir / "discovered_groups.json").write_text(json.dumps(discovered, indent=1))
    (out_dir / "platt_versions.json").write_text(json.dumps(platt_versions, indent=1))
    return table, preds, discovered, fits, platt_versions


def load_v2(include_test=False, extra_preds=None):
    """Rows with v2 roles, base model predictions (v1 B0-B3 plus any v2 ones), groups, and the B2 feature matrix."""
    rows = load_rows(include_test=include_test)
    base = load_base_preds()
    if extra_preds is not None:
        key = ["model", "task_id", "sample_idx"]
        base = base.merge(extra_preds, on=key, how="left") if include_test is False else extra_preds
    rows, G = prepare_v2(rows, base)
    feats = load_features(rows)
    X, xcols = prepare_matrix(feats, VARIANTS["B2"][1])
    return rows, G, X, xcols


if __name__ == "__main__":
    rows, G, X, xcols = load_v2()
    table, _, discovered, _, _ = evaluate_starts(rows, G, X, xcols, ["avg_prob", "B2"], "val_tune",
                                                 Path("runs/v2/v1_boosted"))
    cols = ["model", "start", "method", "brier", "bss", "ece", "auroc", "max_gasce_hand", "max_gasce_discovered",
            "steps"]
    print(table[cols].round(4).to_string(index=False))
    for k, groups in discovered.items():
        print(k, len(groups))
        for g in groups[:8]:
            print("   ", g["round"], g["level"], g["conditions"], round(g["leaf_value"], 3), g["leaf_rows"])
