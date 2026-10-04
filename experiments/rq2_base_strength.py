"""Phase 10 / RQ2, validation only: do multicalibration gains over Platt shrink as the starting score gets stronger?
Calibrators fit on calib_fit; IGLB early stops on calib_stop; everything scored on official validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calib import metrics as M
from calib.bootstrap import paired_diff
from calib.data import load_rows
from calib.groups import EXTENDED
from calib.pipeline import calibrate
from calib.rq_setup import STARTS, arrays, group_sets, load_base_preds, prepare, sel

OUT = Path("runs/phase10")
OUT.mkdir(parents=True, exist_ok=True)
rows, G = prepare(load_rows(), load_base_preds())
print(rows.groupby(["model", "role"]).size().to_dict())

ALPHAS = [0.05, 0.01, 0.003, 0.001, 0.0003]
EPSILONS = [0.005, 0.01, 0.02, 0.05]
PLATT_VERSIONS = ["code", "paper", "logit"]


def setting_runs(model, start, gs_name, cols):
    fit = arrays(rows, G, sel(rows, model, "calib_fit") | sel(rows, model, "calib_stop"), start, cols)
    fit_only = arrays(rows, G, sel(rows, model, "calib_fit"), start, cols)
    stop = arrays(rows, G, sel(rows, model, "calib_stop"), start, cols)
    va = arrays(rows, G, sel(rows, model, "validation"), start, cols)
    return fit, fit_only, stop, va


# 1. Hyperparameter search on validation.
search, sets_used = [], {}
for model in ["qwen3", "gpt-oss"]:
    sets, kept = group_sets(rows, G, model)
    sets_used[model] = {k: [EXTENDED[i] for i in v] for k, v in sets.items()}
    for start in STARTS:
        for gs_name, cols in sets.items():
            fit, fit_only, stop, va = setting_runs(model, start, gs_name, cols)
            for a in ALPHAS:
                q, cal = calibrate("ighb", "code", fit, None, va, alpha=a)
                search.append({"model": model, "start": start, "groups": gs_name, "method": "ighb", "param": a,
                               "val_bss": M.brier_skill(q, va["y"]), "steps": len(cal.rules)})
            for e in EPSILONS:
                q, cal = calibrate("iglb", "code", fit_only, stop, va, epsilon=e)
                search.append({"model": model, "start": start, "groups": gs_name, "method": "iglb", "param": e,
                               "val_bss": M.brier_skill(q, va["y"]), "steps": len(cal.rules)})
            for v in PLATT_VERSIONS:
                q, _ = calibrate("platt", v, fit, None, va)
                search.append({"model": model, "start": start, "groups": gs_name, "method": "platt", "param": v,
                               "val_bss": M.brier_skill(q, va["y"]), "steps": None})
search = pd.DataFrame(search)
search.to_csv(OUT / "hyperparameter_search.csv", index=False)
best_alpha = search[search.method == "ighb"].groupby("param").val_bss.mean().idxmax()
eps_scores = search[search.method == "iglb"].groupby("param").val_bss.mean()
# Ties (within 1e-4) go to the Campos default 0.01.
best_eps = 0.01 if eps_scores.max() - eps_scores[0.01] < 1e-4 else eps_scores.idxmax()
platt_pick = (search[(search.method == "platt") & (search.groups == "no_difficulty")]
              .sort_values("val_bss").groupby(["model", "start"]).param.last().to_dict())
print("mean val BSS by alpha:", search[search.method == "ighb"].groupby("param").val_bss.mean().round(4).to_dict())
print("mean val BSS by epsilon:", search[search.method == "iglb"].groupby("param").val_bss.mean().round(4).to_dict())
print("chosen alpha", best_alpha, "epsilon", best_eps, "platt", platt_pick)

# 2. Main table at the chosen settings, with bootstrap differences vs Platt.
table, diffs = [], []
for model in ["qwen3", "gpt-oss"]:
    sets, _ = group_sets(rows, G, model)
    for start in STARTS:
        for gs_name, cols in sets.items():
            fit, fit_only, stop, va = setting_runs(model, start, gs_name, cols)
            preds = {
                "uncalibrated": calibrate("uncalibrated", None, fit, None, va)[0],
                "platt": calibrate("platt", platt_pick[(model, start)], fit, None, va)[0],
                "hb": calibrate("hb", None, fit, None, va)[0],
                "linr": calibrate("linr", "code", fit, None, va)[0],
                "ighb": calibrate("ighb", "code", fit, None, va, alpha=best_alpha)[0],
                "iglb": calibrate("iglb", "code", fit_only, stop, va, epsilon=best_eps)[0],
            }
            for method, q in preds.items():
                table.append({"model": model, "start": start, "groups": gs_name, "method": method,
                              **M.summary(np.clip(q, 0, 1), va["y"], va["G"])})
            for method in ["hb", "linr", "ighb", "iglb"]:
                d = paired_diff(M.brier_skill, preds[method], preds["platt"], va["y"], va["task"])
                diffs.append({"model": model, "start": start, "groups": gs_name, "method": method,
                              "bss_minus_platt": d["diff"], "lo": d["lo"], "hi": d["hi"]})
table, diffs = pd.DataFrame(table), pd.DataFrame(diffs)
table.to_csv(OUT / "rq2_validation.csv", index=False)
diffs.to_csv(OUT / "rq2_validation_diffs.csv", index=False)
(OUT / "chosen.json").write_text(json.dumps({"ighb_alpha": best_alpha, "iglb_epsilon": best_eps,
                                             "platt_version": {f"{m}/{s}": v for (m, s), v in platt_pick.items()},
                                             "group_sets": sets_used}, indent=1))
show = table[table.groups == "no_difficulty"].pivot_table(index=["model", "start"], columns="method", values="bss")
print(show[["uncalibrated", "platt", "hb", "linr", "ighb", "iglb"]].round(3).to_string())
print(diffs.round(3).to_string(index=False))
