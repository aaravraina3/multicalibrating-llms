"""Phase 10 / RQ2: do multicalibration gains over Platt shrink as the starting score gets stronger?
Calibrators fit on calib (IGLB: calib_fit, early stop on calib_stop). Settings are searched on validation only;
`evaluate_table` is reused unchanged on test by experiments/final.py."""

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

ALPHAS = [0.05, 0.01, 0.003, 0.001, 0.0003]
EPSILONS = [0.005, 0.01, 0.02, 0.05]
PLATT_VERSIONS = ["code", "paper", "logit"]


def setting_arrays(rows, G, model, start, cols, role, eval_mask=None):
    fit = arrays(rows, G, sel(rows, model, ["calib_fit", "calib_stop"]), start, cols)
    fit_only = arrays(rows, G, sel(rows, model, "calib_fit"), start, cols)
    stop = arrays(rows, G, sel(rows, model, "calib_stop"), start, cols)
    m = sel(rows, model, role) if eval_mask is None else sel(rows, model, role) & eval_mask
    return fit, fit_only, stop, arrays(rows, G, m, start, cols)


def search_settings(rows, G):
    """Validation search for IGHB alpha, IGLB epsilon, and the Platt input per starting score."""
    search, sets_used = [], {}
    for model in ["qwen3", "gpt-oss"]:
        sets, _ = group_sets(rows, G, model)
        sets_used[model] = {k: [EXTENDED[i] for i in v] for k, v in sets.items()}
        for start in STARTS:
            for gs_name, cols in sets.items():
                fit, fit_only, stop, va = setting_arrays(rows, G, model, start, cols, "validation")
                base = {"model": model, "start": start, "groups": gs_name}
                for a in ALPHAS:
                    q, cal = calibrate("ighb", "code", fit, None, va, alpha=a)
                    search.append({**base, "method": "ighb", "param": a, "val_bss": M.brier_skill(q, va["y"]),
                                   "steps": len(cal.rules)})
                for e in EPSILONS:
                    q, cal = calibrate("iglb", "code", fit_only, stop, va, epsilon=e)
                    search.append({**base, "method": "iglb", "param": e, "val_bss": M.brier_skill(q, va["y"]),
                                   "steps": len(cal.rules)})
                for v in PLATT_VERSIONS:
                    q, _ = calibrate("platt", v, fit, None, va)
                    search.append({**base, "method": "platt", "param": v, "val_bss": M.brier_skill(q, va["y"]),
                                   "steps": None})
    search = pd.DataFrame(search)
    best_alpha = search[search.method == "ighb"].groupby("param").val_bss.mean().idxmax()
    eps_scores = search[search.method == "iglb"].groupby("param").val_bss.mean()
    # Ties (within 1e-4) go to the Campos default 0.01.
    best_eps = 0.01 if eps_scores.max() - eps_scores[0.01] < 1e-4 else eps_scores.idxmax()
    platt_pick = (search[(search.method == "platt") & (search.groups == "no_difficulty")]
                  .sort_values("val_bss").groupby(["model", "start"]).param.last().to_dict())
    chosen = {"ighb_alpha": float(best_alpha), "iglb_epsilon": float(best_eps),
              "platt_version": {f"{m}/{s}": v for (m, s), v in platt_pick.items()}, "group_sets": sets_used}
    return search, chosen


def evaluate_table(rows, G, chosen, role, eval_mask=None, n_boot=2000):
    """Every method at the chosen settings, scored on `role` rows, with bootstrap differences vs Platt."""
    table, diffs = [], []
    for model in ["qwen3", "gpt-oss"]:
        sets, _ = group_sets(rows, G, model)
        for start in STARTS:
            for gs_name, cols in sets.items():
                fit, fit_only, stop, ev = setting_arrays(rows, G, model, start, cols, role, eval_mask)
                preds = {
                    "uncalibrated": calibrate("uncalibrated", None, fit, None, ev)[0],
                    "platt": calibrate("platt", chosen["platt_version"][f"{model}/{start}"], fit, None, ev)[0],
                    "hb": calibrate("hb", None, fit, None, ev)[0],
                    "linr": calibrate("linr", "code", fit, None, ev)[0],
                    "ighb": calibrate("ighb", "code", fit, None, ev, alpha=chosen["ighb_alpha"])[0],
                    "iglb": calibrate("iglb", "code", fit_only, stop, ev, epsilon=chosen["iglb_epsilon"])[0],
                }
                preds = {k: np.clip(v, 0, 1) for k, v in preds.items()}
                for method, q in preds.items():
                    table.append({"model": model, "start": start, "groups": gs_name, "method": method,
                                  **M.summary(q, ev["y"], ev["G"])})
                for method in ["hb", "linr", "ighb", "iglb"]:
                    d = paired_diff(M.brier_skill, preds[method], preds["platt"], ev["y"], ev["task"], n_boot)
                    diffs.append({"model": model, "start": start, "groups": gs_name, "method": method,
                                  "bss_minus_platt": d["diff"], "lo": d["lo"], "hi": d["hi"]})
    return pd.DataFrame(table), pd.DataFrame(diffs)


if __name__ == "__main__":
    OUT = Path("runs/phase10")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, G = prepare(load_rows(), load_base_preds())
    search, chosen = search_settings(rows, G)
    search.to_csv(OUT / "hyperparameter_search.csv", index=False)
    (OUT / "chosen.json").write_text(json.dumps(chosen, indent=1))
    print("chosen:", chosen["ighb_alpha"], chosen["iglb_epsilon"], chosen["platt_version"])
    table, diffs = evaluate_table(rows, G, chosen, "validation")
    table.to_csv(OUT / "rq2_validation.csv", index=False)
    diffs.to_csv(OUT / "rq2_validation_diffs.csv", index=False)
    show = table[table.groups == "no_difficulty"].pivot_table(index=["model", "start"], columns="method", values="bss")
    print(show[["uncalibrated", "platt", "hb", "linr", "ighb", "iglb"]].round(3).to_string())
