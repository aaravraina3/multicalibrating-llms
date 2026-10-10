"""Runs every v2 analysis on one evaluation split. Development: `--role val_tune`. Phase V8: `--role test`, once,
on the commit tagged v2-final, writing to runs/<git hash>/."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from calib import metrics as M
from calib.base_model import VARIANTS
from calib.bootstrap import interval
from calib.data import DATA_DIR, load_rows
from calib.features import prepare_matrix
from calib.rq_setup import base_preds_with_test, load_base_preds
from calib.shap_tools import background
from calib.v2_eval import METHODS, group_cols, mask, part, predict as cal_predict
from calib.v2_setup import load_features, prepare_v2, v2_preds_path
from calib.xgb_model import make_xgb
from experiments import v2_conformal, v2_murphy_holm, v2_shap, v2_shift
from experiments.v2_boosted import evaluate_starts

STARTS = ["avg_prob", "B2", "B4", "B6", "B7"]
BEST = json.loads(Path("runs/v2/v2_xgboost/best_config.json").read_text())


def v2_predictions(rows, X):
    """B4 refit deterministically; B6/B7 from data/v2_traj_preds_all.parquet, written by
    experiments/v2_predict_traj.py in its own process (torch can't share a process with xgboost and shap here)."""
    out = pd.DataFrame(index=rows.index, columns=["B4"], dtype=float)
    for model in ["qwen3", "gpt-oss"]:
        m, bt = (rows.model == model).to_numpy(), mask(rows, model, "base_train")
        b4 = make_xgb(BEST[model]["params"], 0).fit(X[bt], rows.y.to_numpy()[bt])
        out.loc[m, "B4"] = b4.predict_proba(X[m])[:, 1]
    key = ["model", "task_id", "sample_idx"]
    traj = rows[key].merge(pd.read_parquet(DATA_DIR / "v2_traj_preds_all.parquet"), on=key, how="left", validate="1:1")
    assert traj[["B6", "B7"]].notna().all().all(), "run experiments.v2_predict_traj --with-test first"
    out["B6"], out["B7"] = traj.B6.to_numpy(), traj.B7.to_numpy()
    return out


def load_all(include_test):
    rows = load_rows(include_test=include_test)
    base = base_preds_with_test() if include_test else load_base_preds()
    key = ["model", "task_id", "sample_idx"]
    rows, G = prepare_v2(rows, base[key + ["B0", "B1", "B2", "B3"]])
    X, xcols = prepare_matrix(load_features(rows), VARIANTS["B2"][1])
    if include_test:
        v2 = v2_predictions(rows, X)
        dev = pd.read_parquet(v2_preds_path())
        check = rows[key].join(v2).merge(dev, on=key, suffixes=("", "_saved"))
        for c in ["B4", "B6", "B7"]:
            assert np.allclose(check[c], check[f"{c}_saved"], atol=1e-5), c  # frozen models reproduce dev predictions
        rows = rows.join(v2)
    else:
        rows = rows.merge(pd.read_parquet(v2_preds_path()), on=key, how="left", validate="1:1")
    return rows, G, X, xcols


def run(role, out, protocol):
    out.mkdir(parents=True, exist_ok=True)
    rows, G, X, xcols = load_all(include_test=(role == "test"))
    y, task = rows.y.to_numpy(), rows.task_id.to_numpy()

    # Base predictor x calibrator table with bootstrap intervals on BSS.
    table, preds, discovered, fits, _ = evaluate_starts(rows, G, X, xcols, STARTS, role, out / "calibrators",
                                                        platt_versions=protocol["platt_versions"])
    lo, hi = [], []
    for _, r in table.iterrows():
        m = mask(rows, r.model, role)
        iv = interval(M.brier_skill, preds[(r.model, r.start, r.method)], y[m], task[m])
        lo.append(iv["lo"])
        hi.append(iv["hi"])
    table["bss_lo"], table["bss_hi"] = lo, hi
    table.to_csv(out / "calibrators" / "calibrators.csv", index=False)

    # Base predictor differences with intervals.
    from calib.bootstrap import paired_diff
    diffs = []
    for model in ["qwen3", "gpt-oss"]:
        m = mask(rows, model, role)
        for a, b in [("B4", "B2"), ("B6", "B4"), ("B7", "B4"), ("B7", "B6")]:
            for metric_name, fn in [("brier", M.brier), ("log_loss", M.log_loss)]:
                d = paired_diff(fn, preds[(model, a, "uncalibrated")], preds[(model, b, "uncalibrated")], y[m], task[m])
                diffs.append({"model": model, "comparison": f"{a} - {b}", "metric": metric_name, **d})
    pd.DataFrame(diffs).to_csv(out / "base_predictor_differences.csv", index=False)

    # V6: Murphy decomposition and the Holm family.
    fam = {m: ["avg_prob", "B2", "B4", protocol["transformer"][m]] for m in ["qwen3", "gpt-oss"]}
    starts_used = sorted({s for v in fam.values() for s in v}, key=STARTS.index)
    murphy = v2_murphy_holm.murphy_table(rows, preds, starts_used, METHODS, role)
    murphy.to_csv(out / "murphy.csv", index=False)
    holm = v2_murphy_holm.holm_family(rows, preds, fam, role)
    holm.to_csv(out / "holm_family.csv", index=False)

    # V5: conformal risk control on the frozen best base predictor (primary Qwen3).
    best = protocol["conformal_base"]
    cals = fits[("qwen3", best)]
    cols = group_cols(rows, G, "qwen3")
    scores = {c: {r: cal_predict(cals, c, part(rows, G, X, "qwen3", r, best, cols))
                  for r in ["val_conformal", role]} for c in v2_conformal.CALIBRATORS}
    raw = {r: part(rows, G, X, "qwen3", r, best, cols)["p"] for r in ["val_conformal", role]}
    conf = v2_conformal.run(rows, scores, role, raw)
    conf.to_csv(out / "conformal.csv", index=False)
    v2_conformal.simulation_check().to_csv(out / "conformal_simulation.csv", index=False)
    # Routing explanation at target 0.10 with the conformal threshold of the best calibrated score.
    b4 = make_xgb(BEST["qwen3"]["params"], 0).fit(X[mask(rows, "qwen3", "base_train")], y[mask(rows, "qwen3", "base_train")])
    bg = background(X[mask(rows, "qwen3", "base_train")], 200, seed=0)
    pick = protocol["conformal_explain_calibrator"]
    t = conf[(conf.calibrator == pick) & (conf.target == 0.10)].threshold.iloc[0]
    expl, top, n_esc, n_all = v2_conformal.explain_escalations(rows, X, xcols, b4, bg, scores[pick]["val_conformal"], t)
    expl.to_csv(out / "routing_explanation.csv", index=False)
    (out / "routing_explanation_top_features.json").write_text(json.dumps(
        {"calibrator": pick, "target": 0.10, "threshold": t, "escalated": n_esc, "rows": n_all,
         "top_features_escalated_mean_shap": top}, indent=1))

    # V3 step 5 on this split: the three group sets, using the frozen SHAP groups.
    comp = []
    for model in ["qwen3", "gpt-oss"]:
        comp.append(v2_shap.compare_group_sets(rows, G, X, xcols, model, role).assign(model=model))
    pd.concat(comp).to_csv(out / "group_set_comparison.csv", index=False)

    # V7: shift.
    shift, _ = v2_shift.run(rows, G, X, xcols, role, out / "shift")

    return {"table": table, "murphy": murphy, "holm": holm, "conformal": conf, "explanation": (expl, top, n_esc, n_all),
            "group_sets": pd.concat(comp), "shift": shift, "discovered": discovered, "diffs": pd.DataFrame(diffs)}


if __name__ == "__main__":
    role = sys.argv[sys.argv.index("--role") + 1] if "--role" in sys.argv else "val_tune"
    protocol = json.loads(Path("runs/v2/protocol_v2_frozen.json").read_text())
    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = Path("runs") / git_hash if role == "test" else Path("runs/v2/dev_val_tune")
    res = run(role, out, protocol)
    print("wrote", out)
