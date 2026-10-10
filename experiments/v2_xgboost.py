"""Phase V2: B4 = XGBoost with Optuna TPE search (100 trials) over GroupKFold(5) on base_train.
Refit the best config on all of base_train, then run every calibrator on top of B4."""

import json
from pathlib import Path

import pandas as pd

from calib import metrics as M
from calib.base_model import VARIANTS
from calib.data import load_rows
from calib.features import prepare_matrix
from calib.v2_eval import mask
from calib.v2_setup import load_features, prepare_v2, v2_preds_path
from calib.rq_setup import load_base_preds
from calib.xgb_model import SEED, make_xgb, search
from experiments.v2_boosted import evaluate_starts

OUT = Path("runs/v2/v2_xgboost")
OUT.mkdir(parents=True, exist_ok=True)

rows, G = prepare_v2(load_rows(), load_base_preds())
X, xcols = prepare_matrix(load_features(rows), VARIANTS["B2"][1])
y = rows.y.to_numpy()

best = {}
for model in ["qwen3", "gpt-oss"]:
    tr = mask(rows, model, "base_train")
    study = search(X[tr], y[tr], rows.task_id.to_numpy()[tr], n_trials=100, seed=SEED)
    best[model] = {"params": study.best_params, "cv_log_loss": study.best_value}
    study.trials_dataframe().to_csv(OUT / f"optuna_trials_{model}.csv", index=False)
    b4 = make_xgb(study.best_params, SEED).fit(X[tr], y[tr])
    m = (rows.model == model).to_numpy()
    rows.loc[m, "B4"] = b4.predict_proba(X[m])[:, 1]
    print(model, best[model])
(OUT / "best_config.json").write_text(json.dumps(best, indent=1))
key = ["model", "task_id", "sample_idx"]
rows[key + ["B4"]].to_parquet(v2_preds_path())

# Base predictors side by side on val_tune and on the full validation set.
table = []
for model in ["qwen3", "gpt-oss"]:
    for role in [["val_tune"], ["val_tune", "val_conformal"]]:
        m = mask(rows, model, role)
        for start in ["avg_prob", "B0", "B1", "B2", "B3", "B4"]:
            q = rows[start].to_numpy()[m].clip(0, 1)
            table.append({"model": model, "eval": "+".join(role), "start": start, **M.summary(q, y[m])})
table = pd.DataFrame(table)
table.to_csv(OUT / "base_predictors.csv", index=False)
print(table[["model", "eval", "start", "log_loss", "brier", "bss", "auroc", "ece"]].round(4).to_string(index=False))

# Every calibrator on top of B4, fit on calib.
cal_table, _, discovered, _, platt_versions = evaluate_starts(rows, G, X, xcols, ["B4"], "val_tune", OUT / "calibrators")
print(cal_table[["model", "start", "method", "brier", "bss", "ece", "auroc", "max_gasce_hand", "steps"]]
      .round(4).to_string(index=False))
print("platt", platt_versions, "discovered", {k: len(v) for k, v in discovered.items()})
