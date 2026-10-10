"""Phase V4 evaluation: B6 and B7 next to B0 to B4 on val_tune, B6 vs B7 choice, every calibrator on both."""

import json
from pathlib import Path

import pandas as pd

from calib import metrics as M
from calib.bootstrap import paired_diff
from calib.v2_eval import mask
from experiments.v2_boosted import evaluate_starts
from experiments.v2_run import load_all

OUT = Path("runs/v2/v4_transformer")
rows, G, X, xcols = load_all(include_test=False)
y, task = rows.y.to_numpy(), rows.task_id.to_numpy()

table, choice, diffs = [], {}, []
for model in ["qwen3", "gpt-oss"]:
    m = mask(rows, model, "val_tune")
    for start in ["avg_prob", "B0", "B1", "B2", "B3", "B4", "B6", "B7"]:
        table.append({"model": model, "start": start, **M.summary(rows[start].to_numpy()[m].clip(0, 1), y[m])})
    briers = {s: M.brier(rows[s].to_numpy()[m], y[m]) for s in ["B6", "B7"]}
    choice[model] = min(briers, key=briers.get)
    for a, b in [("B6", "B2"), ("B7", "B2"), ("B6", "B4"), ("B7", "B4")]:
        d = paired_diff(M.log_loss, rows[a].to_numpy()[m], rows[b].to_numpy()[m], y[m], task[m])
        diffs.append({"model": model, "comparison": f"{a} - {b}", "metric": "log_loss", **d})
table = pd.DataFrame(table)
table.to_csv(OUT / "base_predictors_val_tune.csv", index=False)
pd.DataFrame(diffs).to_csv(OUT / "differences_val_tune.csv", index=False)
(OUT / "b6_b7_choice.json").write_text(json.dumps(choice, indent=1))
print(table[["model", "start", "log_loss", "brier", "bss", "auroc", "ece"]].round(4).to_string(index=False))
print(pd.DataFrame(diffs).round(4).to_string(index=False))
print("choice by val_tune Brier:", choice)

cal, _, disc, _, pv = evaluate_starts(rows, G, X, xcols, ["B6", "B7"], "val_tune", OUT / "calibrators")
print(cal[["model", "start", "method", "brier", "bss", "ece", "auroc", "max_gasce_hand", "steps"]].round(4).to_string(index=False))
print("platt", pv)
