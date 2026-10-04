"""Phase 7: every method, code and paper versions, replication groups.
Fit on official train, IGLB early stops on validation, evaluate on validation (IGLB is optimistic here)."""

from pathlib import Path

import pandas as pd

from calib import metrics as M
from calib.calibrators.ighb import IGHB
from calib.data import load_rows
from calib.groups import REPLICATION, build_groups
from calib.pipeline import METHODS, calibrate, part
from calib.scores import add_scores

OUT = Path("runs/phase7")
OUT.mkdir(parents=True, exist_ok=True)
rows = add_scores(load_rows())
G, _ = build_groups(rows, REPLICATION)

table = []
for model in ["qwen3", "gpt-oss"]:
    tr = part(rows, G, "avg_prob", (rows.model == model) & (rows.split == "train"))
    va = part(rows, G, "avg_prob", (rows.model == model) & (rows.split == "validation"))
    for version in ["code", "paper"]:
        for method in METHODS:
            if method == "uncalibrated" and version == "paper":
                continue
            q, cal = calibrate(method, version, tr, va, va)
            steps = len(cal.rules) if hasattr(cal, "rules") else None
            table.append({"model": model, "method": method, "version": version, "steps": steps,
                          **M.summary(q, va["y"], va["G"]),
                          "ece_grid": M.ece(q, va["y"], round_to_grid=True),
                          "max_gasce_grid": M.max_gasce(q, va["y"], va["G"], round_to_grid=True)})
            print(table[-1]["model"], method, version, round(table[-1]["bss"], 3), steps)
table = pd.DataFrame(table)
table.to_csv(OUT / "validation.csv", index=False)
print(table.round(3).to_string(index=False))

# IGHB stopping rule: Campos alpha = 1/M = 0.05 vs tighter. Train BSS shows under- vs overfitting.
sweep = []
for model in ["qwen3", "gpt-oss"]:
    tr = part(rows, G, "avg_prob", (rows.model == model) & (rows.split == "train"))
    va = part(rows, G, "avg_prob", (rows.model == model) & (rows.split == "validation"))
    for alpha in [0.05, 0.02, 0.01, 0.003, 0.001, 0.0003]:
        m = IGHB("code", alpha=alpha).fit(tr["p"], tr["y"], tr["G"])
        sweep.append({"model": model, "alpha": alpha, "steps": len(m.rules),
                      "train_bss": M.brier_skill(m.fitted_, tr["y"]),
                      "val_bss": M.brier_skill(m.predict(va["p"], va["G"]), va["y"])})
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "ighb_alpha_sweep.csv", index=False)
print(sweep.round(3).to_string(index=False))
