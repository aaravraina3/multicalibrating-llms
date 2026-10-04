"""Phase 4: how good is each raw starting score on validation, before any calibration."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from calib import metrics as M
from calib.data import load_rows
from calib.plots import reliability_diagram
from calib.scores import add_scores

OUT = Path("runs/phase4")
OUT.mkdir(parents=True, exist_ok=True)

rows = add_scores(load_rows())
rows["n_tokens"] = rows.logprobs.map(len)
print("min tokens per output:", rows.n_tokens.min())
val = rows[rows.split == "validation"]

table = []
for subset, data in [("all rows", val), ("code span present", val[val.code_end > val.code_start])]:
    for model in ["qwen3", "gpt-oss"]:
        d = data[data.model == model]
        for score in ["avg_prob", "code_prob_campos", "tail_prob"]:
            p, y = d[score].to_numpy(), d.y.to_numpy()
            table.append({"subset": subset, "model": model, "score": score, "n": len(d),
                          "mean_p": p.mean(), "pass_rate": y.mean(), **M.summary(p, y)})
table = pd.DataFrame(table)
table.to_csv(OUT / "uncalibrated_validation.csv", index=False)
print(table.round(3).to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
for i, model in enumerate(["qwen3", "gpt-oss"]):
    d = val[val.model == model]
    for j, score in enumerate(["avg_prob", "code_prob_campos", "tail_prob"]):
        p, y = d[score].to_numpy(), d.y.to_numpy()
        reliability_diagram(p, y, axes[i, j], title=f"{model} {score}, ECE {M.ece(p, y):.3f}")
fig.tight_layout()
fig.savefig(OUT / "reliability_validation.png", dpi=130)
