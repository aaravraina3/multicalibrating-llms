"""Phase 5: Platt, HB, isotonic. Fit on official train, evaluate on validation, starting score avg_prob."""

from pathlib import Path

import pandas as pd

from calib import metrics as M
from calib.calibrators.histogram import HistogramBinning, Isotonic
from calib.calibrators.platt import Platt
from calib.data import load_rows
from calib.scores import add_scores

OUT = Path("runs/phase5")
OUT.mkdir(parents=True, exist_ok=True)
rows = add_scores(load_rows())

methods = {
    "uncalibrated": None,
    "platt_code": lambda: Platt("code"),
    "platt_paper": lambda: Platt("paper"),
    "platt_logit": lambda: Platt("logit"),
    "hb": lambda: HistogramBinning(),
    "isotonic (ours)": lambda: Isotonic(),
}
table = []
for model in ["qwen3", "gpt-oss"]:
    tr = rows[(rows.model == model) & (rows.split == "train")]
    va = rows[(rows.model == model) & (rows.split == "validation")]
    for name, make in methods.items():
        if make is None:
            q, extra = va.avg_prob.to_numpy(), {}
        else:
            cal = make().fit(tr.avg_prob.to_numpy(), tr.y.to_numpy())
            q = cal.predict(va.avg_prob.to_numpy())
            extra = {"a": cal.a, "b": cal.b} if isinstance(cal, Platt) else {}
        y = va.y.to_numpy()
        table.append({"model": model, "method": name, **M.summary(q, y),
                      "ece_grid": M.ece(q, y, round_to_grid=True), **extra})
table = pd.DataFrame(table)
table.to_csv(OUT / "validation.csv", index=False)
print(table.round(3).to_string(index=False))
