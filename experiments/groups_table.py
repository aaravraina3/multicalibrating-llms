"""Phase 6: group sizes in problems per split, thresholds from train only."""

import json
from pathlib import Path

import pandas as pd

from calib.data import load_rows
from calib.groups import EXTENDED, build_groups

OUT = Path("runs/phase6")
OUT.mkdir(parents=True, exist_ok=True)
rows = load_rows()
G, thresholds = build_groups(rows, EXTENDED)
(OUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
print(json.dumps(thresholds, indent=2))

table = []
for model in ["qwen3", "gpt-oss"]:
    for g, name in enumerate(EXTENDED):
        rec = {"model": model, "group": name}
        for split in ["train", "validation"]:
            m = (rows.model == model).to_numpy() & (rows.split == split).to_numpy() & G[:, g]
            rec[f"{split}_rows"] = int(m.sum())
            rec[f"{split}_problems"] = rows[m].task_id.nunique()
        tr = (rows.model == model).to_numpy() & (rows.split == "train").to_numpy() & G[:, g]
        rec["train_pass_rate"] = rows[tr].y.mean()
        rec["keep"] = rec["train_problems"] >= 40
        table.append(rec)
table = pd.DataFrame(table)
table.to_csv(OUT / "group_sizes.csv", index=False)
print(table.round(3).to_string(index=False))
