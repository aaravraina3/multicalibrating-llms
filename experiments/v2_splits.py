"""Phase V0.5: split official validation into val_tune and val_conformal, log sizes."""

import json

from calib.data import load_rows
from calib.v2_setup import SPLITS_V2, make_splits_v2, roles_v2

rows = load_rows()
SPLITS_V2.parent.mkdir(parents=True, exist_ok=True)
splits = make_splits_v2(rows[rows.split == "validation"].task_id)
SPLITS_V2.write_text(json.dumps(splits, indent=1))
rows["role"] = roles_v2(rows)
sizes = rows.groupby(["model", "role"]).agg(rows=("y", "size"), problems=("task_id", "nunique"),
                                             pass_rate=("y", "mean")).round(3)
print(sizes.to_string())
(SPLITS_V2.parent / "split_sizes.txt").write_text(sizes.to_string() + "\n")
