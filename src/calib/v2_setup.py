"""v2 data roles: base_train, calib, val_tune, val_conformal, test. Same assignment for both models."""

import json
from pathlib import Path

from calib.data import DATA_DIR
from calib.groups import EXTENDED, build_groups
from calib.scores import add_scores
from calib.splits import split_train

SALT_V2 = "calib-routing-v2-2026"
SPLITS_V1 = Path("runs/phase9/splits.json")
SPLITS_V2 = Path("runs/v2/splits_v2.json")


def make_splits_v2(validation_ids):
    """Half of official validation to val_tune, half to val_conformal, by sha256(salt + id)."""
    halves = split_train(validation_ids, frac_base=0.5, salt=SALT_V2)
    return {"salt": SALT_V2, "val_tune": halves["base_train"], "val_conformal": halves["calib"]}


def roles_v2(rows):
    """Role per row from the v1 train split and the v2 validation split."""
    v1 = json.loads(SPLITS_V1.read_text())
    v2 = json.loads(SPLITS_V2.read_text())
    lookup = {t: r for r in ["base_train", "calib"] for t in v1[r]}
    lookup.update({t: r for r in ["val_tune", "val_conformal"] for t in v2[r]})
    return [lookup[t] if s != "test" else "test" for t, s in zip(rows.task_id, rows.split)]


def prepare_v2(rows, base_preds=None):
    """Scores, v2 roles, optional base model predictions, and the extended group matrix (train thresholds)."""
    rows = add_scores(rows)
    rows["role"] = roles_v2(rows)
    if base_preds is not None:
        key = ["model", "task_id", "sample_idx"]
        cols = [c for c in base_preds.columns if c not in key + ["role", "y"]]
        rows = rows.merge(base_preds[key + cols], on=key, how="left", validate="1:1")
    G, _ = build_groups(rows, EXTENDED)
    return rows, G


def v2_preds_path():
    return DATA_DIR / "v2_base_preds.parquet"
