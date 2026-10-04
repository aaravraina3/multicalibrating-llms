"""Shared setup for RQ2 and RQ3: roles, starting scores, group sets."""

import numpy as np
import pandas as pd

from calib.data import DATA_DIR
from calib.groups import EXTENDED, build_groups, keep_groups
from calib.scores import add_scores
from calib.splits import assign, split_train

STARTS = ["avg_prob", "B1", "B2", "B3"]
DIFFICULTY = ["comp_easy", "comp_medium", "comp_hard"]


def prepare(rows, base_preds):
    """Adds roles (base_train, calib_fit, calib_stop, validation, test), scores, base model predictions,
    and the extended group matrix. Group thresholds come from official train."""
    rows = add_scores(rows)
    splits = split_train(rows[rows.split == "train"].task_id)
    rows["role"] = assign(rows, splits)
    inner = split_train(splits["calib"], frac_base=0.7)  # IGLB early stopping holdout inside calib
    stop = set(inner["calib"])
    rows.loc[(rows.role == "calib") & rows.task_id.isin(stop), "role"] = "calib_stop"
    rows.loc[rows.role == "calib", "role"] = "calib_fit"
    key = ["model", "task_id", "sample_idx"]
    rows = rows.merge(base_preds[key + ["B0", "B1", "B2", "B3"]], on=key, how="left", validate="1:1")
    G, _ = build_groups(rows, EXTENDED)
    return rows, G


def group_sets(rows, G, model):
    """Column indices for the group sets of one model. Groups with < 40 train problems are dropped."""
    m = (rows.model == model).to_numpy()
    kept = keep_groups(G[m], rows[m], EXTENDED)
    with_diff = [EXTENDED.index(n) for n in kept]
    no_diff = [EXTENDED.index(n) for n in kept if n not in DIFFICULTY]
    return {"no_difficulty": no_diff, "with_difficulty": with_diff}, kept


def load_base_preds():
    return pd.read_parquet(DATA_DIR / "base_preds.parquet")


def sel(rows, model, role):
    return ((rows.model == model) & (rows.role == role)).to_numpy()


def arrays(rows, G, mask, score, cols):
    return {"p": np.clip(rows[score].to_numpy()[mask], 0, 1), "y": rows.y.to_numpy()[mask],
            "G": G[mask][:, cols], "task": rows.task_id.to_numpy()[mask]}
