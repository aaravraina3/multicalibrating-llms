"""Shared setup for RQ2 and RQ3: roles, starting scores, group sets."""

import numpy as np
import pandas as pd

from calib.data import DATA_DIR
from calib.groups import EXTENDED, build_groups, keep_groups
from calib.pipeline import calibrate
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
    """role: one role name or a list of them."""
    roles = [role] if isinstance(role, str) else role
    return ((rows.model == model) & rows.role.isin(roles)).to_numpy()


def arrays(rows, G, mask, score, cols):
    return {"p": np.clip(rows[score].to_numpy()[mask], 0, 1), "y": rows.y.to_numpy()[mask],
            "G": G[mask][:, cols], "task": rows.task_id.to_numpy()[mask]}


def calibrated(rows, G, model, start, method, eval_role, cols, chosen):
    """Calibrator fit on calib (IGLB: calib_fit, early stop on calib_stop), applied to eval_role rows.
    Returns predictions keyed by (task_id, sample_idx)."""

    fit = arrays(rows, G, sel(rows, model, "calib_fit") | sel(rows, model, "calib_stop"), start, cols)
    fit_only = arrays(rows, G, sel(rows, model, "calib_fit"), start, cols)
    stop = arrays(rows, G, sel(rows, model, "calib_stop"), start, cols)
    ev_mask = sel(rows, model, eval_role)
    ev = arrays(rows, G, ev_mask, start, cols)
    if method == "platt":
        q, _ = calibrate("platt", chosen["platt_version"][f"{model}/{start}"], fit, None, ev)
    elif method == "iglb":
        q, _ = calibrate("iglb", "code", fit_only, stop, ev, epsilon=chosen["iglb_epsilon"])
    elif method == "ighb":
        q, _ = calibrate("ighb", "code", fit, None, ev, alpha=chosen["ighb_alpha"])
    else:
        q, _ = calibrate(method, "code", fit, None, ev)
    out = rows.loc[ev_mask, ["task_id", "sample_idx", "y"]].copy()
    out["q"] = np.clip(q, 0, 1)
    return out


def base_preds_with_test():
    """Dev base predictions plus test predictions from base models refit on base_train with the frozen
    Phase 9 hyperparameters. Same computation as experiments/final.py; cached after the first call."""
    import json
    from pathlib import Path

    from calib.base_model import VARIANTS, fit_fixed
    from calib.data import load_rows
    from calib.features import build_features, prepare_matrix

    cache = DATA_DIR / "base_preds_with_test.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    splits = json.loads(Path("runs/phase9/splits.json").read_text())
    hyper = json.loads(Path("runs/phase9/chosen_hyperparameters.json").read_text())
    dev, base_preds = load_rows(), load_base_preds()
    all_rows = load_rows(include_test=True)
    test = all_rows[all_rows.split == "test"]
    feats_dev, feats_test = pd.read_parquet(DATA_DIR / "features.parquet"), build_features(test)
    dev["role"] = assign(dev, splits)
    extra = test[["task_id", "sample_idx", "model", "y"]].assign(role="test")
    for model in ["qwen3", "gpt-oss"]:
        tr = ((dev.model == model) & (dev.role == "base_train")).to_numpy()
        m_te = (test.model == model).to_numpy()
        for name, (kind, blocks) in VARIANTS.items():
            X_dev, _ = prepare_matrix(feats_dev, blocks)
            X_te, _ = prepare_matrix(feats_test, blocks)
            est = fit_fixed(kind, tuple(hyper[f"{model}/{name}"]["params"]), X_dev[tr], dev.y.to_numpy()[tr])
            extra.loc[m_te, name] = est.predict_proba(X_te[m_te])[:, 1]
    out = pd.concat([base_preds, extra], ignore_index=True)
    out.to_parquet(cache)
    return out
