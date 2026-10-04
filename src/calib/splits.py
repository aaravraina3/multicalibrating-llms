"""Deterministic split of official train problems into base_train (60%) and calib (40%)."""

import hashlib
import json

SALT = "calib-routing-2026"


def split_train(task_ids, frac_base=0.6, salt=SALT):
    """Sort problems by sha256(salt + id); first 60% go to base_train. Same answer on every machine, no labels."""
    ids = sorted(set(task_ids), key=lambda t: hashlib.sha256((salt + t).encode()).hexdigest())
    cut = round(frac_base * len(ids))
    return {"base_train": sorted(ids[:cut]), "calib": sorted(ids[cut:])}


def save(splits, path):
    with open(path, "w") as f:
        json.dump({"salt": SALT, **splits}, f, indent=1)


def assign(rows, splits):
    """Column with 'base_train', 'calib', or the official split name for non-train rows."""
    lookup = {t: name for name in ["base_train", "calib"] for t in splits[name]}
    return [lookup.get(t, s) if s == "train" else s for t, s in zip(rows.task_id, rows.split)]
