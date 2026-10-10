"""Phase V4: B6 (transformer over the token confidence trajectory) and B7 (same plus the B2 features).
5 seeds each, early stopping on an inner base_train split by problem, probabilities averaged over seeds."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from calib import metrics as M
from calib.base_model import VARIANTS
from calib.data import DATA_DIR, load_rows
from calib.features import prepare_matrix
from calib.rq_setup import load_base_preds
from calib.splits import split_train
from calib.trajectory import Standardizer, build_tensors, predict, train_one
from calib.v2_eval import mask
from calib.v2_setup import load_features, prepare_v2, v2_preds_path

SEEDS = [0, 1, 2, 3, 4]
INNER_SALT = "calib-routing-v2-inner"
MODEL_DIR = DATA_DIR / "trajectory_models"


def tensors_for(rows, cache):
    if cache.exists():
        z = np.load(cache)
        return z["seq"], z["pad"], z["scalar"]
    seq, pad, scalar = build_tensors(rows)
    np.savez(cache, seq=seq, pad=pad, scalar=scalar)
    return seq, pad, scalar


def train_all(rows, X, seq, pad, scalar):
    """Fit B6 and B7 per model; returns predictions for every row and saves weights and scalers."""
    MODEL_DIR.mkdir(exist_ok=True)
    y = rows.y.to_numpy()
    preds, info = {"B6": np.full(len(rows), np.nan), "B7": np.full(len(rows), np.nan)}, {}
    for model in ["qwen3", "gpt-oss"]:
        bt = mask(rows, model, "base_train")
        inner = split_train(rows.task_id[bt].unique(), frac_base=0.8, salt=INNER_SALT)
        tr = np.flatnonzero(bt & rows.task_id.isin(inner["base_train"]).to_numpy())
        va = np.flatnonzero(bt & rows.task_id.isin(inner["calib"]).to_numpy())
        std = Standardizer().fit(seq[bt], pad[bt], scalar[bt])
        xs = StandardScaler().fit(X[bt])
        s_seq, s_scalar, _ = std.transform(seq, pad, scalar)
        extra = xs.transform(X).astype(np.float32)
        with open(MODEL_DIR / f"scalers_{model}.pkl", "wb") as f:
            pickle.dump({"traj": std, "features": xs}, f)
        m = (rows.model == model).to_numpy()
        for name, ex in [("B6", None), ("B7", extra)]:
            seed_preds, epochs = [], []
            for seed in SEEDS:
                net, hist = train_one(s_seq, pad, s_scalar, ex, y, tr, va, seed)
                torch.save(net.state_dict(), MODEL_DIR / f"{name}_{model}_seed{seed}.pt")
                seed_preds.append(predict(net, s_seq[m], pad[m], s_scalar[m], None if ex is None else ex[m]))
                epochs.append({"seed": seed, "epochs_run": len(hist), "best_epoch": int(np.argmin(hist)) + 1,
                               "best_inner_log_loss": float(min(hist))})
            preds[name][m] = np.mean(seed_preds, axis=0)
            info[f"{model}/{name}"] = {"inner_train_problems": len(inner["base_train"]),
                                       "inner_val_problems": len(inner["calib"]), "seeds": epochs}
            print(model, name, epochs, flush=True)
    return preds, info


if __name__ == "__main__":
    torch.set_num_threads(4)
    OUT = Path("runs/v2/v4_transformer")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, G = prepare_v2(load_rows(), load_base_preds())
    X, xcols = prepare_matrix(load_features(rows), VARIANTS["B2"][1])
    seq, pad, scalar = tensors_for(rows, DATA_DIR / "trajectories_dev.npz")
    preds, info = train_all(rows, X, seq, pad, scalar)
    (OUT / "training.json").write_text(json.dumps(info, indent=1))
    v2p = pd.read_parquet(v2_preds_path())
    key = ["model", "task_id", "sample_idx"]
    new = rows[key].assign(B6=preds["B6"], B7=preds["B7"])
    v2p = v2p.drop(columns=[c for c in ["B6", "B7"] if c in v2p]).merge(new, on=key, validate="1:1")
    v2p.to_parquet(v2_preds_path())
    print("saved", v2_preds_path(), list(v2p.columns))
