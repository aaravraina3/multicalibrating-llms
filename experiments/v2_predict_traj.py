"""B6 and B7 predictions for every row (dev and test) from the frozen transformer weights, on the CPU.
Runs in its own process: importing torch next to xgboost and shap crashes on macOS (OpenMP clash).
Checks that dev predictions match the saved ones, then writes data/v2_traj_preds_all.parquet."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from calib.base_model import VARIANTS
from calib.data import DATA_DIR, load_rows
from calib.features import prepare_matrix
from calib.trajectory import TrajectoryNet, build_tensors, predict
from calib.v2_setup import load_features, prepare_v2, v2_preds_path

MODELS = Path("runs/v2/v4_transformer/models")
SEEDS = [0, 1, 2, 3, 4]
OUT = DATA_DIR / "v2_traj_preds_all.parquet"

if __name__ == "__main__":
    include_test = "--with-test" in sys.argv
    rows, _ = prepare_v2(load_rows(include_test=include_test))
    X, _ = prepare_matrix(load_features(rows), VARIANTS["B2"][1])
    seq, pad, scalar = build_tensors(rows)
    out = rows[["model", "task_id", "sample_idx"]].copy()
    for model in ["qwen3", "gpt-oss"]:
        m = (rows.model == model).to_numpy()
        with open(MODELS / f"scalers_{model}.pkl", "rb") as f:
            sc = pickle.load(f)
        s_seq, s_scalar, _ = sc["traj"].transform(seq[m], pad[m], scalar[m])
        extra = sc["features"].transform(X[m]).astype(np.float32)
        for name, ex in [("B6", None), ("B7", extra)]:
            ps = []
            for seed in SEEDS:
                net = TrajectoryNet(n_extra=0 if ex is None else ex.shape[1])
                net.load_state_dict(torch.load(MODELS / f"{name}_{model}_seed{seed}.pt"))
                ps.append(predict(net, s_seq, pad[m], s_scalar, ex))
            out.loc[m, name] = np.mean(ps, axis=0)
    saved = pd.read_parquet(v2_preds_path())
    check = out.merge(saved, on=["model", "task_id", "sample_idx"], suffixes=("", "_saved"))
    for c in ["B6", "B7"]:
        gap = float(np.abs(check[c] - check[f"{c}_saved"]).max())
        print(c, "max |CPU replay - saved| on dev rows:", gap)
        assert gap < 1e-4, c
    out.to_parquet(OUT)
    print("wrote", OUT, len(out))
