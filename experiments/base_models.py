"""Phase 9: base_train / calib split, features, base predictors B0-B3, ablation, tree auditor."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

from calib import metrics as M
from calib.base_model import VARIANTS, fit_select
from calib.data import DATA_DIR, load_rows
from calib.features import build_features, prepare_matrix
from calib.splits import assign, save, split_train

OUT = Path("runs/phase9")
OUT.mkdir(parents=True, exist_ok=True)
rows = load_rows()

splits = split_train(rows[rows.split == "train"].task_id)
save(splits, OUT / "splits.json")
rows["role"] = assign(rows, splits)
print({k: len(v) for k, v in splits.items()}, rows.groupby(["model", "role"]).size().to_dict())

feat_path = DATA_DIR / "features.parquet"
if feat_path.exists():
    feats = pd.read_parquet(feat_path)
else:
    feats = build_features(rows)
    feats.to_parquet(feat_path)

preds = pd.DataFrame({"task_id": rows.task_id, "sample_idx": rows.sample_idx, "model": rows.model,
                      "role": rows.role, "y": rows.y})
table, chosen, ablation = [], {}, []
for model in ["qwen3", "gpt-oss"]:
    m = (rows.model == model).to_numpy()
    tr, va = m & (rows.role == "base_train").to_numpy(), m & (rows.role == "validation").to_numpy()
    y = rows.y.to_numpy()
    for name, (kind, blocks) in VARIANTS.items():
        X, cols = prepare_matrix(feats, blocks)
        est, params, _ = fit_select(kind, X[tr], y[tr], X[va], y[va])
        chosen[f"{model}/{name}"] = {"kind": kind, "params": params, "n_features": len(cols)}
        p_all = est.predict_proba(X[m])[:, 1]
        preds.loc[m, name] = p_all
        q = preds.loc[va, name].to_numpy()
        table.append({"model": model, "variant": name, "kind": kind, "params": str(params), **M.summary(q, y[va])})
    # Ablation: B2 without each block. "logprob" also drops avg_prob.
    full = VARIANTS["B2"][1]
    for drop in [None, "logprob", "structure", "size", "self_consistency"]:
        blocks = [b for b in full if b != drop and not (drop == "logprob" and b == "avg")]
        X, _ = prepare_matrix(feats, blocks)
        est, params, loss = fit_select("logistic", X[tr], y[tr], X[va], y[va])
        q = est.predict_proba(X[va])[:, 1]
        ablation.append({"model": model, "dropped": drop or "(none)", "C": params[0], "val_log_loss": loss,
                         "val_auroc": M.auroc(q, y[va]), "val_bss": M.brier_skill(q, y[va])})

preds.to_parquet(DATA_DIR / "base_preds.parquet")
table, ablation = pd.DataFrame(table), pd.DataFrame(ablation)
table.to_csv(OUT / "base_models_validation.csv", index=False)
ablation.to_csv(OUT / "ablation_validation.csv", index=False)
(OUT / "chosen_hyperparameters.json").write_text(json.dumps(chosen, indent=1))
print(table.round(3).to_string(index=False))
print(ablation.round(4).to_string(index=False))

# Optional: shallow tree on B2 residuals on calib. Leaves are data discovered groups.
X, cols = prepare_matrix(feats, VARIANTS["B2"][1])
lines = []
for model in ["qwen3", "gpt-oss"]:
    c = (rows.model == model).to_numpy() & (rows.role == "calib").to_numpy()
    resid = rows.y.to_numpy()[c] - preds.loc[c, "B2"].to_numpy()
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=400, random_state=0).fit(X[c], resid)
    leaf = tree.apply(X[c])
    lines.append(f"## {model}\n" + export_text(tree, feature_names=cols, decimals=3))
    for lf in np.unique(leaf):
        sel = leaf == lf
        lines.append(f"leaf {lf}: rows {sel.sum()}, problems {rows[c][sel].task_id.nunique()}, "
                     f"mean residual {resid[sel].mean():+.3f}")
(OUT / "tree_auditor.txt").write_text("\n".join(lines))
print("\n".join(lines))
