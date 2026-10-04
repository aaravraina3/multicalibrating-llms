"""Phase 13: the single final run on test, with every setting frozen in PROTOCOL.md.
`--dry-run` runs the same pipeline with validation in place of test (no test data loaded) to catch bugs first."""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calib import metrics as M
from calib.base_model import VARIANTS, fit_fixed
from calib.data import DATA_DIR, load_rows
from calib.features import build_features, prepare_matrix
from calib.pipeline import calibrate
from calib.plots import reliability_diagram
from calib.rq_setup import arrays, group_sets, load_base_preds, prepare, sel
from calib.splits import assign
from experiments import rq1_replication, rq3_routing, rq4_overfitting
from experiments.rq2_base_strength import evaluate_table

DRY = "--dry-run" in sys.argv
ROLE = "validation" if DRY else "test"
N_BOOT = 2000
git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
OUT = Path("runs") / (f"dryrun-{git_hash}" if DRY else git_hash)
OUT.mkdir(parents=True, exist_ok=True)
chosen = json.loads(Path("runs/phase10/chosen.json").read_text())
splits = json.loads(Path("runs/phase9/splits.json").read_text())
hyper = json.loads(Path("runs/phase9/chosen_hyperparameters.json").read_text())

# Base model predictions for test rows: refit on base_train with the frozen hyperparameters.
dev = load_rows()
base_preds = load_base_preds()
if not DRY:
    all_rows = load_rows(include_test=True)
    test = all_rows[all_rows.split == "test"]
    feats_dev = pd.read_parquet(DATA_DIR / "features.parquet")
    feats_test = build_features(test)
    dev["role"] = assign(dev, splits)
    extra = test[["task_id", "sample_idx", "model", "y"]].assign(role="test")
    for model in ["qwen3", "gpt-oss"]:
        m_dev = (dev.model == model).to_numpy()
        tr = m_dev & (dev.role == "base_train").to_numpy()
        va = m_dev & (dev.role == "validation").to_numpy()
        m_te = (test.model == model).to_numpy()
        for name, (kind, blocks) in VARIANTS.items():
            X_dev, _ = prepare_matrix(feats_dev, blocks)
            X_te, _ = prepare_matrix(feats_test, blocks)
            est = fit_fixed(kind, tuple(hyper[f"{model}/{name}"]["params"]), X_dev[tr], dev.y.to_numpy()[tr])
            # The refit must reproduce the Phase 9 validation predictions exactly.
            assert np.allclose(est.predict_proba(X_dev[va])[:, 1], base_preds.loc[va, name].to_numpy())
            extra.loc[m_te, name] = est.predict_proba(X_te[m_te])[:, 1]
    base_preds = pd.concat([base_preds, extra], ignore_index=True)
    rows, G = prepare(all_rows, base_preds)
else:
    rows, G = prepare(dev, base_preds)

summary = {"git_hash": git_hash, "eval_role": ROLE, "n_boot": N_BOOT}

# RQ1: replication (test only; already run once in Phase 8 under the bug-fix-only rule).
if not DRY:
    t1 = rq1_replication.run(OUT / "rq1", N_BOOT)
    summary["rq1"] = t1[t1.medians == "train"][["model", "method", "bss_value", "bss_lo", "bss_hi", "paper_bss",
                                                 "acc", "paper_acc"]].to_dict("records")

# RQ2: all rows, then empty programs excluded.
t2, d2 = evaluate_table(rows, G, chosen, ROLE, n_boot=N_BOOT)
t2.to_csv(OUT / "rq2.csv", index=False)
d2.to_csv(OUT / "rq2_diffs.csv", index=False)
nonempty = (rows.code.str.strip() != "").to_numpy()
t2e, d2e = evaluate_table(rows, G, chosen, ROLE, eval_mask=nonempty, n_boot=N_BOOT)
t2e.to_csv(OUT / "rq2_nonempty.csv", index=False)
d2e.to_csv(OUT / "rq2_nonempty_diffs.csv", index=False)
prim = d2[(d2.groups == "no_difficulty") & (d2.method == "iglb") & d2.start.isin(["avg_prob", "B2"])]
summary["rq2_primary"] = prim[["model", "start", "bss_minus_platt", "lo", "hi"]].to_dict("records")

# RQ3: routing; P4 cutoffs from validation.
t3, b3, s3 = rq3_routing.run(rows, G, chosen, ROLE, "validation", OUT / "rq3", N_BOOT)
prim3 = b3[(b3["compare"] == "iglb - platt") & (b3.policy == "P6 pick higher") & (b3.primary == "qwen3")]
summary["rq3_primary"] = prim3[["start", "diff", "lo", "hi"]].to_dict("records")
summary["rq3_platt_rank_sanity_max_differences"] = int(s3.decisions_differ.max())

# RQ4: subsampling, scored on the eval role.
s4 = rq4_overfitting.run(rows, G, chosen, ROLE, OUT / "rq4")
summary["rq4"] = s4[s4["size"].isin([25, 211])].to_dict("records")

# Figures: reliability diagrams on the eval role, avg_prob and B2, uncalibrated / Platt / IGLB.
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i, (model, start) in enumerate([(m, s) for m in ["qwen3", "gpt-oss"] for s in ["avg_prob", "B2"]]):
    cols = group_sets(rows, G, model)[0]["no_difficulty"]
    fit = arrays(rows, G, sel(rows, model, ["calib_fit", "calib_stop"]), start, cols)
    fit_only = arrays(rows, G, sel(rows, model, "calib_fit"), start, cols)
    stop = arrays(rows, G, sel(rows, model, "calib_stop"), start, cols)
    ev = arrays(rows, G, sel(rows, model, ROLE), start, cols)
    preds = {"uncalibrated": ev["p"],
             "Platt": calibrate("platt", chosen["platt_version"][f"{model}/{start}"], fit, None, ev)[0],
             "IGLB": calibrate("iglb", "code", fit_only, stop, ev, epsilon=chosen["iglb_epsilon"])[0]}
    for j, (name, q) in enumerate(preds.items()):
        reliability_diagram(np.clip(q, 0, 1), ev["y"], axes[i, j],
                            title=f"{model} {start} {name}, BSS {M.brier_skill(np.clip(q, 0, 1), ev['y']):.3f}")
fig.tight_layout()
fig.savefig(OUT / "reliability.png", dpi=110)

(OUT / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
print(json.dumps({k: summary[k] for k in summary if k.startswith(("rq2_primary", "rq3_primary", "rq3_platt"))},
                 indent=1, default=float))
