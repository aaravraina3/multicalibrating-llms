"""Post-run reporting fixes (no test number changes, nothing refit except deterministic replays):
(a) exact routing explanation for B2, the logistic model that actually makes the cascade's decisions:
    contribution of feature j = coefficient_j x (standardized x_j - base_train mean), on the log odds scale;
(b) worst miscalibrated slices of B4 on val_tune with at least 20 problems per slice (was 30 rows)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calib.base_model import VARIANTS, fit_fixed
from calib.calibrators.platt import Platt
from calib.groups import EXTENDED
from calib.shap_tools import BLOCK_OF, background, block_shap, explain_probability
from calib.v2_eval import group_cols, mask
from calib.xgb_model import make_xgb
from experiments.v2_shap import DISCOVERED, load, membership, slices_over

OUT = Path("runs/0bd9043/reporting_fixes")
OUT.mkdir(parents=True, exist_ok=True)
HYPER = json.loads(Path("runs/phase9/chosen_hyperparameters.json").read_text())
FROZEN = json.loads(Path("runs/v2/protocol_v2_frozen.json").read_text())
BEST = json.loads(Path("runs/v2/v2_xgboost/best_config.json").read_text())
conf = pd.read_csv("runs/0bd9043/conformal.csv")

rows, G, X, xcols = load()
y, task = rows.y.to_numpy(), rows.task_id.to_numpy()

# (a) Exact B2 explanation of escalations on val_conformal (Platt on B2, target 0.10, frozen threshold).
bt, ca, vc = mask(rows, "qwen3", "base_train"), mask(rows, "qwen3", "calib"), mask(rows, "qwen3", "val_conformal")
b2 = fit_fixed("logistic", tuple(HYPER["qwen3/B2"]["params"]), X[bt], y[bt])
assert np.allclose(b2.predict_proba(X[vc])[:, 1], rows.B2.to_numpy()[vc], atol=1e-8)  # same model as the cascade
scaler, lr = b2.named_steps["standardscaler"], b2.named_steps["logisticregression"]
z = scaler.transform(X[vc])
contrib = lr.coef_[0] * (z - scaler.transform(X[bt]).mean(0))  # log odds, relative to the base_train average
platt = Platt(FROZEN["platt_versions"]["qwen3/B2"]).fit(rows.B2.to_numpy()[ca], y[ca])
t = float(conf[(conf.calibrator == "platt") & (conf.target == 0.10)].threshold.iloc[0])
esc = platt.predict(rows.B2.to_numpy()[vc]) < t
blocks = {b: [xcols.index(c) for c in cols if c in xcols] for b, cols in BLOCK_OF.items()}
tab = pd.DataFrame([{"block": b, "escalated_mean_logodds": contrib[esc][:, idx].sum(1).mean(),
                     "accepted_mean_logodds": contrib[~esc][:, idx].sum(1).mean()} for b, idx in blocks.items()])
tab["difference"] = tab.escalated_mean_logodds - tab.accepted_mean_logodds
tab = tab.sort_values("difference")
tab.to_csv(OUT / "routing_explanation_b2_exact.csv", index=False)
feat = pd.DataFrame({"feature": xcols, "escalated_mean_logodds": contrib[esc].mean(0),
                     "accepted_mean_logodds": contrib[~esc].mean(0)})
feat["difference"] = feat.escalated_mean_logodds - feat.accepted_mean_logodds
feat = feat.reindex(feat.difference.abs().sort_values(ascending=False).index)
feat.to_csv(OUT / "routing_explanation_b2_exact_features.csv", index=False)
print(f"(a) threshold {t:.3f}, escalated {esc.sum()} of {len(esc)}")
print(tab.round(3).to_string(index=False))
print(feat.head(8).round(3).to_string(index=False))

# (b) Worst slices with at least 20 problems.
results = {}
for model in ["qwen3", "gpt-oss"]:
    bt, vt = mask(rows, model, "base_train"), mask(rows, model, "val_tune")
    b4 = make_xgb(BEST[model]["params"], 0).fit(X[bt], y[bt])
    sv, _ = explain_probability(b4, background(X[bt], 200, seed=0), X[vt])
    bs = block_shap(sv, xcols)
    hand = group_cols(rows, G, model)
    shap_groups = json.loads(Path(f"runs/v2/v3_shap/{model}/shap_groups.json").read_text())
    disc = DISCOVERED.get(f"{model}/B4", [])
    U = np.column_stack([G[:, hand], membership(X, xcols, disc), membership(X, xcols, shap_groups)])
    names = [EXTENDED[i] for i in hand] + ["disc: " + str(g["conditions"]) for g in disc] + \
            ["shap: " + g["name"] for g in shap_groups]
    p = rows.B4.to_numpy()
    sl = sorted(slices_over(p[vt], y[vt], task[vt], U[vt], names, min_problems=20), key=lambda s: -s["weight"])
    worst, picked = [], []
    for s in sl:
        if any((s["mask"] & pm).sum() / (s["mask"] | pm).sum() > 0.5 for pm in picked):
            continue
        picked.append(s["mask"])
        prof = sv[s["mask"]].mean(0) - sv.mean(0)
        top = np.argsort(-np.abs(prof))[:3]
        worst.append({k: v for k, v in s.items() if k != "mask"} |
                     {"significant_2se": abs(s["gap"]) > 2 * s["se"],
                      "block_profile_vs_overall": {b: float(v[s["mask"]].mean() - v.mean()) for b, v in bs.items()},
                      "top_features_vs_overall": [(xcols[j], float(prof[j]), float(X[vt][s["mask"], j].mean()),
                                                   float(X[vt][:, j].mean())) for j in top]})
        if len(worst) == 5:
            break
    results[model] = worst
    print(f"(b) {model}")
    for w in worst:
        print("  ", w["group"], "bin", w["bin"], "problems", w["problems"], "rows", w["rows"], "p", round(w["mean_p"], 3),
              "pass", round(w["pass_rate"], 3), "gap", round(w["gap"], 3), "se", round(w["se"], 3),
              {k: round(v, 3) for k, v in w["block_profile_vs_overall"].items()},
              [(f, round(d, 3), round(a, 2), round(b, 2)) for f, d, a, b in w["top_features_vs_overall"]])
(OUT / "worst_slices_min20_problems.json").write_text(json.dumps(results, indent=1, default=float))
