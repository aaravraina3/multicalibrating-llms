"""Phase V7 (optional): distribution shift. Fit the base predictor and calibrators on one model's generations,
evaluate on the other's. Group memberships use each row's own model thresholds (label free)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from calib import metrics as M
from calib.base_model import VARIANTS, fit_fixed
from calib.calibrators.boosted import BoostedMC
from calib.calibrators.iglb import IGLB
from calib.calibrators.platt import Platt
from calib.conformal import choose_threshold, realized_risk
from calib.features import prepare_matrix
from calib.v2_eval import group_cols, mask
from calib.xgb_model import make_xgb

V1_CHOSEN = json.loads(Path("runs/phase10/chosen.json").read_text())
HYPER = json.loads(Path("runs/phase9/chosen_hyperparameters.json").read_text())
BEST = json.loads(Path("runs/v2/v2_xgboost/best_config.json").read_text())
TARGETS = [0.05, 0.10, 0.15]


def fit_base(kind, source, X, y, bt):
    if kind == "B2":
        return fit_fixed("logistic", tuple(HYPER[f"{source}/B2"]["params"]), X[bt], y[bt])
    return make_xgb(BEST[source]["params"], 0).fit(X[bt], y[bt])


def risk_coverage(q, y, task, n=21):
    out = []
    for t in np.linspace(0, 1, n):
        out.append({"threshold": t, "coverage": float((q >= t).mean()), "risk": realized_risk(q, y, task, t)})
    return out


def run(rows, G, X, xcols, eval_role, out, kinds=("B2", "B4")):
    out.mkdir(parents=True, exist_ok=True)
    y, task = rows.y.to_numpy(), rows.task_id.to_numpy()
    cols = sorted(set(group_cols(rows, G, "qwen3")) & set(group_cols(rows, G, "gpt-oss")))
    results, curves = [], []
    for kind in kinds:
        for source, target in [("qwen3", "qwen3"), ("gpt-oss", "gpt-oss"), ("qwen3", "gpt-oss"), ("gpt-oss", "qwen3")]:
            base = fit_base(kind, source, X, y, mask(rows, source, "base_train"))
            p = base.predict_proba(X)[:, 1]
            ca, vt = mask(rows, source, "calib"), mask(rows, source, "val_tune")
            vc, ev = mask(rows, source, "val_conformal"), mask(rows, target, eval_role)
            Gs = G[:, cols]
            cals = {"uncalibrated": None,
                    "platt": Platt("logit").fit(p[ca], y[ca]),
                    "iglb": IGLB("code", epsilon=V1_CHOSEN["iglb_epsilon"]).fit(p[ca], y[ca], Gs[ca], p[vt], y[vt], Gs[vt]),
                    "bmc": BoostedMC().fit(p[ca], y[ca], X[ca], p[vt], y[vt], X[vt])}
            for name, cal in cals.items():
                def apply(mk):
                    if cal is None:
                        return p[mk]
                    if name == "platt":
                        return cal.predict(p[mk])
                    return np.clip(cal.predict(p[mk], X[mk] if name == "bmc" else Gs[mk]), 0, 1)
                q_ev, q_vc = apply(ev), apply(vc)
                rec = {"base": kind, "fit_on": source, "evaluated_on": target, "calibrator": name,
                       **M.summary(q_ev, y[ev])}
                for tg in TARGETS:
                    t = choose_threshold(q_vc, y[vc], task[vc], tg)
                    rec[f"risk_at_{tg}"] = realized_risk(q_ev, y[ev], task[ev], t)
                    rec[f"coverage_at_{tg}"] = float((q_ev >= t).mean())
                results.append(rec)
                for c in risk_coverage(q_ev, y[ev], task[ev]):
                    curves.append({"base": kind, "fit_on": source, "evaluated_on": target, "calibrator": name, **c})
    results, curves = pd.DataFrame(results), pd.DataFrame(curves)
    results.to_csv(out / "shift.csv", index=False)
    curves.to_csv(out / "risk_coverage.csv", index=False)
    return results, curves
