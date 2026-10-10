"""Fit every calibrator once per (model, starting score) on calib, then predict any role.
IGLB and boosted multicalibration early stop on val_tune (v2 rule)."""

import numpy as np

from calib import metrics as M
from calib.calibrators.boosted import BoostedMC
from calib.calibrators.histogram import HistogramBinning
from calib.calibrators.ighb import IGHB
from calib.calibrators.iglb import IGLB
from calib.calibrators.linr import LINR
from calib.calibrators.logr import LOGR
from calib.calibrators.platt import Platt
from calib.groups import EXTENDED, keep_groups

METHODS = ["uncalibrated", "platt", "hb", "linr", "logr", "ighb", "iglb", "bmc"]
GROUP_AWARE = ["linr", "iglb", "bmc"]
DIFFICULTY = ["comp_easy", "comp_medium", "comp_hard"]


def mask(rows, model, roles):
    roles = [roles] if isinstance(roles, str) else roles
    return ((rows.model == model) & rows.role.isin(roles)).to_numpy()


def group_cols(rows, G, model, with_difficulty=False):
    m = (rows.model == model).to_numpy()
    kept = keep_groups(G[m], rows[m], EXTENDED)
    return [EXTENDED.index(n) for n in kept if with_difficulty or n not in DIFFICULTY]


def part(rows, G, X, model, roles, start, cols):
    m = mask(rows, model, roles)
    return {"p": np.clip(rows[start].to_numpy()[m], 0, 1), "y": rows.y.to_numpy()[m], "G": G[m][:, cols],
            "X": X[m], "task": rows.task_id.to_numpy()[m], "mask": m}


def choose_platt(fit, tune):
    """Platt input (raw p, log p, logit p) with the best val_tune Brier. Used for starting scores v1 never tuned."""
    scores = {v: M.brier(Platt(v).fit(fit["p"], fit["y"]).predict(tune["p"]), tune["y"])
              for v in ["code", "paper", "logit"]}
    return min(scores, key=scores.get)


def fit_all(fit, tune, platt_version, alpha, epsilon, methods=METHODS):
    """Calibrators fit on `fit` (calib). IGLB and boosted multicalibration early stop on `tune` (val_tune)."""
    cals = {}
    if "platt" in methods:
        cals["platt"] = Platt(platt_version).fit(fit["p"], fit["y"])
    if "hb" in methods:
        cals["hb"] = HistogramBinning().fit(fit["p"], fit["y"])
    if "linr" in methods:
        cals["linr"] = LINR("code").fit(fit["p"], fit["y"], fit["G"])
    if "logr" in methods:
        cals["logr"] = LOGR("paper").fit(fit["p"], fit["y"], fit["G"])
    if "ighb" in methods:
        cals["ighb"] = IGHB("code", alpha=alpha).fit(fit["p"], fit["y"], fit["G"])
    if "iglb" in methods:
        cals["iglb"] = IGLB("code", epsilon=epsilon).fit(fit["p"], fit["y"], fit["G"], tune["p"], tune["y"], tune["G"])
    if "bmc" in methods:
        cals["bmc"] = BoostedMC().fit(fit["p"], fit["y"], fit["X"], tune["p"], tune["y"], tune["X"])
    return cals


def predict(cals, method, ev):
    if method == "uncalibrated":
        return ev["p"].copy()
    cal = cals[method]
    if method in ["platt", "hb"]:
        q = cal.predict(ev["p"])
    elif method == "bmc":
        q = cal.predict(ev["p"], ev["X"])
    else:
        q = cal.predict(ev["p"], ev["G"])
    return np.clip(q, 0, 1)
