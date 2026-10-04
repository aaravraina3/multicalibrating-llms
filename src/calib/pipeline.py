"""Shared plumbing: pick rows, fit a calibrator, predict."""

from calib.calibrators.histogram import HistogramBinning, Isotonic
from calib.calibrators.ighb import IGHB
from calib.calibrators.iglb import IGLB
from calib.calibrators.linr import LINR
from calib.calibrators.logr import LOGR
from calib.calibrators.platt import Platt

METHODS = ["uncalibrated", "platt", "hb", "linr", "logr", "ighb", "iglb"]


def part(rows, G, score, mask):
    """Arrays for one slice of rows: starting score p, labels y, groups G, problem ids."""
    m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
    return {"p": rows[score].to_numpy()[m], "y": rows.y.to_numpy()[m], "G": G[m], "task": rows.task_id.to_numpy()[m]}


def calibrate(method, version, fit, stop, ev, **kw):
    """Fit `method` on `fit`, early stop on `stop` (IGLB only), return predictions for `ev`.

    version: 'code' or 'paper' (Platt also takes 'logit'). HB and isotonic ignore it.
    """
    if method == "uncalibrated":
        return ev["p"].copy(), None
    if method == "platt":
        cal = Platt(version).fit(fit["p"], fit["y"])
    elif method == "hb":
        cal = HistogramBinning().fit(fit["p"], fit["y"])
    elif method == "isotonic":
        cal = Isotonic().fit(fit["p"], fit["y"])
    elif method == "linr":
        cal = LINR(version).fit(fit["p"], fit["y"], fit["G"])
    elif method == "logr":
        cal = LOGR(version).fit(fit["p"], fit["y"], fit["G"])
    elif method == "ighb":
        cal = IGHB(version, **kw).fit(fit["p"], fit["y"], fit["G"])
    elif method == "iglb":
        cal = IGLB(version, **kw).fit(fit["p"], fit["y"], fit["G"], stop["p"], stop["y"], stop["G"])
    else:
        raise ValueError(method)
    return cal.predict(ev["p"], ev["G"]), cal
