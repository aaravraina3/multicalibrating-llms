"""Phase V6: Murphy decomposition per method with bootstrap intervals vs Platt, and the 28 comparison
Holm family (fixed in PROTOCOL_V2.md before the test run)."""

import numpy as np
import pandas as pd

from calib import metrics as M
from calib.stats import bootstrap_diffs, holm, murphy, two_sided_p
from calib.v2_eval import mask

GROUP_AWARE = ["linr", "iglb", "bmc"]


def _rel(p, y):
    return murphy(p, y)["reliability"]


def _res(p, y):
    return murphy(p, y)["resolution"]


def murphy_table(rows, preds, starts, methods, role, n_boot=2000):
    out = []
    for model in ["qwen3", "gpt-oss"]:
        m = mask(rows, model, role)
        y, task = rows.y.to_numpy()[m], rows.task_id.to_numpy()[m]
        for start in starts:
            platt = preds[(model, start, "platt")]
            for method in methods:
                q = preds[(model, start, method)]
                rec = {"model": model, "start": start, "method": method, **murphy(q, y)}
                if method != "platt":
                    for name, fn in [("reliability", _rel), ("resolution", _res)]:
                        d = bootstrap_diffs(fn, q, platt, y, task, n_boot)
                        rec[f"d_{name}_vs_platt"] = fn(q, y) - fn(platt, y)
                        rec[f"d_{name}_lo"], rec[f"d_{name}_hi"] = np.percentile(d, [2.5, 97.5])
                out.append(rec)
    return pd.DataFrame(out)


def holm_family(rows, preds, family_starts, role, n_boot=2000):
    """family_starts: the 4 starting scores per model (raw, B2, B4, selected transformer).
    24 comparisons: each group aware calibrator minus Platt on Brier. 4 more: B4 minus B2 and transformer minus
    B4 on uncalibrated Brier, per model."""
    rows_out = []
    for model in ["qwen3", "gpt-oss"]:
        m = mask(rows, model, role)
        y, task = rows.y.to_numpy()[m], rows.task_id.to_numpy()[m]
        for start in family_starts[model]:
            for method in GROUP_AWARE:
                a, b = preds[(model, start, method)], preds[(model, start, "platt")]
                d = bootstrap_diffs(M.brier, a, b, y, task, n_boot)
                rows_out.append({"model": model, "comparison": f"{start}: {method} - platt", "brier_diff": M.brier(a, y) - M.brier(b, y),
                                 "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": two_sided_p(d)})
        raw = family_starts[model]
        pairs = [("B4", "B2"), (raw[3], "B4")]
        for s1, s2 in pairs:
            a, b = preds[(model, s1, "uncalibrated")], preds[(model, s2, "uncalibrated")]
            d = bootstrap_diffs(M.brier, a, b, y, task, n_boot)
            rows_out.append({"model": model, "comparison": f"{s1} - {s2} (uncalibrated)", "brier_diff": M.brier(a, y) - M.brier(b, y),
                             "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5), "p": two_sided_p(d)})
    df = pd.DataFrame(rows_out)
    df["p_holm"] = holm(df.p.to_numpy())
    df["significant_holm"] = df.p_holm < 0.05
    return df
