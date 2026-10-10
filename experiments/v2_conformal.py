"""Phase V5: conformal risk control for the Qwen3 -> GPT OSS cascade.
Thresholds are chosen on val_conformal (nothing was fit, tuned, or early stopped there); risk is measured on
`eval_role` (val_tune during development, test in V8)."""

import numpy as np
import pandas as pd

from calib import routing as R
from calib.conformal import choose_threshold, realized_risk
from calib.shap_tools import block_shap, explain_probability
from calib.v2_eval import mask

TARGETS = [0.05, 0.10, 0.15]
CALIBRATORS = ["uncalibrated", "platt", "iglb", "bmc"]


def synthetic(n_problems, rng):
    difficulty = rng.normal(0, 1.5, n_problems)
    task = np.repeat(np.arange(n_problems), 10)
    p = 1 / (1 + np.exp(-(difficulty[task] + rng.normal(0, 0.5, len(task)))))
    return p, (rng.random(len(task)) < p).astype(int), task


def simulation_check(reps=500, n_cal=132, n_test=500, seed=1):
    """Fresh calibration and test draws with known pass probabilities; average realized risk per target."""
    rng = np.random.default_rng(seed)
    out = []
    for target in TARGETS:
        risks = []
        for _ in range(reps):
            p, y, task = synthetic(n_cal, rng)
            t = choose_threshold(p, y, task, target)
            pt, yt, tt = synthetic(n_test, rng)
            risks.append(realized_risk(pt, yt, tt, t))
        out.append({"target": target, "mean_realized_risk": float(np.mean(risks)),
                    "share_of_draws_above_target": float(np.mean(np.array(risks) > target))})
    return pd.DataFrame(out)


def paired(rows, primary_q, fallback_y, role, primary="qwen3", fallback="gpt-oss"):
    """Primary rows of `role` with their calibrated score, paired by (problem, sample) with the fallback's label."""
    a = rows.loc[mask(rows, primary, role), ["task_id", "sample_idx", "y"]].assign(q=primary_q[role])
    b = rows.loc[mask(rows, fallback, role), ["task_id", "sample_idx"]].assign(y_b=fallback_y[role])
    return a.merge(b, on=["task_id", "sample_idx"], validate="1:1")


def run(rows, scores, eval_role, raw_scores):
    """scores[calibrator][role] = calibrated primary probabilities (row order of mask(rows, 'qwen3', role)).
    raw_scores[role] = uncalibrated starting score, used for the matched budget rank based comparison."""
    fb_y = {r: rows.y.to_numpy()[mask(rows, "gpt-oss", r)] for r in ["val_conformal", eval_role]}
    out = []
    for cal in CALIBRATORS:
        conf = paired(rows, scores[cal], fb_y, "val_conformal")
        ev = paired(rows, scores[cal], fb_y, eval_role)
        raw = paired(rows, raw_scores, fb_y, eval_role)
        y_a, y_b, task = ev.y.to_numpy(), ev.y_b.to_numpy(), ev.task_id.to_numpy()
        n = conf.task_id.nunique()
        for target in TARGETS:
            t = choose_threshold(conf.q.to_numpy(), conf.y.to_numpy(), conf.task_id.to_numpy(), target)
            use_b = ev.q.to_numpy() < t
            res = R.evaluate(use_b, y_a, y_b, c_b=1.0)
            k = int(use_b.sum())
            # Matched budget: rank based P4 on the raw score escalating the same number of answers.
            order = np.argsort(raw.q.to_numpy(), kind="stable")
            p4 = np.zeros(len(raw), bool)
            p4[order[:k]] = True
            p4_pass = np.where(p4, raw.y_b, raw.y).mean()
            oracle = R.oracle_pass_at(y_a, y_b, use_b.mean())
            out.append({"calibrator": cal, "target": target, "threshold": t,
                        "realized_risk": realized_risk(ev.q.to_numpy(), y_a, task, t),
                        "bound_on_val_conformal": n / (n + 1) * realized_risk(conf.q.to_numpy(), conf.y.to_numpy(),
                                                                             conf.task_id.to_numpy(), t) + 1 / (n + 1),
                        "escalation": res["escalation"],
                        "system_pass": res["pass_rate"], "oracle_pass_matched": oracle,
                        "regret": oracle - res["pass_rate"], "p4_raw_pass_matched": float(p4_pass),
                        "p4_raw_regret": oracle - float(p4_pass), "accepted_error": res["accepted_error"]})
    return pd.DataFrame(out)


def explain_escalations(rows, X, xcols, b4_model, bg, q_conf, t):
    """Mean B4 block SHAP (probability scale) of escalated vs accepted Qwen3 answers on val_conformal.
    Explains the B4 base model's prediction before calibration, not the routing system."""
    m = mask(rows, "qwen3", "val_conformal")
    sv, _ = explain_probability(b4_model, bg, X[m])
    bs = block_shap(sv, xcols)
    esc = q_conf < t
    out = [{"block": b, "escalated_mean": float(v[esc].mean()) if esc.any() else np.nan,
            "accepted_mean": float(v[~esc].mean()) if (~esc).any() else np.nan} for b, v in bs.items()]
    df = pd.DataFrame(out)
    df["difference"] = df.escalated_mean - df.accepted_mean
    imp = np.abs(sv[esc]).mean(0) if esc.any() else np.zeros(len(xcols))
    top = [(xcols[j], float(sv[esc, j].mean())) for j in np.argsort(-imp)[:5]] if esc.any() else []
    return df.sort_values("difference"), top, int(esc.sum()), int(len(esc))
