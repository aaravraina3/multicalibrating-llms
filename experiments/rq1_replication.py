"""Phase 8 / RQ1: replicate Campos Table 1 (LiveCodeBench) on the test split.
avg_prob, code faithful versions, replication groups, fit on train, IGLB early stops on validation."""

from pathlib import Path

import pandas as pd

from calib import metrics as M
from calib.bootstrap import interval
from calib.data import load_rows
from calib.groups import REPLICATION, build_groups
from calib.pipeline import METHODS, calibrate, part
from calib.scores import add_scores

PAPER = {  # Campos Table 1, LiveCodeBench: (BSS, ACC)
    "qwen3": {"uncalibrated": (-0.543, 0.458), "platt": (0.377, 0.798), "hb": (0.383, 0.797), "linr": (0.463, 0.822),
              "logr": (0.306, 0.828), "ighb": (0.232, 0.731), "iglb": (0.480, 0.830)},
    "gpt-oss": {"uncalibrated": (-0.075, 0.511), "platt": (0.187, 0.707), "hb": (0.237, 0.717), "linr": (0.721, 0.924),
                "logr": (0.733, 0.933), "ighb": (0.412, 0.814), "iglb": (0.764, 0.930)},
}
REPO = {  # reference/results/livecodebench_*/output/scores_avg_prob.txt: (BSS, ACC)
    "qwen3": {"uncalibrated": (-0.543181, 0.457955), "platt": (0.376859, 0.798485), "hb": (0.383174, 0.797348),
              "linr": (0.463083, 0.821591), "logr": (0.302645, 0.826894), "ighb": (0.223826, 0.728409),
              "iglb": (0.468046, 0.823864)},
    "gpt-oss": {"uncalibrated": (-0.0745873, 0.510606), "platt": (0.186541, 0.706818), "hb": (0.237245, 0.717424),
                "linr": (0.731528, 0.925), "logr": (0.72715, 0.931818), "ighb": (0.428296, 0.821591),
                "iglb": (0.757701, 0.924242)},
}


def run(out, n_boot=2000):
    """Replication table on test. Adds a 95% task clustered bootstrap interval for each BSS."""
    out.mkdir(parents=True, exist_ok=True)
    rows = add_scores(load_rows(include_test=True))
    table = []
    for medians, fit_splits in [("train", ("train",)), ("pooled", ("train", "validation", "test"))]:
        G, _ = build_groups(rows, REPLICATION, fit_splits)
        for model in ["qwen3", "gpt-oss"]:
            sel = lambda s: (rows.model == model) & (rows.split == s)
            tr, va, te = (part(rows, G, "avg_prob", sel(s)) for s in ["train", "validation", "test"])
            for method in METHODS:
                q, cal = calibrate(method, "code", tr, va, te)
                table.append({
                    "medians": medians, "model": model, "method": method,
                    **{f"bss_{k}": v for k, v in interval(M.brier_skill, q, te["y"], te["task"], n_boot).items()},
                    "acc": M.accuracy(q, te["y"]),
                    "paper_bss": PAPER[model][method][0], "repo_bss": REPO[model][method][0],
                    "paper_acc": PAPER[model][method][1], "repo_acc": REPO[model][method][1],
                    "brier": M.brier(q, te["y"]), "ece_grid": M.ece(q, te["y"], round_to_grid=True),
                    "max_gasce_grid": M.max_gasce(q, te["y"], te["G"], round_to_grid=True),
                    "steps": len(cal.rules) if hasattr(cal, "rules") else None,
                    "test_pass_rate": te["y"].mean(),
                })
    table = pd.DataFrame(table)
    table["bss_minus_repo"] = table.bss_value - table.repo_bss
    table["bss_minus_paper"] = table.bss_value - table.paper_bss
    table.to_csv(out / "rq1_test.csv", index=False)
    cols = ["medians", "model", "method", "bss_value", "bss_lo", "bss_hi", "repo_bss", "paper_bss", "acc", "repo_acc",
            "paper_acc", "steps"]
    print(table[cols].round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    run(Path("runs/phase8"))
