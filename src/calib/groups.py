"""Group definitions. Groups use only label free row properties; thresholds come from train only."""

import json

import numpy as np
import pandas as pd

from calib.features import code_structure

REPLICATION = ["loc_high", "prompt_len_high", "len_high", "comp_easy", "comp_medium", "comp_hard"]
EXTENDED = REPLICATION + ["nested", "uses_imports", "syntax_invalid", "truncated", "all"]
MAX_TOKENS = 2000


def raw_properties(rows):
    """Per-row values that groups are thresholded on."""
    s = pd.DataFrame([code_structure(c) for c in rows.code], index=rows.index)
    return pd.DataFrame({
        "loc": rows.code.str.count("\n"),  # newline count, as in their code
        "prompt_len": rows.prompt.str.len(),
        "output_len": rows.output.str.len(),
        "nesting": s.nesting,
        "imports": s.imports,
        "syntax_invalid": ~s.empty_code & ~s.syntax_valid,
        "truncated": rows.logprobs.map(len) >= MAX_TOKENS,
        "difficulty": rows.difficulty,
    }, index=rows.index)


def fit_thresholds(props_train):
    """Medians over train rows (each problem counted 10 times, like their code). Nesting median over parsed code only."""
    return {"loc": float(props_train["loc"].median()),
            "prompt_len": float(props_train.prompt_len.median()),
            "output_len": float(props_train.output_len.median()),
            "nesting": float(props_train.nesting.dropna().median())}


def group_matrix(props, thresholds, names=REPLICATION):
    """Boolean n x G matrix, columns in the order of names."""
    cols = {
        "loc_high": props["loc"] > thresholds["loc"],
        "prompt_len_high": props.prompt_len > thresholds["prompt_len"],
        "len_high": props.output_len > thresholds["output_len"],
        "comp_easy": props.difficulty == "easy",
        "comp_medium": props.difficulty.isin(["medium", "middle"]),
        "comp_hard": props.difficulty == "hard",
        "nested": props.nesting.fillna(-1) > thresholds["nesting"],
        "uses_imports": props.imports.fillna(0) > 0,
        "syntax_invalid": props.syntax_invalid,
        "truncated": props.truncated,
        "all": pd.Series(True, index=props.index),
    }
    return np.column_stack([cols[n].to_numpy(bool) for n in names])


def save_thresholds(thresholds, path):
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)


def build_groups(rows, names=REPLICATION, fit_splits=("train",)):
    """Group matrix for all rows, thresholds fit per model on that model's rows in fit_splits.
    Default is train only. fit_splits=("train", "validation", "test") reproduces their pooled medians."""
    props = raw_properties(rows)
    G = np.zeros((len(rows), len(names)), bool)
    thresholds = {}
    for model in rows.model.unique():
        m = (rows.model == model).to_numpy()
        thresholds[model] = fit_thresholds(props[m & rows.split.isin(fit_splits).to_numpy()])
        G[m] = group_matrix(props[m], thresholds[model], names)
    return G, thresholds


def keep_groups(G, rows, names, min_problems=40):
    """Names of groups with at least min_problems train problems for every model in rows."""
    keep = []
    for g, name in enumerate(names):
        ok = all(rows[(rows.model == m).to_numpy() & (rows.split == "train").to_numpy() & G[:, g]].task_id.nunique()
                 >= min_problems for m in rows.model.unique())
        if ok:
            keep.append(name)
    return keep
