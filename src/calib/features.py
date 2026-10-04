"""Label free features for each generation. Nothing here may read y."""

import ast
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

MAX_TOKENS = 2000

NEST_NODES = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
          ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _nesting(node, depth=0):
    """Deepest stack of nested blocks (loops, ifs, functions, ...) under node."""
    best = depth
    for child in ast.iter_child_nodes(node):
        best = max(best, _nesting(child, depth + isinstance(child, NEST_NODES)))
    return best


def code_structure(code):
    """AST counts. For empty or unparsable code, counts are NaN and the flags say why."""
    out = {"empty_code": code.strip() == "", "syntax_valid": False, "nesting": np.nan, "branches": np.nan,
           "loops": np.nan, "functions": np.nan, "imports": np.nan}
    if out["empty_code"]:
        return out
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return out
    nodes = list(ast.walk(tree))
    out.update(syntax_valid=True, nesting=_nesting(tree),
               branches=sum(isinstance(n, (ast.If, ast.IfExp, ast.Match)) for n in nodes),
               loops=sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While, ast.comprehension)) for n in nodes),
               functions=sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for n in nodes),
               imports=sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes))
    return out


def logprob_stats(lp, prefix):
    """Mean, 10th percentile, standard deviation, share below -2. NaN when there are no tokens."""
    if len(lp) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_p10": np.nan, f"{prefix}_std": np.nan, f"{prefix}_low": np.nan}
    return {f"{prefix}_mean": float(lp.mean()), f"{prefix}_p10": float(np.percentile(lp, 10)),
            f"{prefix}_std": float(lp.std()), f"{prefix}_low": float((lp < -2).mean())}


def normalize_code(code):
    """Strip # comments and collapse whitespace."""
    return " ".join(re.sub(r"#.*", "", code).split())


def self_consistency(rows):
    """Leave one out within (model, problem): each generation vs the other 9. Uses code text only, never labels.
    Assumes all 10 samples exist at inference, which costs 10x the generation."""
    out = pd.DataFrame(np.nan, index=rows.index, columns=["sc_others_empty", "sc_mean_sim", "sc_identical"])
    for _, idx in rows.groupby(["model", "task_id"]).groups.items():
        codes = [normalize_code(c) for c in rows.loc[idx, "code"]]
        k = len(codes)
        sim = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(i + 1, k):
                if codes[i] and codes[j]:
                    sim[i, j] = sim[j, i] = SequenceMatcher(None, codes[i], codes[j]).ratio()
        for i, row in enumerate(idx):
            others = [j for j in range(k) if j != i]
            out.at[row, "sc_others_empty"] = np.mean([codes[j] == "" for j in others])
            if codes[i]:
                s = sim[i, others]
                out.at[row, "sc_mean_sim"] = np.nanmean(s) if np.isfinite(s).any() else np.nan
                out.at[row, "sc_identical"] = np.mean([codes[j] == codes[i] for j in others])
    return out


BLOCKS = {
    "avg": ["avg_prob"],
    "size": ["log_prompt_chars", "log_code_chars", "log_code_lines", "log_output_tokens", "truncated"],
    "logprob": ["all_mean", "all_p10", "all_std", "all_low", "code_mean", "code_p10", "code_std", "code_low",
                "code_lp_missing"],
    "structure": ["nesting", "branches", "loops", "functions", "imports", "syntax_valid", "empty_code",
                  "structure_missing"],
    "self_consistency": ["sc_others_empty", "sc_mean_sim", "sc_identical", "sc_missing"],
}


def build_features(rows, with_self_consistency=True):
    """One row of label free features per generation. NaNs are kept; prepare_matrix fills them."""
    n_tok = rows.logprobs.map(len)
    f = pd.DataFrame({
        "log_prompt_chars": np.log1p(rows.prompt.str.len()),
        "log_code_chars": np.log1p(rows.code.str.len()),
        "log_code_lines": np.log1p(rows.code.map(lambda c: len(c.splitlines()))),
        "log_output_tokens": np.log1p(n_tok),
        "truncated": (n_tok >= MAX_TOKENS).astype(float),
        "avg_prob": rows.logprobs.map(lambda lp: float(np.exp(lp.mean()))),
    }, index=rows.index)
    lp_all = pd.DataFrame([logprob_stats(lp, "all") for lp in rows.logprobs], index=rows.index)
    lp_code = pd.DataFrame([logprob_stats(lp[s:e], "code") for lp, s, e in
                            zip(rows.logprobs, rows.code_start, rows.code_end)], index=rows.index)
    struct = pd.DataFrame([code_structure(c) for c in rows.code], index=rows.index).astype(float)
    f = pd.concat([f, lp_all, lp_code, struct], axis=1)
    f["code_lp_missing"] = lp_code.code_mean.isna().astype(float)
    f["structure_missing"] = struct.nesting.isna().astype(float)
    if with_self_consistency:
        sc = self_consistency(rows)
        f = pd.concat([f, sc], axis=1)
        f["sc_missing"] = sc.sc_mean_sim.isna().astype(float)
    return f


def prepare_matrix(features, blocks):
    """Columns for the chosen blocks, NaN filled with 0 (each NaN source has a missing flag)."""
    cols = [c for b in blocks for c in BLOCKS[b]]
    return features[cols].fillna(0.0).to_numpy(float), cols
