"""TreeSHAP helpers for B4. Main effects: interventional mode, probability scale, 200 row background.
Interactions: shap only supports them in path dependent mode on the log odds scale, so they're labeled that way."""

import numpy as np
import shap

from calib.features import BLOCKS

BLOCK_OF = {"logprob statistics": BLOCKS["avg"] + BLOCKS["logprob"], "size": BLOCKS["size"],
            "AST structure": BLOCKS["structure"], "self consistency": BLOCKS["self_consistency"]}


def block_index(xcols):
    return {b: [xcols.index(c) for c in cols if c in xcols] for b, cols in BLOCK_OF.items()}


def background(X_base, n=200, seed=0):
    return X_base[np.random.default_rng(seed).choice(len(X_base), n, replace=False)]


def explain_probability(model, X_bg, X):
    """Interventional TreeSHAP on the probability scale. Rows sum to p(x) - mean p over the background."""
    e = shap.TreeExplainer(model, data=X_bg, feature_perturbation="interventional", model_output="probability")
    return e.shap_values(X), float(e.expected_value)


def interactions_logodds(model, X):
    """Path dependent SHAP interaction values on the log odds scale, shape (n, f, f)."""
    return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_interaction_values(X)


def block_shap(sv, xcols):
    """Per row SHAP summed inside each feature block."""
    return {b: sv[:, idx].sum(1) for b, idx in block_index(xcols).items()}


def top_pairs(iv, xcols, k=5):
    """Feature pairs ranked by mean |interaction value| (off diagonal)."""
    m = np.abs(iv).mean(0)
    pairs = [(m[i, j], xcols[i], xcols[j]) for i in range(len(xcols)) for j in range(i + 1, len(xcols))]
    return [(a, b, float(v)) for v, a, b in sorted(pairs, reverse=True)[:k]]
