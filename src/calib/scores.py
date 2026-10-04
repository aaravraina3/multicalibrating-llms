"""Starting scores: geometric mean token probability over different token ranges."""

import numpy as np


def avg_prob(logprobs):
    """exp(mean logprob) over every output token, reasoning included."""
    return float(np.exp(logprobs.mean()))


def code_prob(logprobs, start, end):
    """Same over the code span. NaN when the span is [0, 0] (their code returns 0.0; see add_scores)."""
    if end <= start:
        return float("nan")
    return float(np.exp(logprobs[start:end].mean()))


def tail_prob(logprobs, k=40):
    """Their code divides the last-k sum by k even when the output is shorter than k."""
    return float(np.exp(logprobs[-k:].sum() / k))


def add_scores(rows):
    """Adds avg_prob, code_prob (NaN when no span), code_prob_campos (0.0 when no span), tail_prob."""
    rows = rows.copy()
    rows["avg_prob"] = rows.logprobs.map(avg_prob)
    rows["code_prob"] = [code_prob(lp, s, e) for lp, s, e in zip(rows.logprobs, rows.code_start, rows.code_end)]
    rows["code_prob_campos"] = rows.code_prob.fillna(0.0)
    rows["tail_prob"] = rows.logprobs.map(tail_prob)
    return rows
