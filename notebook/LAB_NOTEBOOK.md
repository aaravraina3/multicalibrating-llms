# Lab notebook

Dated entries. Decisions, bugs, surprises, numbers.

## 2026-10-03

### Settings from reference repo (Phase 0, step 8)

Source: `github.com/violacampos/multicalibration`, commit `c9b7e5d` (2026-01-20).

| Setting | Their code | Where |
|---|---|---|
| `avg_prob` | `exp(sum(logprobs) / n_tokens)` over full output, reasoning included | `data/dataset.py` `add_features` |
| `code_prob` | same over `tokens[start:end]`; empty span gives 0.0, not NaN | same |
| `tail_prob` | `exp(sum(last 40) / 40)`; always divides by 40, even if output is shorter | same |
| Bins | grid of 21 points: 0, 0.05, ..., 1.0 (`--bin-count 20`) | `tools/binning.py` |
| LiveCodeBench groups | 6: `loc_high` (newline count in program > median), `prompt_len_high` (prompt chars > median), `len_high` (full output chars > median), `comp_easy`, `comp_medium`, `comp_hard`. No low groups, no `all` group | `compare_methods.py`, `data/dataset.py` |
| Group medians | pooled over train + validation + test | `data/dataset.py` `get_median_*` |
| Platt | `LogisticRegression()` on raw `p`, default `C=1` | `methods/platt_calibration.py` |
| HB | round to nearest grid point; shift = `mean(y) - grid value`; empty bins shift 0 | `methods/hb_calibration.py` |
| LINR | `LinearRegression` of `y` on `[p, groups]` with intercept; `p` gets a learned weight; no clip | `methods/lr_calibration.py`, `run_lr.py` |
| LOGR | `LogisticRegression()` on `[p, groups]`, but `predict()` returns hard 0/1 labels | same |
| IGHB stop | `max_g P(g) * gASCE(g) <= 1/20`; no iteration cap | `methods/ighb_calibration.py` |
| IGHB update | 20 disjoint intervals via `np.digitize`; add delta to raw `p` in cell, clip to [0, 1], no rounding back to grid; `p = 1.0` lands in no bin | same |
| IGLB | `epsilon = 0.01`; thresholds at the 21 grid points, both `<=` and `>=`; patch `sigmoid(a + b * logit p)` fit by BFGS on Brier from `(0, 1)`; stop if cell share < epsilon or validation Brier does not drop; no cap, no rounding | `methods/iglb_calibration.py` |
| BSS base rate | from the evaluated set itself (test) | `tools/calibration_scores.py` |
| ECE, gASCE | on probabilities rounded to nearest grid point, confidence = grid value | same |
| Brier | on raw probabilities | same |
| Accuracy | `p > 0.5` (paper says `>= 0.5`) | same |
| Empty programs | included in Table 1; excluded only in Table 2 | paper §6.2 |
| Fit data | all methods fit on train; IGLB also uses validation for early stopping; scored on test | `methods/run_*.py` |
| Dataset ID | code loads `violasara/CALIBRI`; paper links `lavis-nlp/CALIBRI`. Check in Phase 1 | `data/dataset.py` |

### Code vs paper vs our plan

1. Platt: paper Eq. 15 says `sigmoid(a * log p + b)`. Code fits on raw `p`, `C=1`. Table 1 Platt numbers match the code output exactly.
2. LINR: plan has `p` fixed at weight 1 plus group shifts, clipped. Code learns a weight on `p`, no clip.
3. LOGR: hard labels, so its Brier equals 1 minus accuracy (visible in their results file).
4. IGHB and IGLB: paper rounds to grid each step. Code never rounds.
5. Groups: plan Phase 6 has code length low/high. Code uses output length and prompt length, high only.

### Two sets of Phase 8 targets (LiveCodeBench, `avg_prob`, BSS)

| Method | Paper Qwen3 | Repo Qwen3 | Paper GPT OSS | Repo GPT OSS |
|---|---|---|---|---|
| uncalibrated | -0.543 | -0.543 | -0.075 | -0.075 |
| Platt | 0.377 | 0.377 | 0.187 | 0.187 |
| HB | 0.383 | 0.383 | 0.237 | 0.237 |
| LINR | 0.463 | 0.463 | 0.721 | 0.732 |
| LOGR | 0.306 | 0.303 | 0.733 | 0.727 |
| IGHB | 0.232 | 0.224 | 0.412 | 0.428 |
| IGLB | 0.480 | 0.468 | 0.764 | 0.758 |

Repo numbers from `reference/results/livecodebench_*/output/scores_avg_prob.txt`. Group methods differ slightly between paper and repo; group-agnostic ones match.

