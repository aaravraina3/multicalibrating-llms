# Lab notebook

Dated entries. Decisions, bugs, surprises, numbers.

## 2026-10-03

### Research questions

1. RQ1: Can I get the same numbers as the Campos paper? If they match, my implementation probably matches theirs. That's good evidence, not proof, that my code is right.
2. RQ2: If I start from a smart model instead of raw token confidence, does multicalibration still help, or was there nothing left to fix?
3. RQ3: Does better calibration lead to better decisions about when to hand a problem to the second model?
4. RQ4 (optional): With little data, does multicalibration fix noise instead of real errors, and can I stop that?

### Reading notes

Campos §3, background:
* Calibration: among answers given 70%, about 70% pass. Checked by splitting confidence into 20 bins and comparing mean confidence to pass rate in each.
* ECE: average gap across bins. Can be fooled by predicting the same number for everyone.
* Brier: average squared error. BSS: how much better than always guessing the overall pass rate. Below 0 means worse than that guess.
* Groups: sets of answers sharing a property, like difficulty or length. They can overlap.
* Multiaccuracy: each group's average is right.
* Multicalibration: each group is calibrated at every confidence level. Measured with gASCE.

Campos §4.3, methods:
* Platt and histogram binning: fix overall calibration, ignore groups.
* LINR and LOGR: one regression that fixes each group's average. In their code LOGR outputs hard 0/1 labels, so it acts as a classifier. That's why it has high accuracy but low or negative BSS in Table 1.
* IGHB: find the worst group and bin cell, shift it, recheck, repeat. Overfits.
* IGLB: same loop with bigger overlapping cells and a smooth patch. Stops when validation stops improving. Most stable.
* The paper rounds to the grid after every IGHB/IGLB step. Their code doesn't. I reproduce the code's version and describe that one.

HJKRR §1.2, motivation:
* Calibration alone can hide unfairness. Giving everyone in a group the group's average is calibrated but ignores the strong members.
* Rain example: a predictor can be accurate yet treat two cities' outcomes differently after the fact. Before the fact you can't tell which days will rain, so you can't demand accuracy on groups only visible in hindsight.
* Compromise: require calibration on every group a simple rule can identify in advance. Not every possible subset, since that's impossible from limited data.

HJKRR, paragraph after Algorithm 3.2:
* The loop checks every big enough slice and stops once every slice is close enough.
* Then it averages predictions within each bucket.
* The result is calibrated on every big slice, and the number of updates is limited, so it always finishes.

### Phase 0 explain check

* CALIBRI: a public dataset of code generations. Each problem has 10 attempts from Qwen3 and GPT OSS, with per-token confidence and pass/fail labels from running the tests. Split by problem into train, validation, and test.
* Why read their repo but write our own code: reading gives their exact settings, so the replication is a fair comparison. Writing our own means the replication actually checks something; copied code matching their numbers would prove nothing. It's also how I end up able to explain the code.

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


## 2026-10-04

### Phase 1: data loading and validation

Dataset `lavis-nlp/CALIBRI` revision `7a4a7dc7` (same commit as `violasara/CALIBRI`, which their code loads). `src/calib/data.py` explodes to one row per generation, logprobs as float32 arrays. Train and validation go to `data/rows.parquet`, test to `data/rows_test.parquet`. `load_rows()` only returns test when called with `include_test=True`. Token strings are dropped from the cache; `load_raw` reads them from the HF cache when needed. Full check output: `runs/phase1/checks.txt`.

All checks pass:
* rows 5270 / 2640 / 2640 per model, problems 527 / 264 / 264
* no problem in two splits; same IDs, split, prompt, and difficulty for both models
* no generation without logprobs; no code span past the token count

Test split: used for IDs and counts only. No test labels looked at.

| | Qwen3 | GPT OSS |
|---|---|---|
| pass rate train | 0.445 | 0.520 |
| pass rate validation | 0.448 | 0.495 |
| empty program rate train | 0.264 | 0.428 |
| empty program rate validation | 0.284 | 0.468 |
| median tokens per output | 874 | 1468 |
| outputs truncated at 2000 tokens | 0.274 | 0.436 |

Train pass rate by difficulty: Qwen3 easy 0.810, medium 0.435, hard 0.111. GPT OSS easy 0.907, medium 0.506, hard 0.169.

Surprises:
* **Generation was capped at 2000 tokens.** Every empty program I looked at stopped at exactly 2000. Truncated outputs are mostly empty programs and almost always fail. GPT OSS outputs that finish pass 92% of the time (4363 rows); truncated non-empty ones pass 49% (55 rows). Qwen3 finished outputs pass 66%; truncated non-empty ones pass 2%. So for GPT OSS, "did it finish" explains most of the label. This is label free and known at inference, so it's a legal feature for RQ2. It also explains why `len_high` helps GPT OSS so much in Campos Table 1.
* Empty programs: pass rate exactly 0.
* 606 rows (455 Qwen3, 151 GPT OSS) have a non-empty `program` but a `[0, 0]` code span. The extracted program isn't marked in the token stream, often reasoning text that the extractor picked up. Pass rate 0.327. Their `code_prob` gives these 0.0.
* 13% of sampled token logprobs are exactly 0.

Manual examples (train): pass and fail spans contain the fenced Python block, and `program` matches the span text. Empty programs have span `[0, 0]` and 2000 tokens, cut off mid reasoning.

Explain:
* One row is one generation: one of the 10 attempts at one problem by one model, with its tokens, logprobs, extracted code, and pass/fail.
* The 10 rows of a problem share the problem's difficulty, so they pass or fail together. They are not 10 independent pieces of evidence.
* Splitting by problem keeps near-duplicate attempts at the same problem from landing on both sides, which would leak.
* Routing compares Qwen3 and GPT OSS on the same problem, so both need the same IDs with the same split. Checked: they do.

### Phase 2: synthetic sandbox

`experiments/synthetic.py`, outputs in `runs/phase2/`. Population of 100,000 with `long` (40%), `nested` (30%), `imports` (50%); `logit p* = 1.0 - 0.8 long - 0.5 nested - 1.2 (long AND imports)`. Predictor: logistic regression with main effects only, logit doubled.

* Reliability (`reliability.png`): predictor ECE 0.118, true p* ECE 0.004. Overconfident: says 0.92 where 0.72 pass, 0.10 where 0.19 pass. Distance to truth 0.0179.
* HJKRR loop (`src/calib/calibrators/hjkrr.py`): 10 bins; groups all, long, nested, imports, long AND imports. Shift any slice with at least `min_size` rows and gap above `alpha`; bins recomputed before every check; stop when a full pass makes no update.
* Distance per update (`distance_per_update.png`):
  * large slices (n=100,000, min_size=500, alpha=0.02): 8 updates, none moved away from the truth. 0.0179 to 0.0003. ECE after 0.008.
  * small slices (n=1,000, min_size=10, alpha=0.005): 10 updates, 2 moved away from the truth. Winner's curse as expected.
* Winner's curse (`winners_curse.png`): p* as predictor, 30 slices of 200, 2000 repeats. Biggest gap averages 0.077. On fresh labels, the unfixed slice is off by 0.026, the "fixed" slice by 0.076. Fixing made it worse in 86% of repeats.
* Overfitting vs n (`overfitting_vs_n.png`), min_size=10, alpha=0.01, 20 repeats, fresh 100,000 for evaluation. Mean distance to truth:

| n | from overconfident (start 0.0178) | from p* (start 0) |
|---|---|---|
| 200 | 0.0089 | 0.0085 |
| 500 | 0.0039 | 0.0035 |
| 2000 | 0.0009 | 0.0008 |
| 10000 | 0.0002 | 0.0001 |

  Starting from p*, every change is fitted noise, and the damage at n=200 is half the size of the original miscalibration. Two of the 20 n=200 runs from the overconfident start ended worse than they began.

Explain:
* Calibration: among rows predicted near v, the pass rate is about v.
* Each correction moves predictions toward the truth when the measured gap is close to the real gap: shifting a slice by its real average error always lowers squared distance to p* (Lemma 3.6). The plot confirms it for big slices.
* The biggest of 30 noisy gaps is mostly noise: each slice of 200 has standard error about 0.03, so the max of 30 lands near 2.5 standard errors even when nothing is wrong.
* Small samples overfit because every slice estimate is noisy, so the loop fixes noise. The error shrinks roughly like 1/n.
* The loop is adaptive: which slice it checks next depends on predictions it already changed using the same labels. So checking slices on the data you fixed overstates how calibrated you are. HJKRR handle this with the guess and check oracle and differential privacy (§3.3).
