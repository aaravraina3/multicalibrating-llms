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

### Phase 3: metrics and evaluation tools

`src/calib/metrics.py`, `src/calib/bootstrap.py`, `src/calib/plots.py`, `tests/test_metrics.py`. 8 tests pass.

Decisions:
* BSS base rate from the evaluated rows (Campos convention).
* Accuracy is `p > 0.5`, strict, as in their code.
* `ece` and `gasce` take `round_to_grid`. Off: 20 equal width bins, last includes 1.0, confidence is each row's own p. On: Campos code, round to the nearest of 21 grid points with `argmin |p - grid|` (exact ties go to the lower point, tested at 0.375), and the grid value is the confidence. Replication tables use `round_to_grid=True`; everything else uses off.
* Bootstrap: resample problem IDs with replacement, 2000 resamples, seed 0, take all rows of each drawn problem, 2.5 and 97.5 percentiles. `paired_diff` scores both methods on the same resample.

Explain:
* Brier measures overall probability quality: calibration and separation together. ECE measures only calibration and ignores separation. AUROC measures only ranking and ignores calibration entirely. Each misses what the others catch.
* Predicting the base rate for everyone puts every row in one bin whose mean prediction equals its pass rate, so ECE is 0 while the predictions are useless (tested).
* ECE can be 0 overall while one group is off at 90% and another at 30% in opposite directions. gASCE measures calibration inside each group, so it catches that.
* The 10 rows of a problem are correlated. Resampling rows would treat 2640 test rows as independent and give intervals that are too narrow. Resampling problems matches the real amount of evidence, about 264.

### Phase 4: starting scores (validation only)

`src/calib/scores.py`, `experiments/scores_validation.py`, outputs in `runs/phase4/`. `code_prob_campos` is their version: 0.0 when the code span is `[0, 0]`. `tail_prob` divides the last 40 logprobs by 40 like their code; a few outputs are shorter than 40 tokens (minimum 17), where this pulls the score down.

All validation rows:

| model | score | mean p | pass rate | BSS | ECE | AUROC |
|---|---|---|---|---|---|---|
| Qwen3 | avg_prob | 0.879 | 0.448 | -0.549 | 0.432 | 0.853 |
| Qwen3 | code_prob_campos | 0.629 | 0.448 | 0.187 | 0.200 | 0.880 |
| Qwen3 | tail_prob | 0.905 | 0.448 | -0.663 | 0.458 | 0.774 |
| GPT OSS | avg_prob | 0.712 | 0.495 | -0.098 | 0.217 | 0.768 |
| GPT OSS | code_prob_campos | 0.481 | 0.495 | 0.816 | 0.023 | 0.957 |
| GPT OSS | tail_prob | 0.861 | 0.495 | -0.246 | 0.366 | 0.843 |

Rows with a code span (the Campos Table 2 comparison, which excludes rows without code):

| model | score | pass rate | BSS | ECE | AUROC |
|---|---|---|---|---|---|
| Qwen3 | avg_prob | 0.674 | -0.174 | 0.252 | 0.801 |
| Qwen3 | code_prob | 0.674 | -0.341 | 0.293 | 0.720 |
| Qwen3 | tail_prob | 0.674 | -0.380 | 0.300 | 0.595 |
| GPT OSS | avg_prob | 0.936 | -0.689 | 0.198 | 0.527 |
| GPT OSS | code_prob | 0.936 | 0.002 | 0.014 | 0.579 |
| GPT OSS | tail_prob | 0.936 | -0.024 | 0.036 | 0.492 |

Notes:
* All three scores are badly overconfident. Qwen3 avg_prob averages 0.88 against a 0.45 pass rate.
* On rows with code, avg_prob wins for Qwen3, as in Campos Table 2 (their test numbers: avg_prob ECE 0.294, BSS -0.257; code_prob ECE 0.342, BSS -0.453). Same ordering here.
* On all rows, `code_prob_campos` looks best, but only because rows without code get 0.0 and those always fail. It's an "is there code" flag in disguise, not a better confidence.
* GPT OSS rows with code pass 94% of the time, and no token score ranks them (AUROC 0.49 to 0.58). For GPT OSS nearly all the signal is whether it finished and produced code (Phase 1 truncation finding).

Explain:
* Token confidence says how sure the model is about the next token, not whether the program passes. Fluent, confident text can implement the wrong algorithm, so the scores sit far above the pass rate.
* Reasoning tokens might help because hesitant reasoning (low probability tokens while working out the approach) signals a hard problem, and that shows up before any code is written.

### Phase 5: group agnostic calibrators

`src/calib/calibrators/platt.py` (versions `code`, `paper`, `logit`), `histogram.py` (HB and isotonic), `tests/test_calibrators_basic.py`, `experiments/group_agnostic_validation.py`. Fit on official train, evaluated on validation, starting score avg_prob. 11 tests pass: logit Platt returns identity on calibrated input (a and b within 0.05 of 1 and 0); every Platt version keeps ranking; HB cuts train ECE by more than 5x on an overconfident input.

| model | method | BSS | log loss | ECE | ACC | AUROC | Platt a, b |
|---|---|---|---|---|---|---|---|
| Qwen3 | uncalibrated | -0.549 | 1.135 | 0.432 | 0.448 | 0.853 | |
| Qwen3 | Platt code (raw p, C=1) | 0.375 | 0.470 | 0.064 | 0.789 | 0.853 | 14.5, -13.3 |
| Qwen3 | Platt paper (log p) | 0.385 | 0.461 | 0.039 | 0.789 | 0.853 | 16.1, 1.36 |
| Qwen3 | Platt logit | 0.352 | 0.487 | 0.086 | 0.771 | 0.853 | 1.16, -3.30 |
| Qwen3 | HB | 0.375 | 0.472 | 0.048 | 0.786 | 0.847 | |
| Qwen3 | isotonic (ours) | 0.373 | 0.469 | 0.050 | 0.782 | 0.850 | |
| GPT OSS | uncalibrated | -0.098 | 0.747 | 0.217 | 0.495 | 0.768 | |
| GPT OSS | Platt code | 0.197 | 0.593 | 0.086 | 0.715 | 0.768 | 13.7, -9.71 |
| GPT OSS | Platt paper | 0.215 | 0.585 | 0.054 | 0.716 | 0.768 | 13.4, 4.63 |
| GPT OSS | Platt logit | 0.202 | 0.600 | 0.071 | 0.710 | 0.768 | 3.45, -3.12 |
| GPT OSS | HB | 0.226 | 0.571 | 0.021 | 0.689 | 0.762 | |
| GPT OSS | isotonic (ours) | 0.232 | 0.571 | 0.021 | 0.714 | 0.770 | |

Done: Platt and HB beat uncalibrated on validation BSS for both models. Validation numbers already sit near Campos's test numbers (Platt 0.377 / 0.187, HB 0.383 / 0.237).

Notes:
* Platt AUROC equals uncalibrated exactly in every version: it never changes the order. HB and isotonic change AUROC slightly because they merge neighboring scores into ties.
* With raw p, a is about 14. All avg_prob values sit in a narrow high band (most between 0.8 and 1), so the slope has to be steep to spread them over [0, 1]. "a below 1 shrinks toward the middle" only applies to the logit version.
* The paper version (log p) beats the code version slightly on both models. Phase 8 still uses the code version to match their table.

Explain:
* Platt's a sets how steep the curve is (how much to trust differences in the score); b shifts everything up or down.
* HB can fix any shape because each of the 21 grid points gets its own shift, while Platt is locked to one S curve.
* Both look only at the score. Two answers with the same score get the same output, whatever their difficulty or length. That's the gap multicalibration fills.

### Phase 6: groups

`src/calib/groups.py`, `src/calib/features.py` (AST helper), `experiments/groups_table.py`, outputs in `runs/phase6/`.

Replication set, matching their code: `loc_high` (newline count of the program > median), `prompt_len_high` (prompt characters > median), `len_high` (full output characters > median), `comp_easy`, `comp_medium`, `comp_hard`. Same column order as their code, since it decides ties in the argmax. No `all` group. Medians are over train rows only, per model (known deviation: theirs pools all splits).

Extended set for RQ2 and RQ3: the six plus `nested` (AST block nesting depth above the train median of 4), `uses_imports`, `syntax_invalid` (non-empty code that fails `ast.parse`), `truncated` (output hit the 2000 token cap, from Phase 1), and `all`. `truncated` is my addition, motivated by Phase 1. Groups with fewer than 40 train problems are dropped per model (`keep_groups`). The "long prompt" group from the plan is the same as `prompt_len_high`.

Train medians: Qwen3 loc 18, output 3014 chars; GPT OSS loc 13, output 4521.5 chars; prompt 1324 chars for both (same prompts).

| group | Qwen3 train problems | Qwen3 val problems | Qwen3 train pass rate | GPT OSS train problems | GPT OSS val problems | GPT OSS train pass rate |
|---|---|---|---|---|---|---|
| loc_high | 362 | 191 | 0.502 | 374 | 181 | 0.903 |
| prompt_len_high | 263 | 139 | 0.304 | 263 | 139 | 0.395 |
| len_high | 338 | 170 | 0.177 | 347 | 193 | 0.134 |
| comp_easy | 164 | 73 | 0.810 | 164 | 73 | 0.907 |
| comp_medium | 190 | 96 | 0.435 | 190 | 96 | 0.506 |
| comp_hard | 173 | 95 | 0.111 | 173 | 95 | 0.169 |
| nested | 213 | 106 | 0.490 | 178 | 86 | 0.865 |
| uses_imports | 104 | 63 | 0.337 | 348 | 170 | 0.909 |
| syntax_invalid | 104 | 60 | 0.000 | 3 | 2 | 0.000 (dropped) |
| truncated | 226 | 133 | 0.004 | 324 | 173 | 0.006 |
| all | 527 | 264 | 0.445 | 527 | 264 | 0.520 |

Group size counts problems with at least one row in the group, so a problem can count in a group and in its complement.

Explain:
* Groups have to be computable without labels because at deployment you assign a new answer to its groups before knowing if it passes. A group defined by the label would be a perfect cheat that can't be applied.
* Thresholds come from train only so nothing about validation or test leaks into the method. Pooling medians over all splits, as their code does, uses test features, a small leak.
* Difficulty is benchmark metadata. A real coding assistant doesn't know a problem's difficulty label, so results that depend on it may not carry over. RQ2 and RQ3 run with and without it.

### Phase 7: multicalibration methods

`src/calib/calibrators/{linr,logr,ighb,iglb}.py`, each with `version="code"` (default, their repo) or `"paper"` (their algorithms as written). `src/calib/pipeline.py` holds the shared fit/predict plumbing. `tests/test_multicalibration.py`: 11 tests, 22 total pass.
* IGHB and IGLB (both versions) cut max gASCE on a fresh synthetic sample by more than half.
* Replaying saved rules on train reproduces the fitted train predictions exactly (both versions, both methods).
* LINR (both versions) zeroes every group's mean residual on train.
* LOGR code outputs only 0 and 1; paper version outputs probabilities.

Validation table, replication groups, avg_prob, fit on train. IGLB early stops on validation and is scored on validation, so its numbers here are optimistic. Full table: `runs/phase7/validation.csv`.

| method | version | Qwen3 BSS | Qwen3 max gASCE | GPT OSS BSS | GPT OSS max gASCE | steps (Q / G) |
|---|---|---|---|---|---|---|
| uncalibrated | | -0.549 | 0.496 | -0.098 | 0.298 | |
| Platt | code | 0.375 | 0.062 | 0.197 | 0.133 | |
| HB | | 0.375 | 0.057 | 0.226 | 0.123 | |
| LINR | code | 0.458 | 0.018 | 0.771 | 0.040 | |
| LINR | paper | 0.465 | 0.016 | 0.766 | 0.040 | |
| LOGR | code (hard labels) | 0.265 | 0.082 | 0.824 | 0.010 | |
| LOGR | paper (probabilities) | 0.474 | 0.021 | 0.821 | 0.017 | |
| IGHB | code | 0.184 | 0.125 | 0.477 | 0.086 | 5 / 3 |
| IGHB | paper | 0.117 | 0.150 | 0.452 | 0.076 | 5 / 3 |
| IGLB | code | 0.465 | 0.022 | 0.827 | 0.004 | 8 / 5 |
| IGLB | paper | 0.375 | 0.037 | 0.826 | 0.002 | 4 / 5 |

Ordering matches Campos Table 1: IGLB and LINR on top, Platt and HB in the middle, IGHB low. GPT OSS gains are huge because the groups (`len_high`, difficulty) carry the truncation signal raw token probability misses.

**Finding: IGHB is underfit, not overfit.** Campos say IGHB overfits. In their code it stops when max P(g) gASCE(g) <= 1/M = 0.05, which is loose. It stops after 3 to 5 steps with train BSS as bad as validation BSS (`runs/phase7/ighb_alpha_sweep.csv`):

| alpha | Qwen3 steps | Qwen3 train BSS | Qwen3 val BSS | GPT OSS steps | GPT OSS train BSS | GPT OSS val BSS |
|---|---|---|---|---|---|---|
| 0.05 (Campos) | 5 | 0.131 | 0.184 | 3 | 0.466 | 0.477 |
| 0.02 | 7 | 0.301 | 0.294 | 9 | 0.638 | 0.660 |
| 0.01 | 11 | 0.367 | 0.348 | 13 | 0.702 | 0.733 |
| 0.003 | 16 | 0.417 | 0.406 | 19 | 0.749 | 0.778 |
| 0.001 | 29 | 0.451 | 0.456 | 29 | 0.775 | 0.802 |
| 0.0003 | 65 | 0.477 | 0.453 | 42 | 0.790 | 0.815 |

Tighter alpha helps validation until about 0.001; for Qwen3 the train/validation gap opens at 0.0003, the first sign of overfitting. Phase 8 still uses 0.05 to match their table; Phase 10 picks alpha on validation for RQ2 and RQ3. Same effect on the synthetic data: the Phase 2 predictor already passes the 0.05 rule, so Campos IGHB makes zero updates (tested).

Other notes:
* LOGR code on GPT OSS scores BSS 0.824 with hard labels, because hard labels are right 95.6% of the time there. Brier = 1 - accuracy = 0.044.
* Paper IGLB on Qwen3 first looped 1000 steps: the chosen cell was `p >= 1.0` within `comp_easy`, where the logit patch can't move (logit of 1 - 1e-10 is about 23, the gradient vanishes). The unrounded patch gained about 1e-10 in validation Brier, rounding undid it, repeat. Fix: judge the rounded result. Commented in the code.
* Rounding (paper versions) hurts Qwen3: rounded IGLB 0.375 vs unrounded 0.465.

Explain:
* IGHB loop: measure every group x bin cell's average error, pick the cell where (share of rows) x (error squared) is largest, shift that cell by its error, then measure again, because the shift moved rows between bins and changed other cells it overlaps. Stop when every group's weighted error is small.
* Rules are replayed in order because each rule's cell is defined by the predictions at that moment. Applying them in a different order selects different rows.
* IGLB uses cells like "all rows in group g with p <= 0.6", much bigger than one bin, so the error estimate is less noisy. Its patch is a small Platt curve fit on the cell, which keeps the order inside the cell instead of adding one flat shift. It stops when validation Brier stops improving.
* IGHB is known to overfit when run long on small cells (Phase 2, Globus-Harris et al.). With the Campos stopping rule it doesn't run long enough to overfit here; its weak numbers come from stopping early.

### Phase 8: replication checkpoint (RQ1)

Rule, written before looking at any test number: after seeing these test numbers I may only fix bugs. Every fix gets logged here with what changed and why. No setting changes. Tagged `v-replication` before the first test run.

`experiments/rq1_replication.py`, output `runs/phase8/rq1_test.csv`. avg_prob, code faithful versions, replication groups, fit on train, IGLB early stops on validation, scored on test. Run twice: medians from train only (our rule) and medians pooled over all splits (their code).

Test BSS (ACC in the csv):

| method | Qwen3 ours, train medians | Qwen3 paper | Qwen3 ours, pooled | Qwen3 repo | GPT OSS ours, train medians | GPT OSS paper | GPT OSS ours, pooled | GPT OSS repo |
|---|---|---|---|---|---|---|---|---|
| uncalibrated | -0.543 | -0.543 | -0.543 | -0.543 | -0.075 | -0.075 | -0.075 | -0.075 |
| Platt | 0.377 | 0.377 | 0.377 | 0.377 | 0.187 | 0.187 | 0.187 | 0.187 |
| HB | 0.383 | 0.383 | 0.383 | 0.383 | 0.237 | 0.237 | 0.237 | 0.237 |
| LINR | 0.463 | 0.463 | 0.463 | 0.463 | 0.721 | 0.721 | 0.732 | 0.732 |
| LOGR | 0.306 | 0.306 | 0.303 | 0.303 | 0.733 | 0.733 | 0.727 | 0.727 |
| IGHB | 0.232 | 0.232 | 0.224 | 0.224 | 0.412 | 0.412 | 0.428 | 0.428 |
| IGLB | 0.478 | 0.480 | 0.468 | 0.468 | 0.768 | 0.764 | 0.758 | 0.758 |

Result:
* Pooled medians reproduce their repo's results file to 6 decimals on 13 of 14 numbers (BSS and accuracy). The 14th, GPT OSS IGHB, differs by 0.00001 (0.428307 vs 0.428296), float noise.
* Train-only medians reproduce the paper's Table 1 to its printed 3 decimals for every method except IGLB (Qwen3 0.478 vs 0.480, GPT OSS 0.768 vs 0.764; accuracy within 0.005).
* So the paper vs repo gap is the median convention. The paper's numbers look like they came from train-only medians, and the current repo pools all splits. My guess is an earlier version of their code computed medians per split. The remaining IGLB gap is small and in both directions; likely an optimizer or code version difference in the patch fit.
* Method ordering matches: IGLB and LINR top, then LOGR, then HB and Platt, then IGHB, all above uncalibrated.

Bug fixes after seeing test: none.

Explain:
* Replicate first because extensions only mean something if the base pipeline is right. If our Platt or IGLB were off, RQ2 and RQ3 differences could be our bugs.
* A mismatch could mean a bug, a different setting (medians, rounding, regularization), a different data version, or randomness. Here the one systematic mismatch traced to a setting.
* Test is allowed here under the bug-fix-only rule: nothing tuned after looking, so the numbers stay an honest check rather than a target I optimized for.

### Phase 9: features and base predictors (RQ2 setup)

`src/calib/splits.py`, `src/calib/features.py`, `src/calib/base_model.py`, `experiments/base_models.py`, `tests/test_features.py` (25 tests pass, including one that flips every label and checks the features don't change). Outputs in `runs/phase9/`; features cached to `data/features.parquet`, base model predictions to `data/base_preds.parquet`.

Split: official train problems sorted by `sha256("calib-routing-2026" + id)`; first 316 go to `base_train`, last 211 to `calib`. Same assignment for both models (`runs/phase9/splits.json`).

Features (one row per generation, no labels):
* avg: avg_prob
* size: log1p prompt chars, code chars, code lines, output tokens; `truncated` (hit 2000 tokens, my addition from Phase 1)
* logprob: mean, 10th percentile, std, share below -2, over all tokens and over code tokens; missing flag when no code span
* structure (Python `ast`): max block nesting, branches, loops, functions, imports, `syntax_valid`, `empty_code`, missing flag
* self consistency (leave one out vs the other 9 samples of the same problem and model): share of others empty, mean `SequenceMatcher` ratio of normalized code (comments stripped, whitespace collapsed), share identical. Caveat: assumes 10 samples at inference, 10x the generation cost.
* NaNs filled with 0; every NaN source has a missing flag.
* Difficulty is never a base model feature.

Base models fit on base_train, picked by validation log loss. Logistic: `StandardScaler` + `LogisticRegression`, C from {0.01, 0.1, 1, 10}. B3: `HistGradientBoostingClassifier`, 8 leaves, `min_samples_leaf=50`, `early_stopping=False`, `max_iter` from {50, 100, 200, 400} x `learning_rate` from {0.03, 0.1}.

| model | variant | chosen | val BSS | val log loss | val ECE | val AUROC |
|---|---|---|---|---|---|---|
| Qwen3 | B0 avg_prob only | C=10 | 0.386 | 0.461 | 0.037 | 0.853 |
| Qwen3 | B1 no self consistency | C=1 | 0.563 | 0.329 | 0.045 | 0.933 |
| Qwen3 | B2 all features | C=0.01 | 0.585 | 0.317 | 0.045 | 0.939 |
| Qwen3 | B3 boosting | 100 trees, lr 0.03 | 0.520 | 0.356 | 0.081 | 0.922 |
| GPT OSS | B0 | C=0.1 | 0.212 | 0.590 | 0.055 | 0.768 |
| GPT OSS | B1 | C=1 | 0.872 | 0.115 | 0.021 | 0.986 |
| GPT OSS | B2 | C=1 | 0.878 | 0.108 | 0.019 | 0.989 |
| GPT OSS | B3 | 200 trees, lr 0.03 | 0.879 | 0.107 | 0.016 | 0.989 |

Already, before any multicalibration, B2 beats every Campos method from Phase 7 (best there: IGLB 0.465 / 0.827). The features do the work.

Ablation (B2 minus one block, C re-picked, validation):

| dropped | Qwen3 log loss | Qwen3 AUROC | GPT OSS log loss | GPT OSS AUROC |
|---|---|---|---|---|
| none | 0.317 | 0.939 | 0.108 | 0.989 |
| logprob (and avg) | 0.326 | 0.933 | 0.104 | 0.991 |
| structure | 0.317 | 0.935 | 0.109 | 0.990 |
| size | 0.318 | 0.939 | 0.110 | 0.988 |
| self consistency | 0.329 | 0.933 | 0.115 | 0.986 |

Self consistency is the most useful block for both. Dropping token logprobs slightly helps GPT OSS: once truncation and code presence are known, token confidence adds noise there. Blocks overlap a lot (truncation shows up in size, empty code, and self consistency), so no single drop hurts much.

Boosting (B3) is worse than logistic for Qwen3 and badly calibrated (ECE 0.081): 316 problems is small for trees.

Tree auditor (depth 3, at least 400 rows per leaf, on B2 residuals in calib; `runs/phase9/tree_auditor.txt`). Every leaf has at least 63 problems.
* Qwen3: code with nesting <= 4 and at least one branch is under-predicted by 0.121 (107 problems); nesting > 4 over-predicted by 0.066 (85 problems). B2 misses a structure interaction a group could catch.
* GPT OSS: residuals within 0.045 everywhere.
* Caveat: leaf residuals are measured on the rows the tree was fit to, so they are adaptive and somewhat inflated (Phase 2 winner's curse).

Explain:
* The base model trains on base_train and the calibrators on calib so the calibrator sees the base model's honest errors. On its own training rows the base model looks better calibrated than it is, and a calibrator fit there would learn to fix nothing.
* Self consistency is allowed because it only compares code text across samples; it never looks at pass/fail. Leakage would be, for example, similarity to the passing samples, or the share of others that pass. The label flip test would catch that.
* A big coefficient doesn't mean an important feature: features are correlated (truncated, empty code, output tokens), so the model can split weight between them arbitrarily, and coefficient size also depends on scaling. The ablation measures importance directly.

### Phase 10: RQ2 experiments (validation only)

`src/calib/rq_setup.py`, `experiments/rq2_base_strength.py`, outputs in `runs/phase10/`. `PROTOCOL.md` started.

Decisions:
* Calibrators fit on calib (211 problems). IGLB needs an early stopping set; using official validation would make every validation number optimistic, so IGLB fits on `calib_fit` (148 problems) and stops on `calib_stop` (63). Validation stays clean for all methods.
* Platt input picked per starting score on validation (best of raw p, log p, logit p), so the baseline is as strong as it can be. A weak Platt would inflate multicalibration's gains.
* LINR, IGHB, IGLB use the code versions (no rounding; rounding hurt in Phase 7).
* IGHB alpha and IGLB epsilon picked once, globally, by mean validation BSS over both models, 4 starting scores, 2 group sets. Alpha: 0.05 gives 0.642, 0.01 gives 0.675, 0.003 gives 0.693 (chosen), 0.001 gives 0.678, 0.0003 gives 0.639. Epsilon 0.005, 0.01, 0.02 tie at 0.693 (the calib_stop rule stops IGLB first), so I kept the Campos default 0.01.
* Minimum group size stays at 40 train problems (Phase 6).
* Primary group set excludes difficulty (deployment realistic); secondary includes it.

Validation BSS, no difficulty groups:

| model | start | uncalibrated | Platt | HB | LINR | IGHB | IGLB |
|---|---|---|---|---|---|---|---|
| Qwen3 | avg_prob | -0.549 | 0.382 | 0.377 | 0.408 | 0.380 | 0.382 |
| Qwen3 | B1 | 0.563 | 0.572 | 0.557 | 0.558 | 0.556 | 0.563 |
| Qwen3 | B2 | 0.585 | 0.587 | 0.573 | 0.575 | 0.576 | 0.585 |
| Qwen3 | B3 | 0.520 | 0.542 | 0.533 | 0.533 | 0.530 | 0.520 |
| GPT OSS | avg_prob | -0.098 | 0.211 | 0.221 | 0.830 | 0.822 | 0.848 |
| GPT OSS | B1 | 0.872 | 0.879 | 0.874 | 0.872 | 0.870 | 0.878 |
| GPT OSS | B2 | 0.878 | 0.883 | 0.879 | 0.878 | 0.874 | 0.882 |
| GPT OSS | B3 | 0.879 | 0.882 | 0.877 | 0.879 | 0.877 | 0.879 |

Multicalibration minus Platt, validation BSS, 95% task clustered bootstrap (no difficulty groups; full list with difficulty in `rq2_validation_diffs.csv`):

| model | start | LINR | IGHB | IGLB |
|---|---|---|---|---|
| Qwen3 | avg_prob | +0.022 [-0.018, 0.061] | -0.002 [-0.037, 0.032] | +0.001 [-0.032, 0.032] |
| Qwen3 | B2 | -0.014 [-0.033, 0.004] | -0.011 [-0.037, 0.015] | -0.002 [-0.012, 0.007] |
| GPT OSS | avg_prob | +0.619 [0.556, 0.689] | +0.611 [0.548, 0.682] | +0.637 [0.572, 0.708] |
| GPT OSS | B2 | -0.005 [-0.010, -0.001] | -0.009 [-0.018, -0.002] | -0.000 [-0.007, 0.005] |

With difficulty groups, Qwen3 avg_prob: LINR +0.079 [0.028, 0.129], IGHB +0.051 [-0.002, 0.101]. Qwen3 B2: LINR +0.013 [-0.019, 0.044], IGLB +0.014 [-0.023, 0.053].

What it says about the hypothesis:
* Supported. From raw token probability, multicalibration beats Platt by 0.6 BSS on GPT OSS. From any feature based start (B1, B2, B3), every multicalibration method is within about 0.02 of Platt, and on GPT OSS LINR and IGHB are slightly but clearly worse.
* Qwen3 from avg_prob only gains when difficulty is a group. The deployment realistic groups add little for Qwen3 because its failures aren't mostly truncation.
* Difficulty groups still help Qwen3 a bit on top of B1/B2, because the base models never see difficulty. That's new information, not better calibration.
* Max gASCE (worst group calibration) doesn't improve over Platt for strong starts either. For Qwen3 B2: Platt 0.030, LINR 0.052, IGHB 0.062, IGLB 0.035. With 211 calib problems, group corrections add noise.
* IGLB often makes zero patches on strong starts (equal to uncalibrated): the calib_stop rule sees no gain. That's the safe failure mode.
* AUROC: for strong starts, multicalibration slightly lowers ranking quality (GPT OSS B2: 0.989 Platt vs 0.983 LINR).
* Matches Hansen et al. 2024: multicalibration post processing helps most when the base model is weak or uncalibrated on the groups.

Explain:
* The table says multicalibration's big wins in Campos come from starting at a score that knows nothing about the groups. A logistic model trained with log loss on features that define the groups is already roughly calibrated on them (its gradient conditions force average residual 0 along each feature), so there's little left to fix.
* Settings were chosen on validation; test is used once in Phase 13. Picking settings by test score would make test an optimistic estimate, the same overfitting problem as the winner's curse, one level up.

### Phase 11: routing (RQ3, validation)

`src/calib/routing.py`, `experiments/rq3_routing.py`, outputs in `runs/phase11/` (`routing.csv`, `routing_bootstrap.csv`, `routing_per_group.csv`, `platt_rank_sanity.csv`, `tradeoff_curves.png`). Settings in `PROTOCOL.md`.

Primary model: Qwen3. Both models have about 3B active parameters, so "cheap vs expensive" is a scenario, but Qwen3 writes far fewer tokens (median 874 vs 1468) and truncates less, so it's the natural first call. Swapped as a robustness check.

Validation pass rates: Qwen3 0.448, GPT OSS 0.495, oracle 0.602. Qwen3 fails and GPT OSS passes on 15.4% of pairs; the reverse on 10.6%.

Sanity check: Platt gives exactly the same P4 decisions as uncalibrated, 0 differences in all 16 cases (2 primaries x 2 starts x 4 rates). Cutoffs use `np.quantile(method="lower")` so the cutoff is an actual data value and any order preserving map keeps the same rows below it.

Key results, IGLB minus Platt with 95% task clustered bootstrap:

| primary | start | P4 20% pass | P6 pass |
|---|---|---|---|
| Qwen3 | avg_prob | +0.005 [0.000, 0.011] | +0.081 [0.058, 0.106] |
| Qwen3 | B2 | 0.000 | +0.002 [-0.002, 0.006] |
| GPT OSS | avg_prob | +0.037 [0.019, 0.055] | +0.081 [0.058, 0.106] |
| GPT OSS | B2 | 0.000 | +0.002 [-0.002, 0.006] |

(P6 is symmetric in the two models, so it's the same number for both primaries.)

P6 pass rates: from avg_prob uncalibrated 0.459, Platt 0.502, IGLB 0.583; from B2 0.589 to 0.591 for all three. The oracle is 0.602, so P6 with B2 or IGLB gets within 0.02 of it, at double the compute.

P5, cost threshold with overall q_B (total cost, lower is better, c_B = 2): Qwen3 primary from avg_prob: uncalibrated 6.52 (never escalates, raw p is too high), Platt 7.01, IGLB 7.10. Always primary is 6.52. Better calibrated p_A made P5 worse here. Reason: q_B is the fallback's overall pass rate, but the two models fail on the same hard problems, so where Qwen3 looks worst, GPT OSS also does worse than its average. The threshold escalates exactly the rows where escalation helps least. P5c (q_B as a function of p_A, fit on calib pairs) fixes this for Platt but not for IGLB: IGLB's in-sample p_A on calib is sharper than its out-of-sample p_A, so P5c learns too optimistic a q_B. GPT OSS primary from avg_prob is the one place the cost policies clearly prefer IGLB: P5c at c_B = 1, IGLB minus Platt total cost -0.40 [-0.64, -0.18].

From B2, P5 and P5c costs differ by at most 0.05 between calibrators, except Qwen3 P5 c_B = 5, where Platt escalates 13.6% vs 2.7% and costs +0.48 more.

What it says about the hypothesis:
* Rank based (P4): Platt can't change decisions (verified). Multicalibration can, by reordering rows across groups: from avg_prob with GPT OSS primary, +0.037 pass at 20% escalation. From B2 the rank barely moves and decisions are identical.
* Comparing two models (P6): the biggest effect. Raw scores from two models aren't on the same scale; Platt puts each on an honest marginal scale (+0.04); IGLB adds group information (truncation, length) that tells you which model's answer is actually usable (+0.08 more). From B2 there's nothing left to gain.
* Fixed thresholds (P5): calibration of p_A is necessary but not enough. The threshold also needs B's chance on this problem, which is correlated with A's. Using the overall q_B made better calibration look worse.
* Same pattern as RQ2: multicalibration's value comes from fixing a weak starting score. Starting from B2 it changes nothing that matters for decisions.

Explain:
* Calibration vs ranking: ranking is which answer is more likely to pass; calibration is whether "0.7" means 70%. A rank based policy uses only ranking. Any calibrator that keeps order (Platt) can't change it.
* Policies that care about scale: fixed thresholds (P5) and comparisons across two models (P6). There the number itself matters, not just the order.
* The oracle uses labels, so it's not a policy. It's the ceiling: the best pass rate any policy could get at a given escalation rate. Regret is the distance to it.
* Costs are assumptions. Changing c_B or L moves the thresholds and can flip which calibrator wins, so every cost result is labeled with its scenario.

### Phase 12: RQ4, overfitting study (validation only, exploratory)

Literature search first ("multicalibration overfitting", "reusable holdout", "thresholdout calibration", Hansen et al., Globus-Harris et al.):
* Overfitting of iterative multicalibration on small cells is known. Detommaso et al. 2024 motivate IGLB's larger overlapping cells with it; Globus-Harris et al. 2023 frame multicalibration as boosting, which overfits the same way; MCGrad (Meta, web scale multicalibration) relies on validation early stopping.
* HJKRR §3.3 already handle adaptivity with a guess and check oracle and differential privacy. Dwork et al. 2015 (Thresholdout) reuse a holdout by adding noise to the check. Feldman and Steinke 2018 scale the noise to the query's variance, which is close to my noise aware rule.
* Hansen et al. 2024: models that are calibrated out of the box tend to be multicalibrated already; post processing mostly helps uncalibrated models. That matches RQ2.
* So neither guarded variant is a new idea. What's here is an empirical check on code LLM calibration data. I won't call anything "first".

Setup (`src/calib/calibrators/ighb_protected.py`, `experiments/rq4_overfitting.py`, outputs `runs/phase12/`): subsample 25, 50, 100, 200 calib problems (20 random repeats each) or all 211 (once). Starts avg_prob and B2, no difficulty groups. Measure validation Brier minus the starting score's validation Brier (negative = helped).
* IGHB at the chosen alpha 0.003, and IGHB long (alpha 1e-4, cap 300 steps)
* noise aware IGHB (alpha 1e-4): only update a cell whose gap exceeds 2 standard errors, computed by problem (cluster SE), and only if the cell spans at least 5 problems
* holdout IGHB (alpha 1e-4): fit on 70% of the subsample's problems; accept an update only if holdout Brier gain + Laplace(0, 2e-4) noise > 0; a rejected cell is skipped until the next accepted update
* IGLB (fit 70%, early stop 30%) and Platt for reference

Bug found and fixed during the phase: the first noise aware run took more steps at n=25 than at n=200 for Qwen3 B2. With only 2 or 3 problems in a cell, the cluster SE is unreliable and can be tiny, so noise passed the 2 SE test. Added the 5 problem minimum. This is a setting chosen after looking at validation, fine for an exploratory RQ, and logged here.

Mean validation Brier change, start B2 (nothing much left to fix, so any change is mostly fitted noise):

| method | Qwen3 n=25 | 50 | 100 | 200 | GPT OSS n=25 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| Platt | +0.006 | +0.002 | +0.001 | -0.001 | +0.001 | -0.001 | -0.001 | -0.001 |
| IGLB | +0.005 | +0.009 | +0.001 | +0.000 | +0.002 | +0.001 | +0.001 | -0.000 |
| IGHB alpha 0.003 | +0.059 | +0.034 | +0.017 | +0.002 | +0.014 | +0.007 | +0.002 | +0.001 |
| IGHB long | +0.083 | +0.077 | +0.056 | +0.031 | +0.027 | +0.023 | +0.012 | +0.007 |
| holdout IGHB | +0.054 | +0.038 | +0.019 | +0.011 | +0.017 | +0.016 | +0.009 | +0.004 |
| noise aware IGHB | +0.005 | +0.012 | +0.011 | +0.003 | +0.000 | +0.002 | +0.001 | +0.001 |

Start avg_prob (lots to fix), Qwen3 / GPT OSS at n=25: IGHB -0.160 / -0.203, IGHB long -0.119 / -0.182, noise aware -0.152 / -0.190, holdout -0.144 / -0.187, IGLB -0.168 / -0.215, Platt -0.220 / -0.069.

What it shows:
* IGHB overfits when it runs long on little data, and the damage shrinks as data grows, like Phase 2. From B2 with 25 problems, long IGHB makes Qwen3 0.083 Brier worse, about 0.4 of B2's whole skill margin.
* The noise aware rule removes most of it: at n=25 it's as harmless as Platt and IGLB. It costs a little when there is real signal (avg_prob start: -0.152 vs -0.160 for plain IGHB).
* The holdout check helps less. It gives up 30% of an already small sample, and a single noisy Brier check on a small holdout accepts many bad updates.
* IGLB was already safe: early stopping on held out data plus big cells means it rarely patches when there's nothing to fix.
* Noise aware IGHB from avg_prob on GPT OSS hits the 300 step cap at n=200 and 211: overlapping groups keep producing significant, opposite corrections (the "bouncing" failure mode). The cap matters.

Explain:
* Connection to HJKRR §3.3: the loop chooses its next question based on answers it already got from the same sample, so naive reuse overfits. HJKRR fix it by answering through a noisy guess and check oracle (differential privacy). The holdout variant is a direct, crude version of that idea; the noise aware rule is the statistical testing version: only act on gaps too big to be noise.
* The plot (`brier_change_vs_size.png`): from a strong start, every IGHB curve starts above zero (harm) at small n and falls toward zero as n grows; noise aware IGHB, IGLB, and Platt stay near zero throughout.

### Phase 13: freeze and final test run

Before the freeze:
* Refactored the RQ2, RQ3, RQ4, and RQ1 scripts into functions so `experiments/final.py` runs the exact validation code on test. Reran the validation versions after the refactor: identical outputs, with one exception.
* The exception, a consistency fix: the RQ2 bootstrap used unclipped LINR predictions (the code version can go outside [0, 1]) while the table clipped them. Both now clip. Only LINR rows moved, by +0.004 to +0.009 (Qwen3 avg_prob with difficulty: +0.079 became +0.088). Done before any test run.
* `final.py --dry-run` ran the full pipeline with validation in place of test and reproduced the Phase 10 and 11 numbers. It caught one bug: `b3.compare` is a pandas method, so the RQ3 summary filter matched nothing. Fixed with `b3["compare"]`.
* `PROTOCOL.md` finished (split hashes, seeds, metrics, RQ4, run procedure), committed, tagged `v-final`.

Final run: `python -m experiments.final` on commit `8f1de29`, once. Output in `runs/8f1de29/`. The refit base models reproduced the Phase 9 validation predictions exactly (asserted). No bugs found after the run.

**RQ1** (train medians, test, 95% interval): every method within 0.004 BSS of Campos Table 1. Qwen3 IGLB 0.478 [0.394, 0.557], paper 0.480; GPT OSS IGLB 0.768 [0.707, 0.820], paper 0.764. The intervals are wide (about +-0.07), so the paper's differences between neighbors like LINR and IGLB, or Platt and HB, are within noise.

**RQ2 primary** (IGLB minus Platt, test BSS, no difficulty groups):

| model | start | diff | 95% interval |
|---|---|---|---|
| Qwen3 | avg_prob | +0.019 | [-0.016, 0.054] |
| Qwen3 | B2 | +0.001 | [-0.008, 0.012] |
| GPT OSS | avg_prob | +0.593 | [0.522, 0.668] |
| GPT OSS | B2 | -0.002 | [-0.008, 0.003] |

Hypothesis supported for GPT OSS: a huge gain from raw token probability, none from B2. For Qwen3 there's no clear IGLB gain even from avg_prob with deployment realistic groups; LINR does better there (+0.062 [0.025, 0.102]).

Test BSS, no difficulty groups:

| model | start | uncal | Platt | HB | LINR | IGHB | IGLB |
|---|---|---|---|---|---|---|---|
| Qwen3 | avg_prob | -0.543 | 0.390 | 0.382 | 0.452 | 0.401 | 0.409 |
| Qwen3 | B1 | 0.570 | 0.567 | 0.553 | 0.565 | 0.552 | 0.570 |
| Qwen3 | B2 | 0.560 | 0.559 | 0.540 | 0.560 | 0.527 | 0.560 |
| Qwen3 | B3 | 0.520 | 0.519 | 0.514 | 0.539 | 0.512 | 0.520 |
| GPT OSS | avg_prob | -0.075 | 0.197 | 0.213 | 0.773 | 0.762 | 0.790 |
| GPT OSS | B1 | 0.818 | 0.821 | 0.817 | 0.817 | 0.816 | 0.816 |
| GPT OSS | B2 | 0.820 | 0.823 | 0.816 | 0.819 | 0.816 | 0.821 |
| GPT OSS | B3 | 0.816 | 0.819 | 0.820 | 0.817 | 0.814 | 0.816 |

Secondary:
* With difficulty groups, Qwen3 B2 gains: LINR +0.041 [0.010, 0.077], IGLB +0.034 [-0.001, 0.078]. Difficulty is information the base model never saw, not better calibration.
* IGHB hurts strong starts: Qwen3 B2 -0.033 [-0.063, -0.005].
* Every start, every method beats raw scores; B1 and B2 beat every Campos method on avg_prob for Qwen3 (0.57 vs 0.45 for the best). For GPT OSS, IGLB on avg_prob (0.790) is close to B2 (0.820).
* Empty programs excluded: Qwen3 IGLB minus Platt +0.018 [-0.028, 0.066] from avg_prob, +0.002 [-0.011, 0.015] from B2. GPT OSS rows with code pass about 94% of the time, so BSS there is unstable (small reference Brier). No start reaches BSS above 0.1, and Platt fit on all rows collapses (-1.48) because its single curve is dominated by the empty rows. The GPT OSS avg_prob gain is about recognizing empty and truncated outputs.

**RQ3 primary** (P6 pass rate, IGLB minus Platt): avg_prob +0.067 [0.045, 0.089]; B2 +0.001 [-0.002, 0.005]. Supported. P6 pass: avg_prob uncalibrated 0.469, Platt 0.524, IGLB 0.591; B2 0.595 to 0.597; oracle 0.614.

Secondary: Platt vs uncalibrated P4 decisions identical in all 16 cases (test). P4 20% IGLB minus Platt: GPT OSS primary from avg_prob +0.022 [0.005, 0.041]; Qwen3 primary +0.001 [-0.008, 0.009]; from B2 exactly 0. P5 cost thresholds are mixed: with Qwen3 primary from avg_prob, IGLB costs more than Platt at c_B = 5 (+0.74 [0.53, 0.95]) and P5c c_B = 2 (+0.32); with GPT OSS primary, IGLB costs less at c_B = 1 (-0.38 [-0.54, -0.23]). Always primary is often the cheapest policy for Qwen3 under these assumed costs.

**RQ4** (test, B2 start, mean Brier change): at n=25, IGHB long +0.086 (Qwen3) / +0.028 (GPT OSS), plain IGHB +0.063 / +0.015, holdout IGHB +0.061 / +0.017, noise aware IGHB +0.007 / +0.001, IGLB +0.004 / +0.002, Platt +0.008 / +0.001. Same picture as validation.

Explain:
* The protocol is written before the test run so the test can only confirm or reject choices already made. If I picked settings after seeing test numbers, test would turn into a second validation set and the reported numbers would be optimistic.
* If asked whether I tuned on test: no. Every setting is in `PROTOCOL.md` at tag `v-final`, the final run is commit `8f1de29`, and the notebook logs every change made after any test look (RQ1 had zero bug fixes). The only earlier test use was the Phase 8 replication, under a bug-fix-only rule written before looking.

### Phase 14: analysis and writeup

* `PAPER.pdf` (11 pages) at the repo root, built by `experiments/make_pdf.py` from `REPORT.md` (kept local) with headless Chrome. README leads with the paper link and lists exact commands to rerun everything.
* Figures (`runs/8f1de29/figures/`, from `experiments/figures.py`): test reliability diagrams, group calibration scatter, IGHB stopping threshold sweep. Plus `runs/8f1de29/rq3/tradeoff_curves.png` and `runs/8f1de29/rq4/brier_change_vs_size.png` from the final run. The figure script recomputes test predictions with the frozen settings; its BSS values match `rq2.csv` exactly.
* `src/calib/rq_setup.py` gained `base_preds_with_test()`, the same test base model computation as `final.py`, cached to `data/base_preds_with_test.parquet`, so later scripts don't need to rerun the final.
* Checked before writing: in the Qwen3 B2 group scatter, the two off-diagonal groups are `comp_hard` (predicted 0.204, pass 0.062) and `uses_imports` (0.403 vs 0.288). IGLB made no patches on B2 because its 63 problem early stopping set showed no gain.
* Limitations listed in the paper: one benchmark; public data, no execution by me; test-based labels; two similar sized models; assumed costs; 10 samples at inference for self consistency; difficulty is metadata; empty programs; the 2000 token cap; small calib set; bootstrap ignores refit variation; hand made groups.

## 2026-10-10: v2

v2 adds boosted multicalibration with learned groups, an XGBoost base model with grouped Bayesian search, TreeSHAP interpretation with ablations, a transformer over token confidence trajectories, conformal risk control for routing, a Murphy Brier decomposition, and Holm corrected comparisons. **v2 was designed after seeing the v1 test results**, so its test numbers are a second look at the same test set, not a fresh one. Settings in `PROTOCOL_V2.md`.

### Phase V0: protocol and setup

Tagged `v1-final`. `PROTOCOL_V2.md` skeleton committed. Installed xgboost 3.2.0, optuna 5.0.0, shap 0.51.0, torch 2.14.1 (pinned in `requirements.txt`).

### Phase V0.5: data roles

Official validation split by `sha256("calib-routing-v2-2026" + id)` into `val_tune` and `val_conformal` (`runs/v2/splits_v2.json`). `base_train` and `calib` are the v1 split. `tests/test_splits_v2.py` checks no problem is in two roles and the roles cover all 1055 problems.

| role | problems | Qwen3 pass rate | GPT OSS pass rate |
|---|---|---|---|
| base_train | 316 | 0.450 | 0.533 |
| calib | 211 | 0.438 | 0.501 |
| val_tune | 132 | 0.467 | 0.487 |
| val_conformal | 132 | 0.428 | 0.503 |
| test | 264 | (not looked at) | (not looked at) |

v2 fits every calibrator on calib, including on raw avg_prob, so v2 avg_prob numbers differ from the Phase 8 replication (fit on full train). Both are reported.

### Phase V1: boosted multicalibration with learned groups

`src/calib/calibrators/boosted.py`, `src/calib/v2_eval.py` (fits every calibrator once per model and starting score), `experiments/v2_boosted.py`, outputs `runs/v2/v1_boosted/`. Each round: 10 equal width level sets by current p; inside each, a depth 2 regression tree on residuals over the B2 features, `min_samples_leaf=400` (about 40 problems); `p = clip(p + 0.5 * tree)`. Fit on calib, stop at the first round that doesn't lower val_tune Brier, cap 50. Tests: replaying the saved trees reproduces the fitted predictions exactly; on synthetic data it halves fresh max gASCE and finds the missing interaction feature.

Bug fixed during the phase: my first version gave no tree to level sets with fewer than 400 rows. Rows that moved into a small level set then never got corrected again; Qwen3 avg_prob ended at BSS 0.047 after 43 rounds. The plan has no such rule; a small level set's tree just can't split and becomes one constant shift. Removed it.

val_tune (calibrators on calib; v1 alpha 0.003, epsilon 0.01, Platt inputs; LOGR outputs probabilities):

| model | start | uncal BSS | Platt | IGLB | BMC | BMC rounds | max gASCE hand: before / BMC | max gASCE discovered: before / BMC |
|---|---|---|---|---|---|---|---|---|
| Qwen3 | avg_prob | -0.511 | 0.335 | 0.324 | 0.424 | 3 | 0.665 / 0.071 | 0.359 / 0.016 |
| Qwen3 | B2 | 0.545 | 0.550 | 0.545 | 0.549 | 1 | 0.051 / 0.052 | none found |
| GPT OSS | avg_prob | -0.107 | 0.224 | 0.869 | 0.534 | 6 | 0.452 / 0.099 | 0.343 / 0.082 |
| GPT OSS | B2 | 0.901 | 0.907 | 0.904 | 0.905 | 1 | 0.010 / 0.010 | 0.008 / 0.007 |

Discovered groups (leaves of the first 5 rounds with a feature condition, at least 40 calib problems; `runs/v2/v1_boosted/discovered_groups.json`):
* Qwen3 avg_prob, top confidence level: `sc_mean_sim <= 0.68` (leaf residual -0.60) vs above (-0.11). Low agreement with the other 9 samples marks the overconfident answers.
* GPT OSS avg_prob, levels 0.6 to 0.8: `all_std <= 0.45` (-0.58) and `sc_mean_sim <= 0.35` (-0.52). Low token log probability spread and low agreement mark failures.
* GPT OSS B2: `loops > 1.5` at the top level (-0.045), small.
* Qwen3 B2: none (one round, no split covering 40 problems).

Compared with the hand groups: none of the discovered groups are length or difficulty. They are self consistency and token spread, which the hand group set doesn't contain, but B2 does as features. So from B2 the trees find almost nothing, the same story as RQ2.

* From Qwen3 avg_prob, BMC beats every hand group method (0.424 vs 0.361 for LINR), because self consistency is the signal Qwen3 needs and no hand group carries it.
* From GPT OSS avg_prob, it trails IGLB (0.534 vs 0.869). It stops after 6 rounds at the first round that doesn't help val_tune, before fully catching truncation; IGLB gets truncation directly from the hand written `truncated` group.
* Caveat: BMC early stops on val_tune and is scored on val_tune here, so its numbers are optimistic. Discovered groups were found on calib, so measuring calibration on them in val_tune is fair.

Explain:
* Difference from IGHB: IGHB checks a fixed list of groups. Here a small tree picks the group itself, inside each confidence level, from all features.
* Shallow trees with a big minimum leaf: each leaf is a group with an estimated error. Deep trees or tiny leaves would chase noise (the winner's curse from Phase 2). 400 rows is about 40 problems, the same floor as the hand groups.
* Validation decides stopping because calib Brier always improves as trees fit calib's residuals. Only held out data can say when the trees start fitting noise.

### Phase V2: XGBoost with Bayesian hyperparameter search (B4)

`src/calib/xgb_model.py`, `experiments/v2_xgboost.py`, outputs `runs/v2/v2_xgboost/`. B2 features. Optuna TPE sampler (seed 0), 100 trials, objective mean log loss over `GroupKFold(5)` on base_train grouped by problem, no early stopping (`n_estimators` is searched instead). Search space as in the plan. Best config refit on all of base_train (XGBoost `random_state=0`). Trials in `optuna_trials_*.csv`.

Best configs: both models chose depth 2 and slow learning (Qwen3: 597 trees, lr 0.010, min_child_weight 2.8, subsample 0.80, colsample 0.68; GPT OSS: 478 trees, lr 0.013, min_child_weight 1.3, subsample 0.89, colsample 0.80). Grouped CV log loss 0.323 / 0.151.

Base predictors, uncalibrated:

| model | start | val_tune log loss | val_tune Brier | val_tune AUROC | full validation log loss |
|---|---|---|---|---|---|
| Qwen3 | B0 | 0.494 | 0.164 | 0.823 | 0.461 |
| Qwen3 | B1 | 0.349 | 0.115 | 0.929 | 0.329 |
| Qwen3 | B2 | 0.348 | 0.113 | 0.930 | 0.317 |
| Qwen3 | B3 | 0.394 | 0.134 | 0.906 | 0.356 |
| Qwen3 | B4 | 0.354 | 0.121 | 0.919 | 0.331 |
| GPT OSS | B0 | 0.587 | 0.193 | 0.781 | 0.590 |
| GPT OSS | B1 | 0.097 | 0.026 | 0.993 | 0.115 |
| GPT OSS | B2 | 0.090 | 0.025 | 0.995 | 0.108 |
| GPT OSS | B3 | 0.087 | 0.024 | 0.994 | 0.107 |
| GPT OSS | B4 | 0.089 | 0.023 | 0.995 | 0.106 |

XGBoost doesn't beat the logistic model for Qwen3 and ties it for GPT OSS. With 316 problems, the search settles on the simplest trees it's allowed (depth 2, slow learning), close to an additive model, which is what logistic regression already is.

Calibrators on B4 (val_tune BSS): Qwen3 uncalibrated 0.514, Platt 0.537, IGLB 0.542, BMC 0.529, LINR 0.523, IGHB 0.491. GPT OSS uncalibrated 0.906, Platt 0.909, BMC 0.909, IGLB 0.906 (no patch). Same story as B2: Platt does nearly all of it.

Explain:
* Boosting adds small trees one at a time, each fit to the remaining errors (gradient of log loss) of the trees before it.
* Grouped folds keep all 10 samples of a problem in one fold. With row folds, a fold's validation rows would have near-duplicate siblings in training and the search would reward memorizing problems.
* TPE models which hyperparameter values produced good versus bad trials, and samples new trials where good ones are more likely, instead of searching a grid.

### Phase V3: TreeSHAP and ablations on B4

`src/calib/shap_tools.py`, `experiments/v2_shap.py`, outputs `runs/v2/v3_shap/<model>/`. All SHAP explains B4's raw prediction before calibration (safeguard 8), and shows what the model relies on, not what causes code to fail. Main effects: interventional TreeSHAP, 200 base_train background rows (seed 0), probability scale, on val_tune (safeguards 2 to 4). Interactions: path dependent, log odds scale (shap's only option). B4 retrained under seeds 0, 1, 2.

Three bugs fixed during the phase, before the numbers below: a binary feature's "at or below the median" side was every row (`syntax_valid <= 1`); the worst slice list repeated the same rows under different group names; and 5 row slices with noisy gaps ranked as "worst". See DECISIONS.md 46 to 48 and 57.

Block importance (mean |block SHAP|, probability points, 95% task clustered interval, 1000 draws):

| block | Qwen3 | GPT OSS |
|---|---|---|
| self consistency | 0.149 [0.136, 0.163] | 0.163 [0.156, 0.170] |
| size | 0.087 [0.079, 0.094] | 0.165 [0.159, 0.171] |
| AST structure | 0.082 [0.075, 0.088] | 0.114 [0.109, 0.119] |
| logprob statistics | 0.051 [0.047, 0.055] | 0.019 [0.018, 0.020] |

* Seed stability: Spearman of feature importance between seeds 0.986 to 0.989 (Qwen3), 0.901 to 0.946 (GPT OSS).
* Noise check: the random column ranks 22 of 38 for both. Below it: Qwen3 `imports`, `code_std`, `empty_code`, `code_low`, `sc_missing`, `code_lp_missing`; GPT OSS `all_mean`, `all_low`, `sc_identical`, `code_low`, `truncated`, `code_lp_missing`. `truncated` ranking below noise for GPT OSS is the correlated credit problem (safeguard 1): output length, empty code, and others-empty carry the same information, so the model splits on those.
* Stable interactions (top 5 in all 3 seeds, val_tune): Qwen3 code lines x agreement, code lines x syntax valid, prompt length x output tokens; GPT OSS code chars x agreement, code lines x agreement, nesting x agreement. Agreement with the other samples (`sc_mean_sim`) is in almost every stable pair.

Block ablation (B4 config fixed; change in val_tune log loss when the block is dropped, positive = the block helped):

| block dropped | Qwen3 | GPT OSS | retuned (25 trials): Qwen3 / GPT OSS |
|---|---|---|---|
| logprob statistics | +0.010 | +0.004 | +0.010 / +0.005 |
| size | -0.024 | -0.001 | -0.015 / -0.001 |
| AST structure | +0.013 | +0.001 | +0.022 / -0.000 |
| self consistency | -0.005 | +0.003 | +0.004 / +0.006 |

SHAP and the ablation disagree in a useful way. SHAP ranks self consistency first, but dropping it barely matters, because size and AST features carry the same information (correlated credit). Dropping the size block makes Qwen3 B4 better on val_tune (log loss 0.330 vs 0.354): B4 overuses length features that don't hold up on new problems. Token log probability statistics get the least SHAP credit, but they're the hardest to replace, since nothing else carries them.

SHAP groups (frozen; chosen from out of fold SHAP on base_train, safeguard 7):
* Qwen3: `sc_mean_sim > 0.546`, `syntax_valid <= 0.5`, `log_output_tokens > 6.72`, `log_code_lines > 2.94`, and code lines with agreement.
* GPT OSS: `log_code_lines <= 2.77`, `sc_mean_sim > 0.373`, `syntax_valid <= 0.5`, `sc_others_empty <= 0.222`, code chars with agreement, code lines with agreement.

Three group sets, IGLB on B4 (fit calib, stop val_tune), measured on the union of all three sets:

| model | group set | groups | val_tune Brier | max gASCE union | slices > 2 SE (of checked) | IGLB patches |
|---|---|---|---|---|---|---|
| Qwen3 | uncalibrated | | 0.1209 | 0.072 | 28 of 91 | |
| Qwen3 | hand | 8 | 0.1140 | 0.062 | 23 of 90 | 1 |
| Qwen3 | discovered | 0 | | | | |
| Qwen3 | SHAP | 5 | 0.1137 | 0.049 | 22 of 90 | 1 |
| GPT OSS | uncalibrated | | 0.0234 | 0.012 | 38 of 69 | |
| GPT OSS | hand | 7 | 0.0234 | 0.012 | 38 of 69 | 0 |
| GPT OSS | discovered | 2 | 0.0229 | 0.007 | 29 of 59 | 1 |
| GPT OSS | SHAP | 6 | 0.0234 | 0.012 | 38 of 69 | 0 |

Answer to "are the features the model relies on most also where its calibration breaks?": partly. For Qwen3 the SHAP groups give the lowest worst-group error (0.049 vs 0.062 for hand groups), at essentially the same Brier. For GPT OSS only boosted multicalibration's discovered groups led IGLB to patch anything. Every set changes Brier by under 0.001.

Worst slices of uncalibrated B4 on val_tune, one sentence each:
* Qwen3, long code at p 0.55 (passes 0.83, 109 rows): B4 is underconfident on long, syntactically valid code whose samples agree (agreement 0.68 vs 0.46 overall); self consistency pushes it up 9 points, not enough.
* Qwen3, nested code at p 0.56 (passes 0.92, 47 rows): same pattern on deeply nested code; agreement and valid syntax push it up but it still lands 36 points low.
* Qwen3, long prompts at p 0.55 (passes 0.85, 54 rows): long prompts pull the prediction down (size -6 points) even when the samples agree strongly (0.76).
* Qwen3, long code with high agreement at p 0.64 (passes 0.92, 61 rows): agreement adds 12 points; B4 still underweights it.
* Qwen3, nested code at p 0.85 (passes 0.53, 40 rows): the one overconfident slice. Samples agree almost perfectly (0.86) and agreement adds 22 points, but they agree on the same wrong solution.
* GPT OSS, all five worst slices are underconfident at the top (for example p 0.86, passes 0.98, 100 rows): long valid code gets credit from size and structure, but B4 stays a few points short where nearly everything passes.

Per model: Qwen3 leans on agreement between samples; GPT OSS leans equally on size (did it finish and write code) and agreement. Token log probabilities matter least for both and barely at all for GPT OSS (0.019), which matches v1: GPT OSS token scores have AUROC near 0.5 once you know code exists.

Explain:
* A Shapley value is a feature's average contribution to the prediction over every order of adding features, measured against a background. Contributions add up exactly to the prediction minus the average prediction.
* TreeSHAP is exact for trees because a tree's prediction is a sum over leaves, so the average over feature orders can be computed by walking the tree instead of enumerating orders.
* Interventional mode breaks the link between correlated features when "removing" one, so it explains what the model computes. The default mode follows the training data's correlations and can credit a feature the model barely uses.
* Log odds vs probability: contributions add on the scale used. On log odds they're not percentage points; the probability scale ones are, so they can be said out loud.
* Correlated features split credit: if two features carry the same signal, SHAP divides it between them. The ablation measures what's unique to a block, so the two together tell you importance and redundancy.
* SHAP groups are chosen on out of fold base_train rows and evaluated on val_tune; choosing and evaluating on the same rows is the winner's curse again.
* SHAP is not causal: it says what B4 leans on. Long code passing more often doesn't mean making code longer helps.

### Phase V4: transformer over token confidence trajectories (B6, B7)

`src/calib/trajectory.py`, `experiments/v2_transformer.py`, `experiments/v2_transformer_eval.py`, outputs `runs/v2/v4_transformer/`. Each generation's per-token log probabilities become 256 chunk summaries (mean, min, share below -2, share in the code span) plus log token count, standardized on base_train. Model: linear 4 to 64, learned positions, 2 encoder layers, 4 heads, dropout 0.1, masked mean pool, scalar concatenated, 2 layer MLP head. BCE, AdamW lr 1e-3 wd 1e-2, batch 128, up to 50 epochs, early stopping patience 5 on an inner 80/20 base_train split by problem. 5 seeds averaged. B7 also concatenates the standardized B2 features before the head. Tests: chunking for long and short outputs; garbage in padded positions doesn't change the output. Seeds stopped after 12 to 17 epochs (best epoch 7 to 12). Trained on the Apple GPU after the CPU run was too slow next to V3.

val_tune, uncalibrated:

| model | start | log loss | Brier | AUROC |
|---|---|---|---|---|
| Qwen3 | B2 (logistic) | 0.348 | 0.113 | 0.930 |
| Qwen3 | B4 (XGBoost) | 0.354 | 0.121 | 0.919 |
| Qwen3 | B6 (trajectory) | 0.366 | 0.122 | 0.921 |
| Qwen3 | B7 (trajectory + features) | 0.369 | 0.126 | 0.922 |
| GPT OSS | B2 | 0.090 | 0.025 | 0.995 |
| GPT OSS | B4 | 0.089 | 0.023 | 0.995 |
| GPT OSS | B6 | 0.125 | 0.031 | 0.992 |
| GPT OSS | B7 | 0.088 | 0.024 | 0.994 |

Log loss differences (95% task clustered interval): Qwen3 B6 - B2 +0.018 [-0.023, 0.055], B7 - B2 +0.021 [0.000, 0.041]; GPT OSS B6 - B2 +0.035 [0.014, 0.060], B7 - B2 -0.002 [-0.012, 0.010].

**The trajectory did not beat the summary features.** Alone (B6) it's worse than logistic regression on the summary features for both models, clearly so for GPT OSS, whose signal is mostly "did it finish and write code", which the summary features state directly. Adding the summary features (B7) only brings it back to where B2 already was. With 316 training problems, a sequence model has little room to find shape information the summaries miss. Selected by val_tune Brier: B6 for Qwen3, B7 for GPT OSS (frozen in PROTOCOL_V2.md for the Holm family).

Calibrators on top (val_tune BSS): Qwen3 B6 uncalibrated 0.510, Platt 0.530, IGLB 0.536, BMC 0.530; GPT OSS B7 uncalibrated 0.906, Platt 0.905, BMC 0.907. GPT OSS B6 is the one place a group method clearly helps a learned start (IGHB 0.897 vs Platt 0.874), because B6 never sees whether code was written and the hand groups (`truncated`, `syntax_invalid`) supply it.

Explain:
* Chunking: outputs run up to 2000 tokens; 256 chunks keeps attention cheap (256 x 256) and gives every output the same length, while keeping the order of the trajectory.
* Attention lets each chunk's representation depend on every other chunk, so the model can relate, say, a confident ending to a hesitant middle. Pooling then summarizes the whole sequence.
* Backpropagation: the loss's gradient flows back through the head, pooling, attention, and projection, and AdamW nudges every weight against it.
* Several seeds: a small network on 316 problems lands in different places depending on its random start and batch order; averaging 5 reduces that variance.
