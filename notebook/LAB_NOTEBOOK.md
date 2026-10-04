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
