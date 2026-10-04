# Protocol

Settings fixed on validation before the final test run. Not changed after the run.

## Data

* `lavis-nlp/CALIBRI` revision `7a4a7dc7`, configs `livecodebench_qwen3` and `livecodebench_gpt-oss`
* Official splits by problem: train 527, validation 264, test 264
* Official train split again by `sha256("calib-routing-2026" + problem_id)`: first 316 problems `base_train`, last 211 `calib`. Inside `calib`, the same hash order puts the first 70% (148 problems) in `calib_fit` and the rest (63) in `calib_stop`. Same assignment for both models. Lists in `runs/phase9/splits.json`.
* Empty programs included in main results. Robustness check with them excluded.

## RQ1, replication

Settings match the Campos code (`github.com/violacampos/multicalibration`, commit `c9b7e5d`):

* starting score `avg_prob`; 21 point grid (0, 0.05, ..., 1.0)
* Platt on raw p, `C=1`; HB nearest grid point; LINR linear regression on `[p, groups]`; LOGR hard labels; IGHB alpha 0.05, no rounding; IGLB epsilon 0.01, BFGS patch, no rounding
* replication groups: `loc_high`, `prompt_len_high`, `len_high`, `comp_easy`, `comp_medium`, `comp_hard`; medians from official train
* fit on official train, IGLB early stops on official validation, scored on test
* ECE and gASCE on grid rounded probabilities; accuracy `p > 0.5`; BSS base rate from the evaluated rows

## RQ2, starting score strength

* Starting scores: `avg_prob`, B1 (logistic, no self consistency), B2 (logistic, all features), B3 (gradient boosting, all features). Base models fit on `base_train`. Chosen hyperparameters in `runs/phase9/chosen_hyperparameters.json`: Qwen3 B1 C=1, B2 C=0.01, B3 100 trees lr 0.03; GPT OSS B1 C=1, B2 C=1, B3 200 trees lr 0.03. All B3: 8 leaves, `min_samples_leaf=50`, `early_stopping=False`.
* Calibrators fit on `calib` (`calib_fit` + `calib_stop`), except IGLB, which fits on `calib_fit` and early stops on `calib_stop`.
* Methods: uncalibrated, Platt, HB, LINR (code version), IGHB (code version, alpha 0.003), IGLB (code version, epsilon 0.01).
* Platt input per starting score, picked on validation: `avg_prob` log p (both models); B1, B2 logit p (both); B3 log p for Qwen3, logit p for GPT OSS.
* Groups: extended set, thresholds from official train, groups with fewer than 40 train problems dropped (drops `syntax_invalid` for GPT OSS). Primary set excludes difficulty; secondary includes it. Lists in `runs/phase10/chosen.json`.
* Primary comparison: BSS of IGLB minus Platt on test, no difficulty groups, with a 95% task clustered paired bootstrap interval (2000 resamples, seed 0), for `avg_prob` and for B2, per model. Hypothesis: the gain is large for `avg_prob` and near zero for B2.
* Everything else in the RQ2 tables is secondary and exploratory.

## RQ3, routing

* Pairs: problem t, sample k of the primary with sample k of the fallback. Primary Qwen3 (median 874 output tokens vs 1468 for GPT OSS, so cheaper to call first), fallback GPT OSS. Roles swapped as a robustness check.
* Calibrators: uncalibrated, Platt, IGLB (RQ2 settings, no difficulty groups), from `avg_prob` and from B2.
* Policies: P0 always primary; P1 always fallback; P2 oracle (labels, reference only); P3 random at rate r (in expectation); P4 escalate the lowest r of primary confidence, r in {10, 20, 30, 40}%, cutoff = validation quantile (`method="lower"`); P5 escalate when `p_A < q_B - c_B / L` with q_B the fallback's calib pass rate; P5c same with q_B fit as a logistic function of logit p_A on calib pairs; P6 run both, keep the higher calibrated probability (ties keep A).
* Assumed costs (scenarios, not measurements): c_A = 1, c_B in {1, 2, 5}, cost of a wrong shipped answer L = 10.
* Metrics: system pass rate, escalation rate, accepted error rate, compute cost `c_A + c_B * escalation` (P6: `c_A + c_B`), total cost = compute + L x failure rate, regret = oracle pass at the same escalation minus policy pass. Also per group.
* Primary comparison: P6 system pass rate, IGLB minus Platt, from `avg_prob` and from B2, 95% task clustered paired bootstrap. Hypothesis: positive from `avg_prob`, zero from B2.
* Secondary: P4 at 20% (IGLB minus Platt), P5 and P5c total cost per scenario. Platt minus uncalibrated on P4 must be exactly 0 (checked).

## RQ4, overfitting (exploratory)

* Subsample `calib` to 25, 50, 100, 200 problems (20 repeats, seed `1000 * size + repeat`) and all 211 (once). Inside each subsample, 70% of problems fit and 30% hold out for IGLB and holdout IGHB.
* Starts `avg_prob` and B2, no difficulty groups. Methods: Platt, IGHB alpha 0.003, IGHB alpha 1e-4 (cap 300), noise aware IGHB (alpha 1e-4, 2 cluster standard errors, at least 5 problems per cell, cap 300), holdout IGHB (alpha 1e-4, Laplace noise scale 2e-4, cap 300), IGLB.
* Metric: Brier on the evaluated split minus the starting score's Brier. No primary comparison; all RQ4 results are exploratory.

## Split hashes

sha256 of the comma joined sorted problem IDs, first 16 hex characters:

| set | problems | hash |
|---|---|---|
| base_train | 316 | 5b53a3c4e3016d2e |
| calib_fit | 148 | e21ce81985906ce7 |
| calib_stop | 63 | c1a1818c2eb308e2 |
| validation | 264 | 14c3d2acef64cefa |

## Seeds and metrics

* Bootstrap: 2000 resamples of problems with replacement, seed 0, 2.5 and 97.5 percentiles.
* Gradient boosting `random_state=0`; holdout IGHB Laplace noise seeded by repeat index.
* Metrics: Brier, BSS (base rate from the evaluated rows), log loss (clip 1e-6), ECE (20 equal width bins; grid rounded only for RQ1 tables), max gASCE over groups, accuracy (`p > 0.5`), AUROC.
* LINR and other outputs clipped to [0, 1] before every metric and bootstrap.

## Final run

* `python -m experiments.final` on the commit tagged `v-final`. Writes `runs/<git hash>/`. Run once. A bug found afterwards gets fixed, logged in the notebook, rerun, and reported.
