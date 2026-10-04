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
