# Protocol, v2

v2 was designed after seeing the v1 test results (`runs/8f1de29/`). Everything below is fixed on `base_train`, `calib`, and `val_tune` before the single v2 test run. v1 settings in `PROTOCOL.md` stay as they are; anything v1 tuned on the full validation set is reused unchanged. New v2 tuning uses `val_tune` only.

## Data roles

| role | problems | used for |
|---|---|---|
| base_train | 316 (v1 split) | base predictors B2, B4, B6, B7; feature scaling; group thresholds; Optuna search |
| calib | 211 (v1 split) | every calibrator on top of every base predictor |
| val_tune | half of official validation by hash | early stopping (IGLB, boosted multicalibration), setting selection, SHAP diagnostics |
| val_conformal | other half | conformal risk control thresholds only |
| test | official test, 264 | one v2 run |

Every base predictor and calibrator is fit separately per model (Qwen3 Coder, GPT OSS). Only the optional shift test (V7) applies one model's fitted objects to the other.

## Methods and settings

* **Calibrators** (fit on calib per model and starting score): Platt (input per starting score: v1 validation choice for avg_prob and B2, val_tune choice for new scores, in `runs/v2/protocol_v2_frozen.json`), HB, LINR (code version), LOGR (probabilities), IGHB (code version, alpha 0.003), IGLB (code version, epsilon 0.01, early stop on val_tune), boosted multicalibration (below). Calibrator groups: extended hand set without difficulty, groups with fewer than 40 train problems dropped.
* **Boosted multicalibration (V1):** 10 equal width level sets; depth 2 regression tree on residuals over the B2 features per level set, `min_samples_leaf=400`, `random_state=0`; update `p = clip(p + 0.5 * tree)`; stop at the first round that doesn't lower val_tune Brier, cap 50. Discovered groups: leaf feature conditions from the first 5 rounds covering at least 40 calib problems.
* **B4 (V2):** XGBoost on the B2 features, `tree_method="hist"`, `random_state=0`. Optuna TPE (seed 0), 100 trials, mean log loss over `GroupKFold(5)` on base_train by problem, no early stopping. Best configs in `runs/v2/v2_xgboost/best_config.json` (both depth 2: Qwen3 597 trees lr 0.0101; GPT OSS 478 trees lr 0.0125).

## Primary comparisons and Holm family

(filled in before the test run)

## Conformal risk control

Targets: 0.05, 0.10, 0.15.

## Seeds

(filled in per phase)
