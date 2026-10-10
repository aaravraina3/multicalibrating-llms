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

(filled in per phase)

## Primary comparisons and Holm family

(filled in before the test run)

## Conformal risk control

Targets: 0.05, 0.10, 0.15.

## Seeds

(filled in per phase)
