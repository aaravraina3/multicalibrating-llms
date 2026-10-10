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

* **TreeSHAP (V3):** B4 seeds 0, 1, 2; interventional, probability scale, 200 base_train background rows (seed 0); interactions path dependent on the log odds scale; diagnostics on val_tune. Frozen SHAP groups (from out of fold SHAP on base_train): Qwen3 `sc_mean_sim > 0.546`, `syntax_valid <= 0.5`, `log_output_tokens > 6.72`, `log_code_lines > 2.94`, `log_code_lines > 2.94 & sc_mean_sim > 0.546`; GPT OSS `log_code_lines <= 2.77`, `sc_mean_sim > 0.373`, `syntax_valid <= 0.5`, `sc_others_empty <= 0.222`, `log_code_chars > 5.85 & sc_mean_sim > 0.373`, `log_code_lines <= 2.77 & sc_mean_sim > 0.373` (`runs/v2/v3_shap/<model>/shap_groups.json`). Three way comparison: IGLB on B4 with hand, discovered, and SHAP groups; worst group error over their union; slices of at least 5 problems in 10 bins, flagged when |gap| > 2 standard errors by problem.
* **B6 / B7 (V4):** 256 chunks x 4 channels plus log token count; linear 4 to 64, learned positions, 2 encoder layers, 4 heads, feed forward 128, dropout 0.1, masked mean pool, 2 layer head. AdamW lr 1e-3, wd 1e-2, batch 128, up to 50 epochs, patience 5 on an inner 80/20 base_train split by problem; seeds 0 to 4 averaged. Weights frozen in `runs/v2/v4_transformer/models/`. Selected per model by val_tune Brier: **Qwen3 B6, GPT OSS B7.**
* **Conformal routing (V5):** Qwen3 primary, GPT OSS fallback, paired by problem and sample. Base predictor **B2** (best Qwen3 val_tune Brier after calibration). Scores: uncalibrated, Platt, IGLB, boosted multicalibration. Loss: accepted (p >= t) and fails, averaged per problem; smallest t with `(n / (n + 1)) * mean_loss(t) + 1 / (n + 1) <= target` on val_conformal. Reported per target: realized risk, escalation, system pass, oracle pass at the same escalation, regret, and rank based P4 on the raw B2 score at the same escalation. Routing explanation: B4 block SHAP of escalated vs accepted val_conformal answers under Platt at target 0.10.
* **Murphy decomposition (V6):** 20 equal width bins; changes in reliability and resolution vs Platt with 2000 resample task clustered intervals.
* **Shift (V7):** B2 and B4 fit on one model and evaluated on the other; Platt (logit p), IGLB, boosted multicalibration; conformal threshold from the source model's val_conformal.
* All frozen choices in machine readable form: `runs/v2/protocol_v2_frozen.json`.

## Primary comparisons and Holm family

28 comparisons, all on test Brier, two sided task clustered bootstrap p values (2000 resamples, seed 0), Holm adjusted together, significant if adjusted p < 0.05. Every one is reported, adjusted and unadjusted.

* For each model (Qwen3, GPT OSS), each starting score (avg_prob, B2, B4, and the selected transformer: B6 for Qwen3, B7 for GPT OSS), and each group aware calibrator (LINR, IGLB, boosted multicalibration): calibrator minus Platt. 2 x 4 x 3 = 24.
* Per model: B4 minus B2 and the selected transformer minus B4, uncalibrated. 4 more.

## Conformal risk control

Targets: 0.05, 0.10, 0.15. Thresholds from val_conformal only. Simulation check: 500 fresh calibration draws of 132 problems with known pass probabilities (seed 1).

## Seeds

Optuna TPE 0; XGBoost 0 (SHAP retrains 0, 1, 2; noise column seed 123); trees in boosted multicalibration 0; transformer seeds 0 to 4; bootstrap 0; SHAP background sample 0; validation split salt `calib-routing-v2-2026`; inner transformer split salt `calib-routing-v2-inner`.

## Test run

On the commit tagged `v2-final`, once:

```bash
.venv/bin/python -m experiments.v2_predict_traj --with-test
.venv/bin/python -m experiments.v2_run --role test
.venv/bin/python -m experiments.v2_summary runs/<git hash> test
```

A bug found afterwards gets fixed, logged in the notebook, rerun, and stated in the report.
