# multicalibrating-llms

**Read my paper on this: [When Does Multicalibration Help Code LLMs? (PDF)](PAPER.pdf)**

Reimplements the calibration and multicalibration methods from Campos et al. (2025) on CALIBRI (LiveCodeBench, Qwen3 Coder and GPT OSS), replicates their results, then tests whether multicalibration still helps when the starting score is a feature model, and whether it changes routing decisions between the two models.

Results on the test split, settings frozen in [PROTOCOL.md](PROTOCOL.md) before the single final run:

* Replication: every method within 0.004 Brier skill score of Campos Table 1. The paper vs released code gap comes from how group medians are computed. IGHB's weak numbers come from a loose stopping rule; in their code it underfits rather than overfits.
* Starting from raw token probability, IGLB beats Platt by +0.593 BSS on GPT OSS [0.522, 0.668]. Starting from a logistic model on label free features: -0.002 [-0.008, 0.003].
* Routing by keeping the more confident of two answers: IGLB beats Platt by +6.7 pass rate points from token probability [4.5, 8.9], +0.1 from the feature model [-0.2, 0.5].
* With 25 calibration problems IGHB fits noise; only updating cells whose gap exceeds two standard errors (by problem) removes most of the damage.

v2, designed after the v1 test run and frozen in [PROTOCOL_V2.md](PROTOCOL_V2.md) before its own test run:

* XGBoost tuned with Optuna over problem grouped folds, and a transformer over token confidence trajectories, don't clearly beat the logistic model (Qwen3 test BSS 0.560 logistic, 0.538 XGBoost, 0.501 transformer; GPT OSS within 0.005).
* Of 28 preregistered comparisons, the 4 that survive Holm's correction all start from raw token probability.
* A conformal Qwen3 to GPT OSS cascade targeting a rate of accepted failures held on validation but missed on test (0.141 at a 0.10 target; see the [diagnosis](runs/0bd9043/conformal_diagnosis.md)). Order preserving recalibration never changes its decisions.
* Full tables: [runs/0bd9043/v2_summary.md](runs/0bd9043/v2_summary.md).

The lab notebook ([notebook/LAB_NOTEBOOK.md](notebook/LAB_NOTEBOOK.md)) logs every decision, bug, and number by phase.

## Reproduce

Python 3.11.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest
```

Then, in order (each step writes to `runs/`):

```bash
.venv/bin/python -m experiments.check_data                  # downloads CALIBRI, data checks
.venv/bin/python -m experiments.synthetic                   # synthetic sandbox
.venv/bin/python -m experiments.scores_validation           # raw starting scores
.venv/bin/python -m experiments.group_agnostic_validation   # Platt, HB, isotonic
.venv/bin/python -m experiments.groups_table                # group sizes and thresholds
.venv/bin/python -m experiments.multicalibration_validation # LINR, LOGR, IGHB, IGLB
.venv/bin/python -m experiments.rq1_replication             # RQ1 on test
.venv/bin/python -m experiments.base_models                 # features and base models
.venv/bin/python -m experiments.rq2_base_strength           # RQ2 on validation, picks settings
.venv/bin/python -m experiments.rq3_routing                 # RQ3 on validation
.venv/bin/python -m experiments.rq4_overfitting             # RQ4 on validation
.venv/bin/python -m experiments.final                       # every RQ on test, once
.venv/bin/python -m experiments.figures                     # paper figures
```

v2, after the above:

```bash
.venv/bin/python -m experiments.v2_splits                   # val_tune / val_conformal
.venv/bin/python -m experiments.v2_boosted                  # boosted multicalibration
.venv/bin/python -m experiments.v2_xgboost                  # B4, Optuna search
.venv/bin/python -m experiments.v2_shap                     # TreeSHAP, ablations, SHAP groups
.venv/bin/python -m experiments.v2_transformer              # B6, B7 (or use the saved weights)
.venv/bin/python -m experiments.v2_transformer_eval
.venv/bin/python -m experiments.v2_predict_traj --with-test # separate process: torch next to xgboost crashes on macOS
.venv/bin/python -m experiments.v2_run --role test          # conformal, Murphy, Holm, shift, once
.venv/bin/python -m experiments.v2_summary runs/<git hash> test
```

The final runs used for the paper are `runs/8f1de29/` (v1) and `runs/0bd9043/` (v2). Optional: `git clone https://github.com/violacampos/multicalibration reference` to compare settings with the Campos code (commit `c9b7e5d`).

## Layout

* `src/calib/`: data loading, scores, features, groups, metrics, bootstrap, calibrators, routing
* `experiments/`: one script per phase or research question
* `tests/`: `pytest`
* `runs/`: outputs by phase, final run by git hash

## Data

Hugging Face [`lavis-nlp/CALIBRI`](https://huggingface.co/datasets/lavis-nlp/CALIBRI), revision `7a4a7dc7`, configs `livecodebench_qwen3` and `livecodebench_gpt-oss`. Splits by problem: 527 train, 264 validation, 264 test.

## References

* Hébert-Johnson, Kim, Reingold, Rothblum (2018). Multicalibration: Calibration for the (Computationally-Identifiable) Masses.
* Detommaso, Bertran, Fogliato, Roth (2024). Multicalibration for Confidence Scoring in LLMs.
* Campos, Kuschnereit, Ulges (2025). Multicalibration for LLM-based Code Generation. arXiv:2512.08810.
* Hansen, Devic, Nakkiran, Sharan (2024). When is Multicalibration Post-Processing Necessary?
* Globus-Harris, Harrison, Kearns, Roth, Sorrell (2023). Multicalibration as Boosting for Regression.
* Dwork, Feldman, Hardt, Pitassi, Reingold, Roth (2015). Generalization in Adaptive Data Analysis and Holdout Reuse.
