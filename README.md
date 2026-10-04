# multicalibrating-llms

Multicalibration and confidence gated routing for code LLMs, on CALIBRI (LiveCodeBench, Qwen3 Coder and GPT OSS).

Questions:

1. Do Platt, histogram binning, LINR, IGHB, and IGLB reproduce the Campos et al. (2025) LiveCodeBench results?
2. Do multicalibration gains over Platt shrink when the starting score is a feature rich learned model instead of raw token probability?
3. Does multicalibration improve routing between the two models, beyond Platt, at matched escalation budgets?
4. Optional: does IGHB overfit at small calibration set sizes, and does noise aware stopping help?

Status: setup. No results yet.

## Setup

Python 3.11.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
git clone https://github.com/violacampos/multicalibration reference
```

`reference/` is the Campos et al. replication repo, used for settings only. It is gitignored.

## Layout

- `notebook/LAB_NOTEBOOK.md`: dated log of decisions and numbers
- `src/calib/`: library code
- `experiments/`: one script per research question
- `tests/`: `pytest`

## Data

Hugging Face [`lavis-nlp/CALIBRI`](https://huggingface.co/datasets/lavis-nlp/CALIBRI), configs `livecodebench_qwen3` and `livecodebench_gpt-oss`. Splits are by problem: 527 train, 264 validation, 264 test. The test split is used only for the replication checkpoint and the final run.

## References

- Hébert-Johnson, Kim, Reingold, Rothblum (2018). Multicalibration: Calibration for the (Computationally-Identifiable) Masses.
- Campos, Kuschnereit, Ulges (2025). Multicalibration for LLM-based Code Generation. arXiv:2512.08810.
- Detommaso, Bertran, Fogliato, Roth (2024). Multicalibration for Confidence Scoring in LLMs.
- Hansen, Devic, Nakkiran, Sharan (2024). When is Multicalibration Post-Processing Necessary?
- Globus-Harris, Harrison, Kearns, Roth, Sorrell (2023). Multicalibration as Boosting for Regression.
