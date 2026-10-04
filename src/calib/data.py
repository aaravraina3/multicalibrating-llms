"""Load CALIBRI LiveCodeBench and explode it into one row per generation."""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ROWS_PATH = DATA_DIR / "rows.parquet"
TEST_ROWS_PATH = DATA_DIR / "rows_test.parquet"

DATASET_ID = "lavis-nlp/CALIBRI"
DATASET_REVISION = "7a4a7dc7"
MODELS = {"qwen3": "livecodebench_qwen3", "gpt-oss": "livecodebench_gpt-oss"}
SPLITS = ["train", "validation", "test"]


def load_raw(model, split):
    """One record per problem, exactly as stored on Hugging Face."""
    return load_dataset(DATASET_ID, MODELS[model], split=split,
                        cache_dir=str(DATA_DIR / "hf_cache"), revision=DATASET_REVISION)


def explode(raw, model, split):
    """One row per generation. Logprobs become float32 arrays; token strings are dropped."""
    rows = []
    for rec in raw:
        for k in range(len(rec["output"])):
            pairs = rec["token_logprobs"][k]
            logprobs = np.array([float(lp) for lp, _ in pairs], dtype=np.float32)
            start, end = rec["code_token_idx"][k]
            rows.append({
                "task_id": rec["id"],
                "sample_idx": k,
                "model": model,
                "split": split,
                "difficulty": rec["difficulty"],
                "name": rec["name"],
                "prompt": rec["prompt"],
                "code": rec["program"][k] or "",
                "output": rec["output"][k],
                "logprobs": logprobs,
                "code_start": int(start),
                "code_end": int(end),
                "y": int(rec["is_correct"][k]),
            })
    return pd.DataFrame(rows)


def build_rows():
    """Download, explode, and cache. Test rows go to a separate file."""
    DATA_DIR.mkdir(exist_ok=True)
    frames = {s: [] for s in SPLITS}
    for model in MODELS:
        for split in SPLITS:
            frames[split].append(explode(load_raw(model, split), model, split))
    dev = pd.concat(frames["train"] + frames["validation"], ignore_index=True)
    test = pd.concat(frames["test"], ignore_index=True)
    dev.to_parquet(ROWS_PATH, index=False)
    test.to_parquet(TEST_ROWS_PATH, index=False)
    return dev, test


def load_rows(include_test=False):
    """Train and validation rows. Test rows only when include_test=True (Phase 8 and 13)."""
    if not ROWS_PATH.exists():
        build_rows()
    rows = pd.read_parquet(ROWS_PATH)
    if include_test:
        rows = pd.concat([rows, pd.read_parquet(TEST_ROWS_PATH)], ignore_index=True)
    rows["logprobs"] = rows["logprobs"].map(lambda a: np.asarray(a, dtype=np.float32))
    return rows


def load_test_ids():
    """Problem IDs in the test split, without labels. For split checks only."""
    test = pd.read_parquet(TEST_ROWS_PATH, columns=["task_id", "model", "sample_idx"])
    return test
