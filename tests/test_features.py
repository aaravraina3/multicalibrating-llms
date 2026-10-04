import numpy as np
import pandas as pd

from calib.features import build_features, code_structure
from calib.splits import split_train


def toy_rows():
    codes = ["x = 1\nprint(x)\n", "", "for i in range(3):\n    if i:\n        print(i)\n", "def f(:\n"] * 5
    return pd.DataFrame({
        "task_id": np.repeat(["a", "b"], 10), "model": "qwen3", "prompt": "solve it",
        "code": codes, "output": ["text"] * 20,
        "logprobs": [np.array([-0.1, -0.5, -3.0, 0.0], np.float32)] * 20,
        "code_start": [0, 0, 1, 0] * 5, "code_end": [2, 0, 4, 0] * 5,
        "y": np.tile([1, 0], 10),
    })


def test_features_ignore_labels():
    rows = toy_rows()
    flipped = rows.assign(y=1 - rows.y)
    pd.testing.assert_frame_equal(build_features(rows), build_features(flipped))


def test_structure_flags():
    assert code_structure("")["empty_code"]
    assert not code_structure("def f(:\n")["syntax_valid"]
    s = code_structure("for i in range(3):\n    if i:\n        print(i)\n")
    assert s["syntax_valid"] and s["nesting"] == 2 and s["loops"] == 1 and s["branches"] == 1


def test_split_is_deterministic_and_disjoint():
    ids = [f"t{i}" for i in range(100)]
    a, b = split_train(ids), split_train(list(reversed(ids)))
    assert a == b and not set(a["base_train"]) & set(a["calib"]) and len(a["base_train"]) == 60
