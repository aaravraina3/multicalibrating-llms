import json
from pathlib import Path

import pytest

from calib.data import TEST_ROWS_PATH, load_test_ids
from calib.v2_setup import SPLITS_V1, SPLITS_V2, make_splits_v2


def test_make_splits_v2_is_deterministic_halves():
    ids = [f"p{i}" for i in range(264)]
    a, b = make_splits_v2(ids), make_splits_v2(list(reversed(ids)))
    assert a == b and len(a["val_tune"]) == 132 and not set(a["val_tune"]) & set(a["val_conformal"])


@pytest.mark.skipif(not SPLITS_V2.exists() or not TEST_ROWS_PATH.exists(), reason="needs data and splits")
def test_no_problem_in_two_v2_roles():
    v1, v2 = json.loads(SPLITS_V1.read_text()), json.loads(SPLITS_V2.read_text())
    test_ids = set(load_test_ids().task_id)
    sets = [set(v1["base_train"]), set(v1["calib"]), set(v2["val_tune"]), set(v2["val_conformal"]), test_ids]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] & sets[j]
    assert sum(len(s) for s in sets) == 527 + 264 + 264
