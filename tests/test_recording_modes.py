"""Mode-contract tests for BaseData and NeuroData."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vneurotk as vnt
from vneurotk.core import BaseData
from vneurotk.neuro.base import NeuroData


@pytest.mark.parametrize("mode", ["continuous", "epochs", "patterns"])
def test_construct_access_save_read_all_modes(mode, tmp_path) -> None:
    if mode == "continuous":
        neuro = np.arange(80, dtype=float).reshape(40, 2)
        data = BaseData(neuro, {"sfreq": 10.0}, data_mode=mode)
        data.configure(stim_ids=[10, 20], trial_window=[-2, 4], vision_onsets=np.array([10, 25]))
    elif mode == "epochs":
        neuro = np.arange(48, dtype=float).reshape(3, 8, 2)
        data = BaseData(neuro, {"sfreq": 10.0}, data_mode=mode)
        data.configure(stim_ids=[10, 20, 10], vision_onsets=np.array([2, 2, 2]))
    else:
        neuro = np.arange(12, dtype=float).reshape(6, 2)
        data = BaseData.for_patterns(neuro, {"ch_names": ["a", "b"]})

    assert data.is_configured
    np.testing.assert_array_equal(data.neuro.data, neuro)

    path = tmp_path / f"{mode}.h5"
    data.save(path)
    loaded = vnt.read(path)
    assert loaded.data_mode == mode
    assert loaded.is_configured
    np.testing.assert_array_equal(loaded.neuro.data, neuro)


def test_invalid_data_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid data_mode"):
        BaseData(np.zeros((2, 2)), {}, data_mode="samples")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode, shape", [("continuous", (20, 2)), ("epochs", (3, 5, 2))])
def test_incomplete_trial_state_is_not_configured(mode, shape) -> None:
    data = BaseData(
        np.zeros(shape),
        {"sfreq": 10.0},
        data_mode=mode,
        stim_labels=np.zeros(shape[:-1]),
        trial=np.zeros(shape[:-1]),
    )

    assert data.is_configured is False
    with pytest.raises(RuntimeError, match="incomplete"):
        data.save("unused.h5")


def test_patterns_without_ids_are_valid_but_not_vision_alignable(tmp_path) -> None:
    data = BaseData.for_patterns(np.ones((4, 3)), {"ch_names": ["a", "b", "c"]})

    assert data.is_configured
    assert data.n_trials == 0
    with pytest.raises(RuntimeError, match="explicit stimulus IDs"):
        _ = data.trial_stim_ids
    with pytest.raises(RuntimeError, match="explicit row stimulus IDs"):
        _ = data.vision

    path = tmp_path / "patterns-no-ids.h5"
    data.save(path)
    loaded = vnt.read(path)
    assert loaded.is_configured
    assert loaded.trial_meta is None
    np.testing.assert_array_equal(loaded.neuro.data, data.neuro.data)


def test_patterns_with_ids_use_row_order_for_vision_alignment(tmp_path) -> None:
    row_ids = np.array([2, 1, 2, 3])
    data = BaseData.for_patterns(
        np.ones((4, 3)),
        {"ch_names": ["a", "b", "c"]},
        trial_meta=pd.DataFrame({"stim_index": row_ids, "split": ["a", "a", "b", "b"]}),
        vision_info={"n_stim": 3, "stim_ids": [1, 2, 3]},
    )

    np.testing.assert_array_equal(data.trial_stim_ids, row_ids)
    np.testing.assert_array_equal(data.vision.output_order, row_ids)

    path = tmp_path / "patterns-with-ids.h5"
    data.save(path)
    loaded = vnt.read(path)
    np.testing.assert_array_equal(loaded.trial_stim_ids, row_ids)
    np.testing.assert_array_equal(loaded.vision.output_order, row_ids)


def test_pattern_metadata_length_must_match_rows() -> None:
    data = BaseData.for_patterns(
        np.zeros((4, 2)),
        {},
        trial_meta=pd.DataFrame({"stim_index": [1, 2, 3]}),
    )
    assert data.is_configured is False


def test_ragged_trial_epochs_raise_domain_error() -> None:
    neuro = NeuroData(
        np.zeros((20, 2)),
        trial_starts=np.array([0, 10]),
        trial_ends=np.array([5, 17]),
        data_mode="continuous",
    )

    with pytest.raises(ValueError, match=r"unequal lengths.*\[5, 7\]"):
        _ = neuro.epochs
