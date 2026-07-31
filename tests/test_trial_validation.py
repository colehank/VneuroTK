"""Focused validation tests for trial-structure configuration."""

from __future__ import annotations

import numpy as np
import pytest

from vneurotk.core import BaseData
from vneurotk.neuro.trial import _validate_continuous_boundaries, _window_to_samples


def test_continuous_onsets_must_be_one_dimensional() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match="vision_onsets must be 1-D"):
        data.configure(
            stim_ids=np.array([1, 2]),
            trial_window=[-5, 10],
            vision_onsets=np.array([[20, 50]]),
        )


@pytest.mark.parametrize(
    ("stim_ids", "onsets", "message"),
    [
        (np.array([[1, 2]]), np.array([20, 50]), "visual_ids must be 1-D"),
        (np.array([1]), np.array([20, 50]), "same length"),
    ],
)
def test_continuous_ids_and_onsets_must_be_aligned(stim_ids, onsets, message) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=stim_ids, trial_window=[-5, 10], vision_onsets=onsets)


@pytest.mark.parametrize(
    ("trial_window", "message"),
    [
        ([0], "exactly two"),
        ([0, 1, 2], "exactly two"),
        ([1, 1], "start must be less than end"),
        ([2, 1], "start must be less than end"),
        ([False, 1], "finite real numbers"),
        ([0, np.inf], "finite real numbers"),
        ([0, "1"], "finite real numbers"),
    ],
)
def test_continuous_window_is_validated(trial_window, message) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=[1], trial_window=trial_window, vision_onsets=np.array([20]))


@pytest.mark.parametrize("sfreq", [0, -1, np.inf, np.nan, True, "100"])
def test_continuous_sfreq_must_be_positive_and_finite(sfreq) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": sfreq})

    with pytest.raises(ValueError, match="sfreq must be a finite number greater than 0"):
        data.configure(stim_ids=[1], trial_window=[-0.01, 0.02], vision_onsets=np.array([20]))


@pytest.mark.parametrize(
    "onsets",
    [
        np.array([20.5]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([True]),
        np.array(["20"]),
    ],
)
def test_continuous_onsets_must_be_integral_finite_non_bool(onsets) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match="vision_onsets must contain finite, non-boolean integers"):
        data.configure(stim_ids=[1], trial_window=[-5, 10], vision_onsets=onsets)


@pytest.mark.parametrize(
    ("neuro", "message"),
    [
        (np.zeros(100), "continuous neuro data must be 2-D"),
        (np.zeros((2, 10, 3)), "continuous neuro data must be 2-D"),
        (np.zeros((0, 2)), "positive time and channel dimensions"),
        (np.zeros((10, 0)), "positive time and channel dimensions"),
    ],
)
def test_continuous_neuro_dimensions_are_validated(neuro, message) -> None:
    data = BaseData(neuro, {"sfreq": 100.0}, data_mode="continuous")

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=[1], trial_window=[0, 1], vision_onsets=np.array([0]))


@pytest.mark.parametrize(
    ("onsets", "window", "message"),
    [
        (np.array([2]), [-5, 5], "fully within the recording"),
        (np.array([98]), [-5, 5], "fully within the recording"),
        (np.array([20, 25]), [-5, 10], "must not overlap"),
    ],
)
def test_continuous_trial_boundaries_are_validated(onsets, window, message) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=np.arange(len(onsets)), trial_window=window, vision_onsets=onsets)


def test_continuous_onsets_may_be_unsorted_when_trials_do_not_overlap() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    data.configure(stim_ids=[1, 2], trial_window=[-5, 5], vision_onsets=np.array([60, 20]))

    np.testing.assert_array_equal(data.vision_onsets, [60, 20])


@pytest.mark.parametrize("stim_ids", [[True, 2], [1, "1"]])
def test_configure_preserves_heterogeneous_stimulus_id_types(stim_ids) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    data.configure(stim_ids=stim_ids, trial_window=[-5, 5], vision_onsets=np.array([20, 60]))

    trial_ids = data.trial_stim_ids.tolist()
    assert data.trial_stim_ids.dtype == object
    assert data.vision_info is not None
    assert trial_ids == stim_ids
    assert [type(value) for value in trial_ids] == [type(value) for value in stim_ids]
    assert [type(value) for value in data.vision_info["stim_ids"]] == [type(value) for value in stim_ids]


@pytest.mark.parametrize("stim_ids", [[1, 2], ["a", "b"]])
def test_configure_keeps_homogeneous_stimulus_ids_in_native_arrays(stim_ids) -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    data.configure(stim_ids=stim_ids, trial_window=[-5, 5], vision_onsets=np.array([20, 60]))

    assert data.trial_stim_ids.dtype != object


@pytest.mark.parametrize(
    ("neuro", "message"),
    [
        (np.zeros((3, 10)), "epochs neuro data must be 3-D"),
        (np.zeros((3, 10, 2, 1)), "epochs neuro data must be 3-D"),
        (np.zeros((0, 10, 2)), "positive time and channel dimensions"),
        (np.zeros((3, 0, 2)), "positive time and channel dimensions"),
        (np.zeros((3, 10, 0)), "positive time and channel dimensions"),
    ],
)
def test_epochs_neuro_dimensions_are_validated(neuro, message) -> None:
    data = BaseData(neuro, {"sfreq": 100.0}, data_mode="epochs")

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=np.arange(neuro.shape[0]))


@pytest.mark.parametrize(
    ("stim_ids", "onsets", "message"),
    [
        (np.array([[1, 2, 3]]), np.array([0, 0, 0]), "visual_ids must be 1-D"),
        (np.array([1, 2, 3]), np.array([[0, 0, 0]]), "vision_onsets must be 1-D"),
        (np.array([1, 2]), np.array([0, 0, 0]), "epoch count"),
        (np.array([1, 2, 3]), np.array([0, 0]), "epoch count"),
    ],
)
def test_epochs_ids_onsets_and_epoch_count_are_consistent(stim_ids, onsets, message) -> None:
    data = BaseData(np.zeros((3, 10, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match=message):
        data.configure(stim_ids=stim_ids, vision_onsets=onsets)


@pytest.mark.parametrize(
    "onsets",
    [
        np.array([0.0, 1.0, 2.0]),
        np.array([0, np.nan, 2]),
        np.array([False, False, False]),
        np.array(["0", "1", "2"]),
    ],
)
def test_epochs_onsets_must_be_integral_finite_non_bool(onsets) -> None:
    data = BaseData(np.zeros((3, 10, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match="vision_onsets must contain finite, non-boolean integers"):
        data.configure(stim_ids=[1, 2, 3], vision_onsets=onsets)


@pytest.mark.parametrize("onsets", [np.array([-1, 0, 0]), np.array([0, 9, 10])])
def test_epochs_onsets_must_be_in_range(onsets) -> None:
    data = BaseData(np.zeros((3, 10, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match="within each epoch"):
        data.configure(stim_ids=[1, 2, 3], vision_onsets=onsets)


def test_epochs_uniform_onsets_keep_scalar_window_metadata() -> None:
    data = BaseData(np.zeros((3, 10, 2)), {"sfreq": 100.0})

    data.configure(stim_ids=[1, 2, 3], vision_onsets=np.array([4, 4, 4]))

    assert data.trial_info == {"baseline": [-4, 0], "trial_window": [-4, 6]}


def test_epochs_varying_onsets_have_accurate_per_trial_metadata() -> None:
    data = BaseData(np.zeros((3, 10, 2)), {"sfreq": 100.0})

    data.configure(stim_ids=[1, 2, 3], vision_onsets=np.array([2, 4, 7]))

    assert data.trial_info == {
        "baseline": [[-2, 0], [-4, 0], [-7, 0]],
        "trial_window": [[-2, 8], [-4, 6], [-7, 3]],
    }


def test_configure_failure_leaves_unconfigured_state_atomic() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})

    with pytest.raises(ValueError, match="must not overlap"):
        data.configure(stim_ids=[1, 2], trial_window=[-5, 10], vision_onsets=np.array([20, 25]))

    assert data.configured is False
    assert data.stim_labels is None
    assert data.trial is None
    assert data.trial_starts is None
    assert data.trial_ends is None
    assert data.vision_onsets is None
    assert data.vision_info is None
    assert data.trial_info is None


def _attach_test_features(data: BaseData) -> None:
    from vneurotk.vision.representation.visual_representations import (
        VisualRepresentation,
        VisualRepresentations,
    )

    data.vision.add(
        VisualRepresentations(
            [
                VisualRepresentation(
                    model="test-model",
                    module_name="features",
                    module_type="Linear",
                    stim_ids=[1, 2],
                    array=np.array([[10.0], [20.0]]),
                )
            ]
        )
    )


def test_reconfigure_updates_existing_vision_output_order() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})
    data.configure(stim_ids=[1, 2], trial_window=[-5, 5], vision_onsets=np.array([20, 60]))
    _attach_test_features(data)
    np.testing.assert_array_equal(data.vision["features"], [[10.0], [20.0]])

    data.configure(stim_ids=[2, 1], trial_window=[-5, 5], vision_onsets=np.array([20, 60]))

    np.testing.assert_array_equal(data.trial_stim_ids, [2, 1])
    np.testing.assert_array_equal(data.vision.output_order, [2, 1])
    np.testing.assert_array_equal(data.vision["features"], [[20.0], [10.0]])


def test_reconfigure_incompatible_with_existing_vision_is_atomic() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})
    data.configure(stim_ids=[1, 2], trial_window=[-5, 5], vision_onsets=np.array([20, 60]))
    _attach_test_features(data)
    assert data.trial_starts is not None
    assert data.stim_labels is not None
    before_order = data.vision.output_order.copy()
    before_starts = data.trial_starts.copy()
    before_labels = data.stim_labels.copy()

    with pytest.raises(ValueError, match="do not cover output_order"):
        data.configure(stim_ids=[1, 3], trial_window=[-5, 5], vision_onsets=np.array([25, 70]))

    np.testing.assert_array_equal(data.trial_stim_ids, [1, 2])
    np.testing.assert_array_equal(data.trial_starts, before_starts)
    np.testing.assert_array_equal(data.stim_labels, before_labels)
    np.testing.assert_array_equal(data.vision.output_order, before_order)
    np.testing.assert_array_equal(data.vision["features"], [[10.0], [20.0]])


def test_reconfigure_failure_preserves_previous_trial_state() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 100.0})
    data.configure(stim_ids=[1, 2], trial_window=[-5, 5], vision_onsets=np.array([20, 60]))
    assert data.stim_labels is not None
    assert data.trial is not None
    assert data.trial_starts is not None
    assert data.trial_ends is not None
    assert data.vision_onsets is not None
    assert data.vision_info is not None
    assert data.trial_info is not None
    before = {
        "stim_labels": data.stim_labels.copy(),
        "trial": data.trial.copy(),
        "trial_starts": data.trial_starts.copy(),
        "trial_ends": data.trial_ends.copy(),
        "vision_onsets": data.vision_onsets.copy(),
        "vision_info": data.vision_info.copy(),
        "trial_info": data.trial_info.copy(),
    }

    with pytest.raises(ValueError, match="must not overlap"):
        data.configure(stim_ids=[3, 4], trial_window=[-5, 10], vision_onsets=np.array([20, 25]))

    assert data.configured is True
    np.testing.assert_array_equal(data.stim_labels, before["stim_labels"])
    np.testing.assert_array_equal(data.trial, before["trial"])
    np.testing.assert_array_equal(data.trial_starts, before["trial_starts"])
    np.testing.assert_array_equal(data.trial_ends, before["trial_ends"])
    np.testing.assert_array_equal(data.vision_onsets, before["vision_onsets"])
    assert data.vision_info == before["vision_info"]
    assert data.trial_info == before["trial_info"]


def test_epoch_existing_onsets_are_validated_when_used_as_fallback() -> None:
    data = BaseData(
        np.zeros((3, 10, 2)),
        {"sfreq": 100.0},
        data_mode="epochs",
        vision_onsets=np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="epoch count"):
        data.configure(stim_ids=[1, 2, 3])


def test_trial_window_rounding_must_still_produce_positive_length() -> None:
    data = BaseData(np.zeros((100, 2)), {"sfreq": 10.0})

    with pytest.raises(ValueError, match="at least one sample"):
        data.configure(stim_ids=[1], trial_window=[0.01, 0.02], vision_onsets=np.array([20]))


@pytest.mark.parametrize("onset_dtype", [np.int8, np.uint8])
def test_continuous_onset_arithmetic_uses_int64(onset_dtype) -> None:
    data = BaseData(np.zeros((400, 2)), {"sfreq": 100.0})

    data.configure(
        stim_ids=[1],
        trial_window=[-20, 20],
        vision_onsets=np.array([120], dtype=onset_dtype),
    )

    assert data.vision_onsets is not None
    assert data.vision_onsets.dtype == np.int64
    np.testing.assert_array_equal(data.trial_starts, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(data.trial_ends, np.array([140], dtype=np.int64))


def test_continuous_boundaries_require_positive_trial_length() -> None:
    with pytest.raises(ValueError, match="trial end must be greater than start"):
        _validate_continuous_boundaries(
            np.array([20], dtype=np.int64),
            np.array([20], dtype=np.int64),
            np.array([20], dtype=np.int64),
            ntime=100,
        )


@pytest.mark.parametrize("value", [np.float32(1.5), np.float64(1.5)])
def test_numpy_float_window_values_are_seconds(value) -> None:
    assert _window_to_samples([value, 2], sfreq=10.0) == [15, 2]


@pytest.mark.parametrize("value", [np.int8(2), np.uint8(2), np.int64(2)])
def test_numpy_integral_window_values_are_samples(value) -> None:
    assert _window_to_samples([value, 3.0], sfreq=10.0) == [2, 30]
