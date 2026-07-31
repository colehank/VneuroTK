"""Trial structure computation for neural recordings.

All knowledge of how to map stimulus onsets + trial windows onto sample indices
lives here.  Two public factory functions produce :class:`TrialStructure` value
objects from raw arrays — no :class:`BaseData` required.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from loguru import logger

from vneurotk.core.metadata import TrialInfo, VisionInfo
from vneurotk.core.stimulus import _coerce_scalar_array


def _make_nan_array(shape: tuple | int, dtype_kind: str) -> np.ndarray:
    """Return a NaN-filled array, using object storage for typed stimulus IDs."""
    if dtype_kind == "O":
        arr = np.empty(shape, dtype=object)
        arr[:] = np.nan
        return arr
    if dtype_kind in ("U", "S"):
        arr = np.empty(shape, dtype=object)
        arr[:] = np.nan
        return arr
    return np.full(shape, np.nan)


@dataclass
class TrialStructure:
    """Value object produced by the trial-structure factory functions.

    All fields are written atomically by :meth:`BaseData._apply_trial_structure`.
    """

    stim_labels: np.ndarray
    trial: np.ndarray
    trial_starts: np.ndarray
    trial_ends: np.ndarray
    vision_onsets: np.ndarray
    vision_info: VisionInfo
    trial_info: TrialInfo


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_1d(name: str, values: np.ndarray) -> None:
    """Require a one-dimensional array."""
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {values.shape}.")


def _validate_aligned(visual_ids: np.ndarray, vision_onsets: np.ndarray) -> None:
    """Require one stimulus ID for every onset."""
    if len(visual_ids) != len(vision_onsets):
        raise ValueError(
            f"visual_ids and vision_onsets must have the same length, got {len(visual_ids)} and {len(vision_onsets)}."
        )


def _validate_sfreq(sfreq: float) -> None:
    """Require a positive, finite sampling frequency."""
    if isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be a finite number greater than 0.")


def _validate_window(trial_window: list[float | int]) -> None:
    """Require a finite, ordered two-entry trial window."""
    if not isinstance(trial_window, (list, tuple)) or len(trial_window) != 2:
        raise ValueError("trial_window must contain exactly two entries.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value)
        for value in trial_window
    ):
        raise ValueError("trial_window entries must be finite real numbers.")
    if trial_window[0] >= trial_window[1]:
        raise ValueError("trial_window start must be less than end.")


def _validate_onsets(vision_onsets: np.ndarray) -> None:
    """Require integer-typed, finite, non-boolean onset samples."""
    if not np.issubdtype(vision_onsets.dtype, np.integer) or np.issubdtype(vision_onsets.dtype, np.bool_):
        raise ValueError("vision_onsets must contain finite, non-boolean integers.")


def _validate_neuro_shape(neuro_shape: tuple[int, ...], expected_ndim: int, mode: str) -> None:
    """Require the documented neural-array dimensions for a data mode."""
    if len(neuro_shape) != expected_ndim:
        raise ValueError(f"{mode} neuro data must be {expected_ndim}-D, got shape {neuro_shape}.")
    if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size <= 0 for size in neuro_shape):
        raise ValueError(f"{mode} neuro data must have positive time and channel dimensions.")


def _validate_continuous_boundaries(
    vision_onsets: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    ntime: int,
) -> None:
    """Require in-recording, pairwise-disjoint continuous trials."""
    if np.any(vision_onsets < 0) or np.any(vision_onsets >= ntime):
        raise ValueError("vision_onsets must be within the recording.")
    if np.any(trial_ends <= trial_starts):
        raise ValueError("continuous trial end must be greater than start.")
    if np.any(trial_starts < 0) or np.any(trial_ends > ntime):
        raise ValueError("continuous trial boundaries must be fully within the recording.")
    order = np.argsort(trial_starts, kind="stable")
    sorted_starts = trial_starts[order]
    sorted_ends = trial_ends[order]
    if np.any(sorted_starts[1:] < sorted_ends[:-1]):
        raise ValueError("continuous trial boundaries must not overlap.")


def _validate_trial_metadata(
    vision_info: VisionInfo,
    trial_info: TrialInfo,
    trial_meta: object,
    n_trials: int,
) -> None:
    """Validate metadata whose dimensions are tied to the trial structure."""
    stim_ids = vision_info.get("stim_ids")
    n_stim = vision_info.get("n_stim")
    if stim_ids is None or not isinstance(stim_ids, (list, tuple, np.ndarray)):
        raise ValueError("vision_info['stim_ids'] must be a one-dimensional sequence.")
    if isinstance(stim_ids, np.ndarray):
        _validate_1d("vision_info['stim_ids']", stim_ids)
    if isinstance(n_stim, (bool, np.bool_)) or not isinstance(n_stim, Integral) or int(n_stim) != len(stim_ids):
        raise ValueError("vision_info['n_stim'] must match the length of vision_info['stim_ids'].")

    if trial_meta is not None and len(trial_meta) != n_trials:  # ty: ignore[invalid-argument-type]
        raise ValueError(f"trial_meta length must match trial count {n_trials}, got {len(trial_meta)}.")  # ty: ignore[invalid-argument-type]

    for name in ("baseline", "trial_window"):
        value = trial_info.get(name)
        if value is None:
            raise ValueError(f"trial_info[{name!r}] is required.")
        shape = np.asarray(value).shape
        if shape not in ((2,), (n_trials, 2)):
            raise ValueError(f"trial_info[{name!r}] must have shape (2,) or ({n_trials}, 2), got {shape}.")


def validate_trial_structure_state(
    *,
    data_mode: str,
    neuro_shape: tuple[int, ...],
    stim_labels: np.ndarray,
    trial: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    vision_onsets: np.ndarray,
    vision_info: VisionInfo,
    trial_info: TrialInfo,
    trial_meta: object = None,
) -> None:
    """Validate a complete persisted or mutable trial structure."""
    expected_ndim = 2 if data_mode == "continuous" else 3
    _validate_neuro_shape(neuro_shape, expected_ndim=expected_ndim, mode=data_mode)

    labels = np.asarray(stim_labels)
    trial_array = np.asarray(trial)
    starts = np.asarray(trial_starts)
    ends = np.asarray(trial_ends)
    onsets = np.asarray(vision_onsets)
    for name, values in (("trial_starts", starts), ("trial_ends", ends), ("vision_onsets", onsets)):
        _validate_1d(name, values)
        _validate_onsets(values)
    starts = starts.astype(np.int64, copy=False)
    ends = ends.astype(np.int64, copy=False)
    onsets = onsets.astype(np.int64, copy=False)

    n_trials = len(starts)
    if len(ends) != n_trials or len(onsets) != n_trials:
        raise ValueError("trial_starts, trial_ends, and vision_onsets must have the same length.")

    if data_mode == "continuous":
        ntime = neuro_shape[0]
        if labels.shape != (ntime,):
            raise ValueError(f"continuous stim_labels must have shape ({ntime},), got {labels.shape}.")
        if trial_array.shape != (ntime,):
            raise ValueError(f"continuous trial must have shape ({ntime},), got {trial_array.shape}.")
        _validate_continuous_boundaries(onsets, starts, ends, ntime)
    else:
        expected_shape = neuro_shape[:2]
        if labels.shape != expected_shape:
            raise ValueError(f"epochs stim_labels must have shape {expected_shape}, got {labels.shape}.")
        if trial_array.shape != expected_shape:
            raise ValueError(f"epochs trial must have shape {expected_shape}, got {trial_array.shape}.")
        if n_trials != neuro_shape[0]:
            raise ValueError(f"trial array lengths must match epoch count {neuro_shape[0]}, got {n_trials}.")
        ntime = neuro_shape[1]
        if np.any(onsets < 0) or np.any(onsets >= ntime):
            raise ValueError("vision_onsets must be within each epoch.")
        if np.any(ends <= starts):
            raise ValueError("epoch trial end must be greater than start.")
        if np.any(starts < 0) or np.any(ends > ntime):
            raise ValueError("epoch trial boundaries must be within each epoch.")

    _validate_trial_metadata(vision_info, trial_info, trial_meta, n_trials)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_trial_structure_continuous(
    visual_ids: np.ndarray,
    trial_window: list[float | int],
    vision_onsets: np.ndarray,
    ntime: int,
    sfreq: float,
    neuro_shape: tuple[int, ...] | None = None,
) -> TrialStructure:
    """Build a :class:`TrialStructure` for continuous (raw) recordings.

    Parameters
    ----------
    visual_ids : np.ndarray
        Stimulus ID per onset, shape ``(n_onsets,)``.
    trial_window : list of float | int
        Two-element ``[start, end]`` relative to each onset.
        Float values are seconds; int values are samples.
    vision_onsets : np.ndarray
        Onset sample indices, shape ``(n_onsets,)``.
    ntime : int
        Total number of time samples in the recording.
    sfreq : float
        Sampling frequency in Hz.
    neuro_shape : tuple of int or None
        Full neural-array shape, validated as ``(ntime, nchan)`` when provided.

    Returns
    -------
    TrialStructure
    """
    visual_ids = _coerce_scalar_array(visual_ids)
    vision_onsets = np.asarray(vision_onsets)
    _validate_1d("visual_ids", visual_ids)
    _validate_1d("vision_onsets", vision_onsets)
    _validate_aligned(visual_ids, vision_onsets)
    _validate_onsets(vision_onsets)
    vision_onsets = vision_onsets.astype(np.int64, copy=False)
    _validate_sfreq(sfreq)
    _validate_window(trial_window)
    if neuro_shape is not None:
        _validate_neuro_shape(neuro_shape, expected_ndim=2, mode="continuous")
        if ntime != neuro_shape[0]:
            raise ValueError("ntime must match the continuous neuro time dimension.")
    elif isinstance(ntime, (bool, np.bool_)) or not isinstance(ntime, Integral) or ntime <= 0:
        raise ValueError("continuous neuro data must have positive time and channel dimensions.")
    tw_samples = _window_to_samples(trial_window, sfreq)
    if tw_samples[0] >= tw_samples[1]:
        raise ValueError("trial_window must span at least one sample after conversion.")
    trial_starts = vision_onsets + tw_samples[0]
    trial_ends = vision_onsets + tw_samples[1]
    _validate_continuous_boundaries(vision_onsets, trial_starts, trial_ends, ntime)
    stim_labels = _stim_labels_continuous(ntime, vision_onsets, visual_ids)
    trial = np.full(ntime, np.nan)
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends, strict=True)):
        trial[ts:te] = i
    vision_info = _build_vision_info(visual_ids)
    logger.info(
        "Configured (raw): {} trials, {} unique stimuli",
        len(trial_starts),
        vision_info["n_stim"],
    )
    return TrialStructure(
        stim_labels=stim_labels,
        trial=trial,
        trial_starts=trial_starts,
        trial_ends=trial_ends,
        vision_onsets=vision_onsets,
        vision_info=vision_info,
        trial_info={"baseline": [tw_samples[0], 0], "trial_window": tw_samples},
    )


def build_trial_structure_epochs(
    visual_ids: np.ndarray,
    vision_onsets: np.ndarray | None,
    neuro_shape: tuple,
    existing_vision_onsets: np.ndarray | None = None,
) -> TrialStructure:
    """Build a :class:`TrialStructure` for pre-epoched recordings.

    Parameters
    ----------
    visual_ids : np.ndarray
        Stimulus ID per trial, shape ``(n_trials,)``.
    vision_onsets : np.ndarray or None
        Per-trial onset offsets within each epoch.
    neuro_shape : tuple
        Shape of the neuro array ``(n_trials, n_timebins, ...)``.
    existing_vision_onsets : np.ndarray or None
        Fallback: onsets already stored on the Recording before this call.

    Returns
    -------
    TrialStructure
    """
    _validate_neuro_shape(neuro_shape, expected_ndim=3, mode="epochs")
    n_trials = neuro_shape[0]
    n_timebins = neuro_shape[1]
    visual_ids = _coerce_scalar_array(visual_ids)

    _validate_1d("visual_ids", visual_ids)
    if len(visual_ids) != n_trials:
        raise ValueError(f"visual_ids length must match epoch count {n_trials}, got {len(visual_ids)}.")

    if vision_onsets is not None:
        selected_onsets = np.asarray(vision_onsets)
    elif existing_vision_onsets is not None:
        selected_onsets = np.asarray(existing_vision_onsets)
    else:
        selected_onsets = np.zeros(n_trials, dtype=int)
        logger.warning("epochs data has no vision_onsets, defaulting to index 0 of each epoch")

    _validate_1d("vision_onsets", selected_onsets)
    _validate_onsets(selected_onsets)
    selected_onsets = selected_onsets.astype(np.int64, copy=False)
    if len(selected_onsets) != n_trials:
        raise ValueError(f"vision_onsets length must match epoch count {n_trials}, got {len(selected_onsets)}.")
    if np.any(selected_onsets < 0) or np.any(selected_onsets >= n_timebins):
        raise ValueError("vision_onsets must be within each epoch.")
    vision_onsets = selected_onsets

    stim_labels = _stim_labels_epochs(n_trials, n_timebins, vision_onsets, visual_ids)
    trial = np.stack([np.full(n_timebins, i, dtype=float) for i in range(n_trials)])
    vision_info = _build_vision_info(visual_ids)
    if np.all(vision_onsets == vision_onsets[0]):
        onset = int(vision_onsets[0])
        trial_info: TrialInfo = {
            "baseline": [-onset, 0],
            "trial_window": [-onset, n_timebins - onset],
        }
    else:
        trial_info: TrialInfo = {
            "baseline": [[-int(onset), 0] for onset in vision_onsets],
            "trial_window": [[-int(onset), n_timebins - int(onset)] for onset in vision_onsets],
        }
    logger.info("Configured (epochs): {} trials, {} unique stimuli", n_trials, vision_info["n_stim"])
    return TrialStructure(
        stim_labels=stim_labels,
        trial=trial,
        trial_starts=np.zeros(n_trials, dtype=int),
        trial_ends=np.full(n_trials, n_timebins, dtype=int),
        vision_onsets=vision_onsets,
        vision_info=vision_info,
        trial_info=trial_info,
    )


# ---------------------------------------------------------------------------
# Private helpers (importable for tests)
# ---------------------------------------------------------------------------


def _stim_labels_continuous(
    n_timebins: int,
    vision_onsets: np.ndarray,
    visual_ids: np.ndarray,
) -> np.ndarray:
    """Build stim_labels for a continuous recording, shape ``(n_timebins,)``."""
    arr = _make_nan_array(n_timebins, visual_ids.dtype.kind)
    for onset, sid in zip(vision_onsets, visual_ids, strict=True):
        arr[int(onset)] = sid
    return arr


def _stim_labels_epochs(
    n_trials: int,
    n_timebins: int,
    vision_onsets: np.ndarray,
    visual_ids: np.ndarray,
) -> np.ndarray:
    """Build stim_labels for pre-epoched data, shape ``(n_trials, n_timebins)``."""
    arr = _make_nan_array((n_trials, n_timebins), visual_ids.dtype.kind)
    for i, (onset, sid) in enumerate(zip(vision_onsets, visual_ids, strict=True)):
        arr[i, int(onset)] = sid
    return arr


def _window_to_samples(trial_window: list[float | int], sfreq: float) -> list[int]:
    """Convert a trial window ``[start, end]`` to sample offsets."""
    result: list[int] = []
    for val in trial_window:
        if isinstance(val, Integral):
            result.append(int(val))
        else:
            result.append(int(round(val * sfreq)))
    return result


def _build_vision_info(visual_ids: np.ndarray) -> VisionInfo:
    """Build vision metadata while preserving scalar ID types and order."""
    from vneurotk.core.stimulus import _unique_ordered_keys

    try:
        unique_ids = sorted(_unique_ordered_keys(visual_ids))
    except TypeError:
        unique_ids = _unique_ordered_keys(visual_ids)
    return {"n_stim": len(unique_ids), "stim_ids": unique_ids}
