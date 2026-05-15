"""Trial structure computation for neural recordings.

All knowledge of how to map stimulus onsets + trial windows onto sample indices
lives here.  Two public factory functions produce :class:`TrialStructure` value
objects from raw arrays — no :class:`BaseData` required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger


def _make_nan_array(shape: tuple | int, dtype_kind: str) -> np.ndarray:
    """Return NaN-filled array; object dtype for string IDs."""
    if dtype_kind in ("U", "S", "O"):
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
    vision_info: dict
    trial_info: dict


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_trial_structure_continuous(
    visual_ids: np.ndarray,
    trial_window: list[float | int],
    vision_onsets: np.ndarray,
    ntime: int,
    sfreq: float,
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

    Returns
    -------
    TrialStructure
    """
    vision_onsets = np.asarray(vision_onsets, dtype=int)
    tw_samples = _window_to_samples(trial_window, sfreq)
    trial_starts = vision_onsets + tw_samples[0]
    trial_ends = vision_onsets + tw_samples[1]
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
    n_trials = neuro_shape[0]
    n_timebins = neuro_shape[1]

    if vision_onsets is not None:
        vision_onsets = np.asarray(vision_onsets, dtype=int)
    elif existing_vision_onsets is not None:
        vision_onsets = existing_vision_onsets
    else:
        vision_onsets = np.zeros(n_trials, dtype=int)
        logger.warning("epochs data has no vision_onsets, defaulting to index 0 of each epoch")

    stim_labels = _stim_labels_epochs(n_trials, n_timebins, vision_onsets, visual_ids)
    trial = np.stack([np.full(n_timebins, i, dtype=float) for i in range(n_trials)])
    vision_info = _build_vision_info(visual_ids)
    logger.info("Configured (epochs): {} trials, {} unique stimuli", n_trials, vision_info["n_stim"])
    return TrialStructure(
        stim_labels=stim_labels,
        trial=trial,
        trial_starts=np.zeros(n_trials, dtype=int),
        trial_ends=np.full(n_trials, n_timebins, dtype=int),
        vision_onsets=vision_onsets,
        vision_info=vision_info,
        trial_info={
            "baseline": [-int(vision_onsets[0]), 0],
            "trial_window": [-int(vision_onsets[0]), n_timebins - int(vision_onsets[0])],
        },
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
        if isinstance(val, float):
            result.append(int(round(val * sfreq)))
        else:
            result.append(int(val))
    return result


def _build_vision_info(visual_ids: np.ndarray) -> dict:
    """Build the ``vision_info`` dict from a stimulus ID array.

    Parameters
    ----------
    visual_ids : np.ndarray
        Stimulus IDs for all onsets / trials.

    Returns
    -------
    dict
        ``{"n_stim": int, "stim_ids": list}`` with sorted unique IDs.
    """
    unique_ids = sorted(np.unique(visual_ids).tolist())
    return {"n_stim": len(unique_ids), "stim_ids": unique_ids}
