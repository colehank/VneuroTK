"""Dictionary-compatible metadata boundary types used by recordings."""

from __future__ import annotations

from typing import Any, TypedDict


class NeuroInfo(TypedDict, total=False):
    """Known keys accepted in :attr:`BaseData.neuro_info`.

    The runtime value remains an ordinary mutable ``dict``. ``total=False``
    preserves existing inputs, including pattern data without a sampling rate
    and lazy data that only declares a shape.
    """

    sfreq: float | int | None
    ch_names: list[str]
    highpass: float | None
    lowpass: float | None
    source_file: str
    shape: tuple[int, ...] | list[int]


class VisionInfo(TypedDict, total=False):
    """Known keys accepted in :attr:`BaseData.vision_info`."""

    n_stim: int
    stim_ids: list[Any]
    teststim: list[Any]


class TrialInfo(TypedDict, total=False):
    """Known keys accepted in :attr:`BaseData.trial_info`.

    Epochs with varying onset positions use one window per trial, while
    continuous and uniformly aligned epochs use a single two-value window.
    """

    baseline: list[int] | list[list[int]]
    trial_window: list[int] | list[list[int]]
