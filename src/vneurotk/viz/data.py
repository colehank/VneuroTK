"""Visualization utilities for neural recordings."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import TypeAlias, cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from .utils import _is_null, _smart_ticks, _truncate_label

PlotNumber: TypeAlias = int | float | np.integer | np.floating
PlotWindow: TypeAlias = Sequence[PlotNumber]


def plot_data(
    neuro: np.ndarray,
    visual: np.ndarray,
    sfreq: float,
    trial: np.ndarray | None = None,
    trial_window: PlotWindow | Sequence[PlotWindow] | None = None,
    figsize: tuple[float, float] = (6, 3),
    window: tuple[float, float] = (0.0, 5.0),
    cmap_neuro: str = "Greys",
    cmap_ontime: str = "summer",
    color_offtime: str = "black",
    marker_size: float = 40,
) -> plt.Figure:
    """Plot neural activity alongside stimulus labels.

    Parameters
    ----------
    neuro : np.ndarray
        Neural data, shape ``(n_samples, n_channels)``.
    visual : np.ndarray
        Label vector, shape ``(n_samples,)``.
    sfreq : float
        Sampling frequency in Hz.
    trial : np.ndarray or None
        Trial-ID vector, shape ``(n_samples,)``.  When provided together
        with *trial_window*, every in-trial timepoint is scatter-plotted
        and coloured by its position inside the trial window.
    trial_window : list of real, list of two-value lists, or None
        A single ``[start, end]`` relative to stimulus onset, or one such
        window per trial ID.  Per-trial windows support epochs whose stimulus
        onset differs between trials.  Real non-integral values are interpreted
        as **seconds** and integral values as **samples** (the same numeric
        convention used by :meth:`~vneurotk.core.recording.BaseData.configure`).
    figsize : tuple of float
        Figure size ``(width, height)`` in inches.
    window : tuple of float | int
        Display window ``(start, end)``.  *float* values are interpreted
        as **seconds**, *int* values as **samples**.
    cmap_neuro : str
        Colormap for the neural activity heatmap.
    cmap_ontime : str
        Colormap for in-trial time of stimulus labels.
    color_offtime : str
        Color for non-stimulus time points.
    marker_size : float
        Marker size for scatter points.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    neuro, visual, sfreq, trial, window = _validate_plot_inputs(neuro, visual, sfreq, trial, trial_window, window)

    # -- Convert window to samples (non-integral real=seconds, integral=samples) --
    raw_start = _plot_bound_to_sample(window[0], sfreq)
    raw_end = _plot_bound_to_sample(window[1], sfreq)
    if raw_start >= raw_end:
        raise ValueError("window must place its start before end.")
    s_start = max(raw_start, 0)
    s_end = min(raw_end, neuro.shape[0])
    if s_start >= s_end:
        raise ValueError("window does not contain any samples from the recording.")
    if isinstance(window[0], Real) and not isinstance(window[0], Integral):
        logger.info("plot window: {}-{} s (samples {}-{}).", window[0], window[1], s_start, s_end)
    else:
        logger.info("plot window: {}-{} samples.", s_start, s_end)
    X_win = neuro[s_start:s_end]
    y_win = visual[s_start:s_end]
    times = np.arange(s_start, s_end) / sfreq
    t_min, t_max = times[0], times[-1]

    # -- Parse labels --
    if trial is not None and trial_window is not None:
        tw_samples = _trial_windows_to_samples(trial_window, sfreq)
        # Build stim_map from FULL arrays so edge-of-window trials are covered
        full_stim_map: dict[int, str] = {}
        for i in range(len(visual)):
            if not _is_null(visual[i]) and not _is_null(trial[i]):
                full_stim_map[int(trial[i])] = str(visual[i])

        trial_win = trial[s_start:s_end]
        is_stim, y_cat, intrial_time, tick_labels = _parse_labels_with_trial(
            y_win,
            trial_win,
            tw_samples,
            sfreq,
            full_stim_map,
        )
    else:
        is_stim, y_cat, intrial_time, tick_labels = _parse_labels(y_win)
        intrial_time = intrial_time / sfreq  # samples -> seconds

    # -- Layout --
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 0.015],
        height_ratios=[0.8, 1],
        hspace=0.12,
        wspace=0.02,
    )
    ax_y = fig.add_subplot(gs[0, 0])
    cax_y = fig.add_subplot(gs[0, 1])
    ax_x = fig.add_subplot(gs[1, 0], sharex=ax_y)
    cax_x = fig.add_subplot(gs[1, 1])

    # -- Upper panel: stimulus labels --
    ax_y.set_title("Trial setting", fontsize=10, loc="left")

    # Non-trial points
    if np.any(~is_stim):
        stride = max(1, len(y_win) // 5000)
        idx = np.where(~is_stim)[0][::stride]
        ax_y.scatter(
            times[idx],
            y_cat[idx],
            c=color_offtime,
            s=marker_size,
            marker=".",
            rasterized=True,
            linewidths=0,
            alpha=0.5,
        )

    # In-trial points with combined baseline+active colormap
    if np.any(is_stim):
        vmin = np.nanmin(intrial_time)
        vmax = np.nanmax(intrial_time)
        combined_cmap = _build_trial_cmap(
            vmin,
            vmax,
            cmap_ontime,
            color_offtime,
        )
        sc = ax_y.scatter(
            times[is_stim],
            y_cat[is_stim],
            c=intrial_time[is_stim],
            cmap=combined_cmap,
            vmin=vmin,
            vmax=vmax,
            s=marker_size,
            marker=".",
            rasterized=True,
            linewidths=0,
        )
        cbar_y = fig.colorbar(sc, cax=cax_y)
        cbar_y.set_label("In-trial Time (s)", fontsize=10)
        _apply_ticks(cbar_y, vmin, vmax, is_cbar=True)
    else:
        cax_y.axis("off")

    ax_y.set_yticks(range(len(tick_labels)))
    ax_y.set_yticklabels(tick_labels, fontsize=8)
    ax_y.set_ylabel("Label", fontsize=10)
    ax_y.grid(True, alpha=0.3, axis="y")
    ax_y.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # -- Lower panel: neural activity --
    ax_x.set_title("Neural Activity", fontsize=10, loc="left")
    im = ax_x.imshow(
        X_win.T,
        aspect="auto",
        origin="lower",
        extent=(t_min, t_max, 0, X_win.shape[1]),
        cmap=cmap_neuro,
        interpolation="nearest",
    )
    cbar_x = fig.colorbar(im, cax=cax_x)
    cbar_x.set_label("Amplitude", fontsize=10)

    _apply_ticks(cbar_x, np.min(X_win), np.max(X_win), is_cbar=True)
    _apply_ticks(ax_x, t_min, t_max, axis="x")
    _apply_ticks(ax_x, 0, X_win.shape[1], axis="y", force_int=True)

    ax_x.set_xlabel("In-sample Time (s)", fontsize=10)
    ax_x.set_ylabel("Channel", fontsize=10)

    # -- Cleanup --
    for ax in (ax_y, ax_x):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_y.spines["bottom"].set_visible(False)
    fig.align_ylabels([ax_y, ax_x])

    return fig


def _validate_plot_inputs(
    neuro: np.ndarray,
    visual: np.ndarray,
    sfreq: float,
    trial: np.ndarray | None,
    trial_window: PlotWindow | Sequence[PlotWindow] | None,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None, tuple[float, float]]:
    """Validate and normalize the public plotting inputs."""
    neuro = np.asarray(neuro)
    visual = np.asarray(visual)
    if neuro.ndim != 2:
        raise ValueError(f"neuro must be a nonempty two-dimensional array, got shape {neuro.shape}.")
    if neuro.shape[0] == 0 or neuro.shape[1] == 0:
        raise ValueError(f"neuro must be a nonempty two-dimensional array, got shape {neuro.shape}.")
    if visual.ndim != 1 or len(visual) != neuro.shape[0]:
        raise ValueError("visual must be one-dimensional with the same number of samples as neuro.")
    if isinstance(sfreq, (bool, np.bool_)) or not isinstance(sfreq, Real) or not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive and finite.")
    if len(window) != 2 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) for value in window
    ):
        raise ValueError("window must contain exactly two finite real bounds.")
    if (trial is None) != (trial_window is None):
        raise ValueError("trial and trial_window must be provided together.")
    if trial is not None:
        trial = np.asarray(trial)
        if trial.ndim != 1 or len(trial) != neuro.shape[0]:
            raise ValueError("trial must be one-dimensional with the same number of samples as neuro.")
        _validate_trial_windows(trial, cast(PlotWindow | Sequence[PlotWindow], trial_window), float(sfreq))
    return neuro, visual, float(sfreq), trial, window


def _validate_trial_windows(
    trial: np.ndarray,
    trial_window: PlotWindow | Sequence[PlotWindow],
    sfreq: float,
) -> None:
    """Validate trial identifiers and their shared or per-trial windows."""
    trial_ids: set[int] = set()
    for value in trial:
        if _is_null(value):
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("trial IDs must be null or nonnegative integers.")
        trial_id = int(value)
        if trial_id != value or trial_id < 0:
            raise ValueError("trial IDs must be null or nonnegative integers.")
        trial_ids.add(trial_id)

    converted = _trial_windows_to_samples(trial_window, sfreq)
    shared = len(converted) == 2 and all(isinstance(value, Integral) for value in converted)
    windows = [cast(list[int], converted)] if shared else cast(list[list[int]], converted)
    if not shared and trial_ids and max(trial_ids) >= len(windows):
        raise ValueError("trial_window must provide a window for every referenced trial ID.")
    selected = windows if shared else [windows[trial_id] for trial_id in trial_ids]
    if any(len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in selected):
        raise ValueError("each trial window must contain two ordered bounds spanning at least one sample.")


def _plot_bound_to_sample(value: PlotNumber, sfreq: float) -> int:
    """Convert one plotting bound using the public real/integral convention."""
    if isinstance(value, Integral):
        return int(value)
    return int(round(float(value) * sfreq))


def _trial_windows_to_samples(
    trial_window: PlotWindow | Sequence[PlotWindow],
    sfreq: float,
) -> list[int] | list[list[int]]:
    """Convert a shared or per-trial plotting window to sample offsets."""
    if len(trial_window) == 2 and all(isinstance(value, Real) for value in trial_window):
        shared = cast(PlotWindow, trial_window)
        return [_plot_bound_to_sample(value, sfreq) for value in shared]
    per_trial = cast(Sequence[PlotWindow], trial_window)
    return [[_plot_bound_to_sample(value, sfreq) for value in window] for window in per_trial]


def _build_trial_cmap(
    vmin: float,
    vmax: float,
    active_cmap: str,
    baseline_color: str,
    n_colors: int = 256,
) -> mcolors.Colormap:
    """Build a colormap that is constant *baseline_color* for values < 0
    and the given *active_cmap* for values >= 0.

    If *vmin* >= 0 (no baseline), the original *active_cmap* is returned.
    """
    if vmin >= 0:
        return plt.get_cmap(active_cmap)

    span = vmax - vmin
    n_baseline = max(1, int(round(abs(vmin) / span * n_colors)))
    n_active = n_colors - n_baseline

    base_rgba = mcolors.to_rgba(baseline_color)
    act_cmap = plt.get_cmap(active_cmap)
    colors = [base_rgba] * n_baseline + [act_cmap(i / max(n_active - 1, 1)) for i in range(n_active)]
    return mcolors.ListedColormap(colors)


def _apply_ticks(
    target,
    vmin: float,
    vmax: float,
    *,
    is_cbar: bool = False,
    axis: str = "x",
    force_int: bool = False,
):
    """Apply smart ticks to an Axes or colorbar.

    Parameters
    ----------
    target : matplotlib Axes or colorbar
        The object to format.
    vmin, vmax : float
        Data range.
    is_cbar : bool
        True when *target* is a colorbar.
    axis : ``'x'`` or ``'y'``
        Which axis to format (ignored for colorbars).
    force_int : bool
        Round tick values to integers.
    """
    if np.isnan(vmin) or np.isnan(vmax) or vmin == vmax:
        return

    ticks, fmt, offset_str = _smart_ticks(vmin, vmax, force_int=force_int)

    if is_cbar:
        target.set_ticks(ticks)
        target.ax.yaxis.set_major_formatter(fmt)
        target.ax.tick_params(labelsize=8)
        if offset_str:
            target.ax.set_title(offset_str, fontsize=8, pad=4, loc="left")
    elif axis == "y":
        target.set_yticks(ticks)
        target.set_ylim(vmin, vmax)
        target.yaxis.set_major_formatter(fmt)
        target.tick_params(axis="y", labelsize=8)
    else:
        target.set_xticks(ticks)
        target.set_xlim(vmin, vmax)
        target.xaxis.set_major_formatter(fmt)
        target.tick_params(axis="x", labelsize=8)


def _parse_labels(y_win: np.ndarray):
    """Parse a label vector into categorical indices and in-trial times.

    Parameters
    ----------
    y_win : np.ndarray
        Label vector (dtype object), length *n*.

    Returns
    -------
    is_stim : np.ndarray
        Boolean mask where True = stimulus present.
    y_cat : np.ndarray
        Integer category index per sample (0 = none).
    intrial_time : np.ndarray
        Seconds since trial onset; NaN outside trials.
    tick_labels : list of str
        Tick labels ``["none", stim_0, stim_1, ...]``.
    """
    n = len(y_win)

    is_none = np.array([_is_null(v) for v in y_win])
    is_stim = ~is_none
    y_clean = np.array(
        [None if null else str(v) for null, v in zip(is_none, y_win, strict=True)],
        dtype=object,
    )

    # In-trial time: seconds since each trial onset
    intrial_time = np.full(n, np.nan)
    if np.any(is_stim):
        changed = np.concatenate(([True], y_clean[1:] != y_clean[:-1]))
        starts = np.where(is_stim & changed)[0]
        for s in starts:
            e = s + 1
            while e < n and y_clean[e] == y_clean[s]:
                e += 1
            intrial_time[s:e] = np.arange(e - s, dtype=float)

    # Categorical mapping
    unique_stims = list(dict.fromkeys(v for v in y_clean if v is not None))
    cat_map = {None: 0, **{stim: i + 1 for i, stim in enumerate(unique_stims)}}
    y_cat = np.array([cat_map[v] for v in y_clean])
    tick_labels = ["none"] + [_truncate_label(s) for s in unique_stims]

    return is_stim, y_cat, intrial_time, tick_labels


def _parse_labels_with_trial(
    y_win: np.ndarray,
    trial_win: np.ndarray,
    trial_windows: list[int] | list[list[int]],
    sfreq: float,
    stim_map: dict[int, str] | None = None,
):
    """Parse labels using the trial array so every in-trial point is coloured.

    Parameters
    ----------
    y_win : np.ndarray
        Sliced visual array (length *n*).
    trial_win : np.ndarray
        Sliced trial array (length *n*).  Non-NaN values give the trial ID.
    trial_windows : list of int or list of two-int lists
        Shared ``[start, end]`` in samples relative to onset, or one window
        per trial ID.
    sfreq : float
        Sampling frequency.
    stim_map : dict or None
        Pre-built mapping ``{trial_id: stim_label}``.  When *None* the
        mapping is derived from *y_win* (may miss trials whose onset is
        outside the window).

    Returns
    -------
    is_in_trial, y_cat, intrial_time, tick_labels
        Same semantics as :func:`_parse_labels` but *is_in_trial* marks
        **every** in-trial timepoint (not just onsets).
    """
    n = len(y_win)
    shared_window = len(trial_windows) == 2 and all(isinstance(value, Integral) for value in trial_windows)
    shared_trial_window = cast(list[int], trial_windows) if shared_window else None
    per_trial_windows = cast(list[list[int]], trial_windows) if not shared_window else None

    # --- identify in-trial timepoints ---
    is_in_trial = np.array([not _is_null(v) for v in trial_win])

    # --- map trial_id -> stim label ---
    if stim_map is None:
        stim_map = {}
        for i in range(n):
            if not _is_null(y_win[i]) and is_in_trial[i]:
                stim_map[int(trial_win[i])] = str(y_win[i])

    # --- categorical mapping (only stims visible in the window) ---
    unique_tids = sorted(set(int(trial_win[i]) for i in range(n) if is_in_trial[i]))
    window_stims = dict.fromkeys(stim_map[tid] for tid in unique_tids if tid in stim_map)
    unique_stims = list(window_stims)
    cat_map = {None: 0}
    cat_map.update({s: i + 1 for i, s in enumerate(unique_stims)})

    y_cat = np.zeros(n, dtype=int)
    intrial_time = np.full(n, np.nan)

    # --- assign y-category and intrial_time for every in-trial point ---
    for tid in unique_tids:
        mask = np.array([is_in_trial[i] and int(trial_win[i]) == tid for i in range(n)])
        indices = np.where(mask)[0]
        count = len(indices)
        if shared_trial_window is not None:
            trial_window = shared_trial_window
        else:
            assert per_trial_windows is not None
            trial_window = per_trial_windows[tid]
        tw_start, tw_end = trial_window
        trial_len = tw_end - tw_start

        # y category
        stim_label = stim_map.get(tid)
        y_cat[mask] = cat_map.get(stim_label, 0)

        # intrial time: handle partial trials at window boundary
        if count < trial_len and indices[0] == 0:
            # partial trial at start — seeing the tail
            offset = trial_len - count
        else:
            offset = 0
        intrial_time[mask] = (np.arange(count) + offset + tw_start) / sfreq

    # --- baseline points (intrial_time < 0) fall back to "none" ---
    is_baseline = is_in_trial & (intrial_time < 0)
    y_cat[is_baseline] = 0

    tick_labels = ["none"] + [_truncate_label(s) for s in unique_stims]

    return is_in_trial, y_cat, intrial_time, tick_labels
