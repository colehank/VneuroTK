"""Joint Data Object — the top-level container for VneuroTK.

This module provides :class:`BaseData`, which couples a neural Recording
(time-series, trial structure) with a Stimulus Set and optional Visual
Representations.  Neither a purely neural concept nor a visual one — it is
the entity that links both domains together for a single experiment unit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from vneurotk.core.info import Info
from vneurotk.core.stimulus import StimulusSet
from vneurotk.neuro.base import NeuroData
from vneurotk.neuro.trial import TrialStructure, build_trial_structure_continuous, build_trial_structure_epochs

NeuroLoader = Callable[[], np.ndarray]  # lazy loader contract


class BaseData:
    """Unified container for neural data, stimulus labels, and trial structure.

    Parameters
    ----------
    neuro : np.ndarray | None
        Neural data array.  ``None`` when using lazy loading.
        Shape ``(ntime, nchan)`` → ``data_mode="continuous"``;
        ``(n_trials, n_timebins, nchan)`` → ``data_mode="epochs"``;
        ``(n, nchan)`` with ``data_mode="patterns"`` for aggregated data.
    neuro_info : dict
        Metadata dict.  Required key: ``sfreq``.  Optional keys:
        ``ch_names``, ``highpass``, ``lowpass``, ``source_file``, ``shape``.
    stim_labels : np.ndarray | None
        Internal stimulus-label array of shape ``(ntime,)`` or
        ``(n_trials, n_timebins)``.  ``np.nan`` at non-stimulus timepoints,
        stimulus ID at onset timepoints.  Not exposed directly; use
        :attr:`trial_stim_ids`.
    vision_info : dict | None
        Dict with ``n_stim`` (int) and ``stim_ids`` (list).
    trial : np.ndarray | None
        Trial-ID array of shape ``(ntime,)``.  ``np.nan`` outside trials.
    trial_info : dict | None
        Dict with ``baseline`` (list[int]) and ``trial_window`` (list).
    trial_starts : np.ndarray | None
        Start sample indices per trial, shape ``(n_trials,)``.
    trial_ends : np.ndarray | None
        End sample indices per trial, shape ``(n_trials,)``.
    vision_onsets : np.ndarray | None
        Stimulus onset sample indices, shape ``(n_trials,)``.
    trial_meta : pd.DataFrame | None
        Per-trial metadata table.
    data_mode : str or None
        ``"continuous"`` for 2-D time-series ``(ntime, nchan)``,
        ``"epochs"`` for 3-D trial-epoched ``(n_trials, n_timebins, nchan)``,
        ``"patterns"`` for 2-D aggregated ``(n, nchan)``.
        ``None`` triggers auto-inference from ``neuro.ndim``
        (3-D → ``"epochs"``, 2-D → ``"continuous"``).

    Examples
    --------
    >>> import numpy as np
    >>> neuro = np.random.randn(1000, 64)
    >>> info = dict(sfreq=250.0, ch_names=[f"ch{i}" for i in range(64)])
    >>> bd = BaseData(neuro, info)
    >>> bd
    BaseData(ntime=1000, nchan=64, n_trials=0, configured=False)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        neuro: np.ndarray | None,
        neuro_info: dict[str, Any],
        stim_labels: np.ndarray | None = None,
        vision_info: dict[str, Any] | None = None,
        trial: np.ndarray | None = None,
        trial_info: dict[str, Any] | None = None,
        trial_starts: np.ndarray | None = None,
        trial_ends: np.ndarray | None = None,
        vision_onsets: np.ndarray | None = None,
        trial_meta: Any = None,
        data_mode: str | None = None,
    ) -> None:
        self._neuro: np.ndarray | None = np.asarray(neuro) if neuro is not None else None
        self._neuro_loader: NeuroLoader | None = None
        self.neuro_info = neuro_info

        self._stim_labels = stim_labels
        self.vision_info = vision_info
        self.trial = trial
        self.trial_info = trial_info
        self.trial_starts = trial_starts
        self.trial_ends = trial_ends
        self.vision_onsets = vision_onsets
        self.trial_meta = trial_meta

        self.data_mode: str = self._infer_data_mode(self._neuro, data_mode)
        self._vision: Any = None  # legacy: VisualRepresentations | ndarray
        self._vision_data: Any = None

        logger.debug("BaseData created: ntime={}, nchan={}", self.ntime, self.nchan)

    @classmethod
    def for_continuous(
        cls,
        neuro: np.ndarray | None = None,
        neuro_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseData:
        """Factory for continuous (2-D time-series) recordings.

        Parameters
        ----------
        neuro : np.ndarray or None
            Neural data, shape ``(ntime, nchan)``.  Pass ``None`` when
            using lazy loading via :meth:`set_neuro_loader`.
        neuro_info : dict or None
            Metadata dict; ``sfreq`` key is required for most operations.
        **kwargs
            Any other :class:`BaseData` constructor parameters
            (e.g. ``stim_labels``, ``trial_info``).

        Returns
        -------
        BaseData
        """
        return cls(neuro=neuro, neuro_info=neuro_info or {}, data_mode="continuous", **kwargs)

    @classmethod
    def for_epochs(
        cls,
        neuro: np.ndarray | None = None,
        neuro_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseData:
        """Factory for pre-epoched recordings.

        Parameters
        ----------
        neuro : np.ndarray or None
            Neural data, shape ``(n_trials, n_timebins, nchan)``.
            Pass ``None`` when using lazy loading.
        neuro_info : dict or None
            Metadata dict; ``sfreq`` key is required for most operations.
        **kwargs
            Any other :class:`BaseData` constructor parameters.

        Returns
        -------
        BaseData
        """
        return cls(neuro=neuro, neuro_info=neuro_info or {}, data_mode="epochs", **kwargs)

    # ------------------------------------------------------------------
    # neuro property (lazy loading)
    # ------------------------------------------------------------------

    @property
    def neuro(self) -> NeuroData:
        """Neural data as a :class:`NeuroData`.

        Behaves like a plain ndarray; additionally exposes
        ``.epochs`` and ``.continuous`` for trial-structured views.
        """
        if self._neuro is None and self._neuro_loader is not None:
            logger.info("Lazy-loading neuro data...")
            self._neuro = self._neuro_loader()
            self._neuro_loader = None
        if self._neuro is None:
            raise RuntimeError("neuro data is not available. Call .load() or set a neuro loader first.")
        return NeuroData(self._neuro, self.trial_starts, self.trial_ends, self.data_mode)

    @neuro.setter
    def neuro(self, value: np.ndarray | None) -> None:
        self._neuro = np.asarray(value) if value is not None else None
        self._neuro_loader = None

    def set_neuro_loader(self, loader: NeuroLoader) -> None:
        """Register a lazy loader for the neuro array.

        Parameters
        ----------
        loader : NeuroLoader
            Callable with no arguments that returns ``np.ndarray`` when called.
            The loader is invoked once on the first access of :attr:`neuro` and
            its result is cached.
        """
        self._neuro = None
        self._neuro_loader = loader

    # ------------------------------------------------------------------
    # Vision attachment
    # ------------------------------------------------------------------

    @property
    def vision(self) -> Any:
        """DNN feature store for this dataset.

        Returns a :class:`VisionData` with the following interface:

        - ``db``         — original stimulus image dict (``vision_db``).
        - ``stim_ids``   — per-onset stimulus IDs, shape ``(n_trials,)``.
        - ``meta``       — :class:`~pandas.DataFrame` with one row per stored
          :class:`~vneurotk.vision.representation.VisualRepresentation`.
        - ``vision[mask]`` — smart accessor: string / int / bool-mask index;
          single VR → aligned ``ndarray``; multiple VRs → ``VisualRepresentations``.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        if not self.configured:
            raise RuntimeError("BaseData has not been configured. Call configure() first, or check .is_configured.")
        if self._vision_data is None:
            try:
                from vneurotk.vision.data import VisionData
            except ImportError as e:
                raise RuntimeError(
                    "Vision features require optional dependencies (torch, etc.). "
                    "Install them with: uv add vneurotk[vision]"
                ) from e
            self._vision_data = VisionData(self.trial_stim_ids)
        return self._vision_data

    @property
    def has_vision(self) -> bool:
        """Whether any DNN features have been stored via :attr:`vision`.extract_from()."""
        return self._vision_data is not None and len(self._vision_data.meta) > 0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def _time_axis_index(self) -> int:
        """Return the axis index that corresponds to time samples.

        Returns
        -------
        int
            ``1`` for ``data_mode="epochs"`` (shape is ``(n_trials, n_timebins, n_chan)``);
            ``0`` otherwise (shape is ``(n_timebins, n_chan)``).
        """
        return 1 if self.data_mode == "epochs" else 0

    def _neuro_shape_dim(self, axis: int) -> int:
        """Return shape dimension *axis* from neuro array or neuro_info, else 0.

        Checks ``self._neuro`` first; falls back to ``neuro_info["shape"]``; returns
        ``0`` when neither is available.  Axis ``-1`` is supported.

        Parameters
        ----------
        axis : int
            Axis index to query (e.g. ``-1`` for channels, ``0``/``1`` for time).

        Returns
        -------
        int
        """
        if self._neuro is not None:
            return self._neuro.shape[axis]
        shape = self.neuro_info.get("shape")
        return shape[axis] if shape is not None else 0

    @property
    def ntime(self) -> int:
        """Number of time samples (first axis for continuous/patterns; second for epochs)."""
        return self._neuro_shape_dim(self._time_axis_index())

    @property
    def nchan(self) -> int:
        """Number of channels."""
        v = self._neuro_shape_dim(-1)
        if v:
            return v
        ch_names = self.neuro_info.get("ch_names")
        return len(ch_names) if ch_names is not None else 0

    @property
    def n_timepoints(self) -> int:
        """Time points per trial."""
        if self.data_mode == "epochs":
            return self.neuro.shape[1]
        if self.trial_starts is not None and self.trial_ends is not None:
            return int(self.trial_ends[0] - self.trial_starts[0])
        return self.ntime

    @property
    def configured(self) -> bool:
        """Whether :meth:`configure` has been called."""
        return self._stim_labels is not None and self.trial is not None

    @property
    def is_configured(self) -> bool:
        """Alias for :attr:`configured`. ``True`` after :meth:`configure` succeeds."""
        return self.configured

    @property
    def is_vision_ready(self) -> bool:
        """``True`` when DNN features have been extracted and :attr:`vision` is safe to access."""
        return self._vision_data is not None and self._vision_data.has_visual_representations

    @property
    def n_trials(self) -> int:
        """Number of trials (0 if not configured)."""
        if self.trial_starts is None:
            return 0
        return len(self.trial_starts)

    def _stim_id_at_trial(self, i: int) -> Any:
        """Return the stimulus ID presented at trial *i*.

        Parameters
        ----------
        i : int
            Trial index (zero-based).

        Returns
        -------
        Any
            Element from ``_stim_labels`` at the vision onset of trial *i*.
            For ``data_mode="epochs"`` the labels array is 2-D and indexed as
            ``[i, onset]``; for continuous/patterns it is 1-D and indexed as
            ``[onset]``.
        """
        onset = int(self.vision_onsets[i])  # ty: ignore[not-subscriptable]
        if self.data_mode == "epochs":
            return self._stim_labels[i, onset]  # ty: ignore[not-subscriptable]
        return self._stim_labels[onset]  # ty: ignore[not-subscriptable]

    @property
    def trial_stim_ids(self) -> np.ndarray:
        """Stimulus ID at the onset of each trial, shape ``(n_trials,)``.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        if not self.configured:
            raise RuntimeError("BaseData not configured. Call configure() first.")
        return np.array([self._stim_id_at_trial(i) for i in range(self.n_trials)])

    @property
    def stim_labels(self) -> np.ndarray | None:
        """Raw stimulus label array from the trial layout.

        Shape depends on ``data_mode``:

        - ``"continuous"`` → ``(ntime,)``
        - ``"epochs"`` → ``(n_trials, n_timebins)``

        ``None`` before :meth:`configure` is called.
        """
        return self._stim_labels

    def _restore_vision_data(self, store: Any) -> None:
        """Controlled write point for reconstructed VisionData (used by h5_persistence).

        Parameters
        ----------
        store : VisionData or None
            Reconstructed :class:`~vneurotk.vision.data.VisionData` instance,
            or ``None`` to clear.
        """
        self._vision_data = store

    @property
    def info(self) -> Info:
        """Summary of neuro, visual, and trial metadata."""
        return Info(
            neuro={
                "n_time": self.ntime,
                "n_chan": self.nchan,
                "sfreq": self.neuro_info.get("sfreq"),
                "highpass": self.neuro_info.get("highpass"),
                "lowpass": self.neuro_info.get("lowpass"),
            },
            visual=self.vision_info,
            trial=self.trial_info,
            configured=self.configured,
            data_mode=self.data_mode,
        )

    # ------------------------------------------------------------------
    # configure()
    # ------------------------------------------------------------------

    def configure(
        self,
        stim_ids: np.ndarray | list,
        trial_window: list[float | int] | None = None,
        vision_onsets: np.ndarray | None = None,
        vision_db: dict | list | np.ndarray | None = None,
    ) -> None:
        """Attach stimulus and trial structure to the data.

        For continuous data (``data_mode == "continuous"``), both
        *trial_window* and *vision_onsets* are required.

        For pre-epoched data (``data_mode == "epochs"``), both parameters
        are optional: *vision_onsets* falls back to any already-stored value,
        then defaults to index 0 of each epoch; *trial_window* is ignored.

        Parameters
        ----------
        stim_ids : array-like, shape (n_onsets,)
            Stimulus ID for each onset / trial, must match *vision_onsets*
            length and order.
        trial_window : list of float | int or None
            Two-element ``[start, end]`` relative to each onset.
            Float → seconds; int → samples.
            Required for continuous data; ignored for epochs data.
        vision_onsets : np.ndarray or None
            1-D array of stimulus onset sample indices.
            Required for continuous data.
            For epochs data defaults to already-stored value or 0.
        vision_db : dict, list, np.ndarray, or None
            Stimulus image source.  Stored immediately as the Stimulus Set for
            this Recording.  Can also be supplied later via
            :meth:`extract_features`.  If a Stimulus Set is already attached,
            it is replaced and an ``info`` message is logged.
        """
        if self.configured:
            logger.warning("re-configuring already configured BaseData, overwriting trial structure")

        visual_ids = np.asarray(stim_ids)

        if self.data_mode == "patterns":
            raise ValueError("configure() is not supported for data_mode='patterns'.")

        if self.data_mode == "epochs":
            if self._neuro is None:
                raise RuntimeError("neuro data must be available before configuring epochs data. Call .load() first.")
            ts = build_trial_structure_epochs(
                visual_ids,
                vision_onsets,
                self._neuro.shape,
                existing_vision_onsets=self.vision_onsets,
            )
        else:
            if trial_window is None or vision_onsets is None:
                raise ValueError("trial_window and vision_onsets are required for continuous data.")
            ts = build_trial_structure_continuous(
                visual_ids, trial_window, vision_onsets, self.ntime, self.neuro_info["sfreq"]
            )
        self._apply_trial_structure(ts)

        if vision_db is not None:
            if self.vision.db is not None:
                logger.info("configure: replacing existing Stimulus Set with newly provided one.")
            self.vision.attach_db(StimulusSet(self.trial_stim_ids, vision_db))

    def _apply_trial_structure(self, ts: TrialStructure) -> None:
        """Write all trial-structure fields from *ts* to self atomically."""
        self._stim_labels = ts.stim_labels
        self.trial = ts.trial
        self.trial_starts = ts.trial_starts
        self.trial_ends = ts.trial_ends
        self.vision_onsets = ts.vision_onsets
        self.vision_info = ts.vision_info
        self.trial_info = ts.trial_info

    # ------------------------------------------------------------------
    # Explicit load
    # ------------------------------------------------------------------

    def load(self) -> BaseData:
        """Explicitly load neuro data into memory and return self.

        Returns
        -------
        BaseData
            self, for method chaining.
        """
        if self._neuro is None and self._neuro_loader is not None:
            _ = self.neuro
        elif self._neuro is not None:
            logger.debug("neuro already loaded, skipping .load()")
        return self

    # ------------------------------------------------------------------
    # plot()
    # ------------------------------------------------------------------

    def plot(
        self,
        window: tuple[float | int, float | int] = (0.0, 5.0),
        figsize: tuple[float, float] = (6, 3),
        cmap_neuro: str = "Greys",
        cmap_ontime: str = "summer",
        color_offtime: str = "black",
        marker_size: float = 40,
    ):
        """Plot neural activity alongside stimulus labels.

        Parameters
        ----------
        window : tuple of float | int
            Display window.  Float values are seconds, int values are samples.
        figsize : tuple of float
            Figure size ``(width, height)``.
        cmap_neuro : str
            Colormap for neural heatmap.
        cmap_ontime : str
            Colormap for in-trial time.
        color_offtime : str
            Color for off-trial points.
        marker_size : float
            Scatter marker size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from vneurotk.viz.data import plot_data

        tw = self.trial_info["trial_window"] if self.trial_info is not None else None

        neuro = self.neuro.data
        stim_labels: np.ndarray = self._stim_labels if self._stim_labels is not None else np.zeros(neuro.shape[0])
        trial = self.trial
        if self.data_mode == "epochs":
            neuro = neuro.reshape(-1, neuro.shape[-1])
            if stim_labels is not None:
                stim_labels = stim_labels.ravel()
            if trial is not None:
                trial = trial.ravel()

        return plot_data(
            neuro=neuro,
            visual=stim_labels,
            sfreq=self.neuro_info["sfreq"],
            trial=trial,
            trial_window=tw,
            figsize=figsize,
            window=window,
            cmap_neuro=cmap_neuro,
            cmap_ontime=cmap_ontime,
            color_offtime=color_offtime,
            marker_size=marker_size,
        )

    # ------------------------------------------------------------------
    # save()
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        """Persist the configured data to an HDF5 file.

        Parameters
        ----------
        path : VTKPath | pathlib.Path | str
            Destination file path.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        if not self.configured:
            raise RuntimeError("Cannot save unconfigured BaseData. Call configure() first.")

        from vneurotk.io.h5_persistence import save_recording

        save_recording(self, self._resolve_path(path))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_data_mode(neuro: np.ndarray | None, explicit: str | None) -> str:
        if explicit is not None:
            return explicit
        if neuro is not None and neuro.ndim == 3:
            return "epochs"
        return "continuous"

    @staticmethod
    def _resolve_path(path: Any) -> Path:
        if hasattr(path, "fpath"):
            return Path(path.fpath)
        return Path(path)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"BaseData(ntime={self.ntime}, nchan={self.nchan}",
            f"n_trials={self.n_trials}, configured={self.configured}",
            f"data_mode='{self.data_mode}'",
        ]
        if self.has_vision:
            parts.append("has_vision=True")
        if self._neuro is None and self._neuro_loader is not None:
            parts.append("neuro=<lazy>")
        return ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        return self.info._repr_html_()
