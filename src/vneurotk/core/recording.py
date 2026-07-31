"""Joint Data Object — the top-level container for VneuroTK.

This module provides :class:`BaseData`, which couples a neural Recording
(time-series, trial structure) with a Stimulus Set and optional Visual
Representations.  Neither a purely neural concept nor a visual one — it is
the entity that links both domains together for a single experiment unit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from loguru import logger

from vneurotk.core.info import Info
from vneurotk.core.metadata import NeuroInfo, TrialInfo, VisionInfo
from vneurotk.core.stimulus import StimulusSet, _coerce_scalar_array
from vneurotk.neuro.base import NeuroData
from vneurotk.neuro.trial import (
    TrialStructure,
    build_trial_structure_continuous,
    build_trial_structure_epochs,
    validate_trial_structure_state,
)

NeuroLoader = Callable[[], np.ndarray]  # lazy loader contract
DataMode: TypeAlias = Literal["continuous", "epochs", "patterns"]
_VALID_DATA_MODES = frozenset(("continuous", "epochs", "patterns"))


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
        Mutable dictionary compatible with :class:`NeuroInfo`. ``sfreq`` is
        required for time-based operations. Optional keys include ``ch_names``,
        ``highpass``, ``lowpass``, ``source_file``, and ``shape``.
    stim_labels : np.ndarray | None
        Internal stimulus-label array of shape ``(ntime,)`` or
        ``(n_trials, n_timebins)``.  ``np.nan`` at non-stimulus timepoints,
        stimulus ID at onset timepoints.  Not exposed directly; use
        :attr:`trial_stim_ids`.
    vision_info : dict | None
        Mutable dictionary compatible with :class:`VisionInfo`; commonly
        contains ``n_stim`` and ``stim_ids``.
    trial : np.ndarray | None
        Trial-ID array of shape ``(ntime,)``.  ``np.nan`` outside trials.
    trial_info : dict | None
        Mutable dictionary compatible with :class:`TrialInfo`; commonly
        contains ``baseline`` and ``trial_window``.
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
        data_mode: DataMode | None = None,
    ) -> None:
        self._neuro: np.ndarray | None = np.asarray(neuro) if neuro is not None else None
        self._neuro_loader: NeuroLoader | None = None
        self.neuro_info: NeuroInfo = cast(NeuroInfo, neuro_info)

        self._stim_labels = stim_labels
        self.vision_info: VisionInfo | None = cast(VisionInfo | None, vision_info)
        self.trial = trial
        self.trial_info: TrialInfo | None = cast(TrialInfo | None, trial_info)
        self.trial_starts = trial_starts
        self.trial_ends = trial_ends
        self.vision_onsets = vision_onsets
        self.trial_meta = trial_meta

        self.data_mode: DataMode = self._infer_data_mode(self._neuro, data_mode)
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

    @classmethod
    def for_patterns(
        cls,
        neuro: np.ndarray | None = None,
        neuro_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BaseData:
        """Factory for row-level response patterns.

        Pattern rows are valid without trial arrays.  Pass ``trial_meta`` with a
        ``stim_index`` column when rows have explicit stimulus identities and
        should align to vision features.
        """
        return cls(neuro=neuro, neuro_info=neuro_info or {}, data_mode="patterns", **kwargs)

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
            raise RuntimeError("BaseData is incomplete for its data_mode. Check .is_configured.")
        if self.data_mode == "patterns" and self._pattern_stim_ids() is None:
            raise RuntimeError(
                "Vision alignment for patterns requires explicit row stimulus IDs in trial_meta['stim_index']."
            )
        if self._vision_data is None:
            try:
                from vneurotk.vision.data import VisionData
            except ImportError as e:
                raise RuntimeError(
                    "Vision features require torch and transformers. Install them with: uv add 'vneurotk[vision]'"
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
        """Whether the state required by the active data mode is complete.

        Continuous and epochs recordings require the full trial structure.
        Patterns are already row-level responses, so a two-dimensional neuro
        array (or declared lazy shape) is sufficient; row stimulus IDs are
        optional and only enable vision alignment.
        """
        try:
            self._validate_state()
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

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
        """Number of trials (patterns have rows rather than trials)."""
        if self.data_mode == "patterns":
            return 0
        if self.trial_starts is None:
            return 0
        return len(self.trial_starts)

    def _pattern_stim_ids(self) -> np.ndarray | None:
        """Return explicit row stimulus IDs, if pattern metadata provides them."""
        if self.data_mode != "patterns" or self.trial_meta is None:
            return None
        columns = getattr(self.trial_meta, "columns", ())
        if "stim_index" not in columns:
            return None
        stim_ids = np.asarray(self.trial_meta["stim_index"])
        n_rows = self._neuro_shape_dim(0)
        if stim_ids.ndim != 1 or len(stim_ids) != n_rows:
            return None
        return stim_ids

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
        """Stimulus IDs aligned to trial or pattern rows.

        Pattern data exposes IDs only when row metadata explicitly contains a
        ``stim_index`` column.  Row position alone never implies stimulus
        identity.

        Raises
        ------
        RuntimeError
            If the active mode is incomplete, or patterns have no explicit row
            stimulus IDs.
        """
        if not self.configured:
            raise RuntimeError("BaseData is incomplete for its data_mode.")
        if self.data_mode == "patterns":
            stim_ids = self._pattern_stim_ids()
            if stim_ids is None:
                raise RuntimeError(
                    "Pattern rows have no explicit stimulus IDs. "
                    "Provide trial_meta with a 'stim_index' column for vision alignment."
                )
            return stim_ids.copy()
        values = [self._stim_id_at_trial(i) for i in range(self.n_trials)]
        labels = self._stim_labels
        if labels is not None and labels.dtype != object and self.vision_info is not None:
            known_ids = self.vision_info.get("stim_ids", [])
            values = [next((known for known in known_ids if known == value), value) for value in values]
        return _coerce_scalar_array(values)

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

        visual_ids = _coerce_scalar_array(stim_ids)

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
                visual_ids,
                trial_window,
                vision_onsets,
                self.ntime,
                self._sampling_frequency(),
                neuro_shape=self._neuro.shape if self._neuro is not None else tuple(self.neuro_info.get("shape", ())),
            )
        staged_db = StimulusSet(visual_ids, vision_db) if vision_db is not None else None
        if self._vision_data is not None:
            self._vision_data._validate_output_order(visual_ids)
            self._vision_data.output_order = visual_ids
        self._apply_trial_structure(ts)

        if staged_db is not None:
            if self.vision.db is not None:
                logger.info("configure: replacing existing Stimulus Set with newly provided one.")
            self.vision.attach_db(staged_db)

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
        try:
            from vneurotk.viz.data import plot_data
        except ImportError as exc:
            if exc.name is not None and exc.name.startswith("matplotlib"):
                raise ImportError("Plotting requires matplotlib. Install it with: uv add 'vneurotk[viz]'") from exc
            raise

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
            sfreq=self._sampling_frequency(),
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

    def save(
        self,
        path: Any,
        *,
        compression: str | None = "gzip",
        compression_opts: Any = 4,
        chunk_target_bytes: int = 1024 * 1024,
    ) -> None:
        """Persist the configured data to an HDF5 file.

        Parameters
        ----------
        path : VTKPath | pathlib.Path | str
            Destination file path.
        compression : str or None
            HDF5 filter for neural and activation arrays. Defaults to ``"gzip"``;
            pass ``None`` to disable compression.
        compression_opts : Any
            Filter-specific options; defaults to gzip level 4.
        chunk_target_bytes : int
            Approximate maximum chunk size for large numerical arrays. Defaults
            to 1 MiB and preserves lazy dataset loading.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        self._validate_state()

        from vneurotk.io.h5_persistence import save_recording

        save_recording(
            self,
            self._resolve_path(path),
            compression=compression,
            compression_opts=compression_opts,
            chunk_target_bytes=chunk_target_bytes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sampling_frequency(self) -> float | int:
        """Return the sampling-frequency value for downstream validation."""
        sfreq = self.neuro_info.get("sfreq")
        if sfreq is None:
            raise ValueError("neuro_info['sfreq'] is required for this operation.")
        return sfreq

    def _validate_state(self) -> None:
        """Raise when mutable state violates the active data-mode contract."""
        raw_shape = self._neuro.shape if self._neuro is not None else self.neuro_info.get("shape")
        if raw_shape is None:
            raise RuntimeError("Cannot save incomplete BaseData state: neural shape is unavailable.")
        try:
            shape = tuple(int(dim) for dim in raw_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid neural shape {raw_shape!r}.") from exc
        if self._neuro is not None and self.neuro_info.get("shape") is not None:
            declared = tuple(int(dim) for dim in self.neuro_info["shape"])
            if declared != shape:
                raise ValueError(f"neuro_info['shape'] {declared!r} contradicts actual neural data shape {shape!r}.")
        if len(shape) == 0 or any(dim <= 0 for dim in shape):
            raise ValueError(f"Neural data must have positive dimensions, got shape {shape!r}.")
        ch_names = self.neuro_info.get("ch_names")
        if ch_names is not None and len(ch_names) != shape[-1]:
            raise ValueError(
                f"neuro_info['ch_names'] length must match channel count {shape[-1]}, got {len(ch_names)}."
            )

        if self.data_mode == "patterns":
            if len(shape) != 2:
                raise ValueError(f"patterns neuro data must be 2-D, got shape {shape}.")
            if self.trial_meta is not None and len(self.trial_meta) != shape[0]:
                raise ValueError(
                    f"trial_meta length must match pattern row count {shape[0]}, got {len(self.trial_meta)}."
                )
            return

        fields = {
            "stim_labels": self._stim_labels,
            "trial": self.trial,
            "trial_starts": self.trial_starts,
            "trial_ends": self.trial_ends,
            "vision_onsets": self.vision_onsets,
            "vision_info": self.vision_info,
            "trial_info": self.trial_info,
        }
        missing = [name for name, value in fields.items() if value is None]
        if missing:
            raise RuntimeError(
                "Cannot save incomplete BaseData state; call configure() to provide missing required field(s): "
                + ", ".join(missing)
                + "."
            )
        validate_trial_structure_state(
            data_mode=self.data_mode,
            neuro_shape=shape,
            stim_labels=cast(np.ndarray, self._stim_labels),
            trial=cast(np.ndarray, self.trial),
            trial_starts=cast(np.ndarray, self.trial_starts),
            trial_ends=cast(np.ndarray, self.trial_ends),
            vision_onsets=cast(np.ndarray, self.vision_onsets),
            vision_info=cast(VisionInfo, self.vision_info),
            trial_info=cast(TrialInfo, self.trial_info),
            trial_meta=self.trial_meta,
        )

    @staticmethod
    def _infer_data_mode(neuro: np.ndarray | None, explicit: DataMode | None) -> DataMode:
        if explicit is not None:
            if explicit not in _VALID_DATA_MODES:
                choices = ", ".join(sorted(_VALID_DATA_MODES))
                raise ValueError(f"Invalid data_mode {explicit!r}; expected one of: {choices}.")
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
