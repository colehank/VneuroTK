"""Data reading functions for VneuroTK."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vneurotk.core.recording import BaseData
from vneurotk.io._image_codec import _decode_image
from vneurotk.io.path import BIDSPath, EphysPath, MNEPath, VTKPath
from vneurotk.neuro.trial import _build_vision_info

_META_DTYPES = frozenset(
    [
        "TrialRecord",
        "ChTrialRecord",
        "UnitProp",
        "ChProp",
    ]
)


class LazyH5Dict(Mapping):
    """HDF5-backed read-only image dict that loads arrays on demand.

    The key index is built on first access (one lightweight pass over
    attribute metadata only).  Each ``__getitem__`` call opens the file,
    reads a single dataset, and closes it immediately.

    Parameters
    ----------
    path : Path or str
        Path to the HDF5 file.
    group : str
        HDF5 group name that contains the image datasets.
        Default ``"stimuli_db"``.
    """

    def __init__(self, path: Path | str, group: str = "stimuli_db") -> None:
        self._path = Path(path)
        self._group = group
        self._index: dict[Any, str] | None = None  # native_key → ds_key

    # ------------------------------------------------------------------
    # Index build (reads only attrs, not image data)
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        if self._index is not None:
            return
        idx: dict[Any, str] = {}
        with h5py.File(self._path, "r") as f:
            grp = f[self._group]
            for ds_key in grp:
                key_type = str(grp[ds_key].attrs.get("key_type", "str"))
                native: Any = ds_key
                if key_type == "int":
                    try:
                        native = int(ds_key)
                    except ValueError:
                        pass
                elif key_type == "float":
                    try:
                        native = float(ds_key)
                    except ValueError:
                        pass
                idx[native] = ds_key
        self._index = idx

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> np.ndarray:
        self._build_index()
        native = key.item() if hasattr(key, "item") else key
        ds_key = self._index[native]  # ty: ignore[not-subscriptable]
        with h5py.File(self._path, "r") as f:
            return self._decode_item(f[self._group][ds_key])

    @staticmethod
    def _decode_item(ds: Any) -> np.ndarray:
        """Decode a stored dataset to a numpy array based on its ``kind`` attribute.

        Parameters
        ----------
        ds : h5py.Dataset
            An open dataset from the stimuli_db group.

        Returns
        -------
        np.ndarray
        """
        kind = str(ds.attrs.get("kind", "array"))
        data = ds[()] if kind in ("path", "image_bytes") else ds[:]
        return _decode_image(data, kind)

    def __len__(self) -> int:
        self._build_index()
        return len(self._index)  # ty: ignore[invalid-argument-type]

    def __iter__(self):
        self._build_index()
        return iter(self._index)  # ty: ignore[no-matching-overload]

    def __repr__(self) -> str:
        self._build_index()
        return f"LazyH5Dict(n={len(self._index)}, path={self._path.name!r})"  # ty: ignore[invalid-argument-type]


_MISSING: object = object()


class LazyNeuroLoader:
    """Deferred loader for a neuro array — invokes *loader_fn* on first call.

    Wraps any ``Callable[[], np.ndarray]`` and guarantees the underlying
    function is executed at most once.  The result is cached so subsequent
    calls return the same array without re-reading from disk.

    Satisfies the ``NeuroLoader = Callable[[], np.ndarray]`` contract, so it
    can be passed directly to :meth:`~vneurotk.neuro.base.BaseData.set_neuro_loader`.

    Parameters
    ----------
    loader_fn : Callable[[], np.ndarray]
        Zero-argument function that reads and returns the neuro array.

    Examples
    --------
    >>> loader = LazyNeuroLoader(lambda: np.load("data.npy"))
    >>> bd.set_neuro_loader(loader)  # called at most once on first bd.neuro access
    """

    def __init__(self, loader_fn: Callable[[], np.ndarray]) -> None:
        self._loader_fn: Callable[[], np.ndarray] | None = loader_fn
        self._data: object = _MISSING

    def __call__(self) -> np.ndarray:
        if self._data is _MISSING:
            self._data = self._loader_fn()  # ty: ignore[call-non-callable]
            self._loader_fn = None  # release closure captures
        return self._data  # ty: ignore[invalid-return-type]

    @property
    def is_loaded(self) -> bool:
        """``True`` after the first call has materialised the array."""
        return self._data is not _MISSING

    def __repr__(self) -> str:
        return f"LazyNeuroLoader(loaded={self.is_loaded})"


def _coo_to_dense(
    fpath: Path,
    original_shape: tuple,
    stored_dtype: str,
    level: str,
) -> np.ndarray:
    """Load a COO-sparse raster from HDF5 and convert to a dense array.

    Parameters
    ----------
    fpath : Path
        HDF5 file containing ``row``, ``col``, ``data`` datasets.
    original_shape : tuple
        Shape before the axis-transpose (from ``original_shape`` attr).
    stored_dtype : str
        NumPy dtype string stored in ``dtype`` attr.
    level : str
        ``"unit"`` → transpose ``(1, 2, 0)``; ``"channel"`` → ``(0, 2, 1)``.

    Returns
    -------
    np.ndarray
        Dense array in ``(n_trials, n_timebins, n_units_or_channels)`` order.
    """
    from scipy.sparse import coo_matrix

    logger.info("Loading COO sparse data from {}", fpath)
    with h5py.File(fpath, "r") as f:
        row = f["row"][:]
        col = f["col"][:]
        data = f["data"][:]
    flat_shape = (original_shape[0] * original_shape[1], original_shape[2])
    sparse = coo_matrix((data, (row, col)), shape=flat_shape, dtype=stored_dtype)
    del row, col, data
    dense = sparse.toarray().reshape(original_shape)
    del sparse
    if level == "unit":
        return dense.transpose(1, 2, 0)
    return dense.transpose(0, 2, 1)


def read(
    path: VTKPath | EphysPath | MNEPath | BIDSPath | Path | str,
    pre_load: bool = False,
) -> BaseData:
    """Read data from various sources into BaseData.

    Parameters
    ----------
    path : VTKPath, EphysPath, MNEPath, BIDSPath, Path, or str
        Data source.  Plain ``Path`` / ``str`` is treated as a direct
        file path (e.g. an ``.h5`` file saved by :meth:`BaseData.save`).
    pre_load : bool
        If ``True``, eagerly load neuro data into memory before returning
        (calls :meth:`BaseData.load` internally).
        If ``False`` (default), data is loaded lazily on first access to
        :attr:`BaseData.neuro` — call :meth:`BaseData.load` explicitly to
        trigger loading at a chosen point.  For data types that carry no
        lazy loader (already eager), this flag is a no-op.

    Returns
    -------
    BaseData
        Data as BaseData object.

    Raises
    ------
    NotImplementedError
        If loading AvgPsth (not yet implemented).
    ValueError
        If path type is unknown or file format unsupported.
    FileNotFoundError
        If the specified file does not exist.
    """
    # Resolve to a pathlib.Path
    if isinstance(path, (str, Path)):
        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"File not found: {fpath}")
        bd = _load_from_h5(fpath)
        return bd.load() if pre_load else bd

    if isinstance(path, VTKPath):
        return path.load(pre_load=pre_load)

    raise ValueError(f"Unknown path type: {type(path)}")


# ======================================================================
# Ephys loading
# ======================================================================


def _load_from_ephys(path: EphysPath) -> BaseData:
    """Dispatch ephys loading by dtype."""
    dtype = path.dtype
    if dtype is None:
        raise ValueError("EphysPath.dtype must be set for loading")

    if dtype in _META_DTYPES:
        raise ValueError(f"'{dtype}' is a metadata file. Pass a neuro dtype (TrialRaster, MeanFr, etc.) instead.")

    if dtype == "TrialRaster":
        return _load_ephys_raster(path, level="unit")
    elif dtype == "ChTrialRaster":
        return _load_ephys_raster(path, level="channel")
    elif dtype == "MeanFr":
        return _load_ephys_mean_fr(path, level="unit")
    elif dtype == "ChMeanFr":
        return _load_ephys_mean_fr(path, level="channel")
    elif dtype == "ChStimFr":
        return _load_ephys_stim_fr(path)
    elif dtype == "AvgPsth":
        raise NotImplementedError("AvgPsth loading not yet implemented")
    else:
        raise ValueError(f"Unsupported ephys dtype for loading: {dtype}")


# ── Ephys companion-file helpers ──────────────────────────────────────────────


def _ephys_level_dtypes(level: str) -> tuple[str, str]:
    """Return (record_dtype, prop_dtype) for a given recording level."""
    if level == "unit":
        return "TrialRecord", "UnitProp"
    return "ChTrialRecord", "ChProp"


def _load_ephys_csv(path: EphysPath, dtype: str) -> pd.DataFrame:
    """Construct a sibling EphysPath with *dtype*, check existence, and load CSV."""
    csv_path = EphysPath(
        root=path.root,
        session_id=path.session_id,
        dtype=dtype,
        probe=path.probe,
        extension="csv",
    )
    if not csv_path.fpath.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path.fpath}")
    return pd.read_csv(csv_path.fpath)


def _load_ephys_prop(path: EphysPath, prop_dtype: str) -> tuple[pd.DataFrame, list[str]]:
    """Load ChProp / UnitProp CSV and return (prop_df, ch_names)."""
    prop_df = _load_ephys_csv(path, prop_dtype)
    return prop_df, prop_df["id"].astype(str).tolist()


def _load_ephys_record(path: EphysPath, record_dtype: str) -> pd.DataFrame:
    """Load TrialRecord / ChTrialRecord CSV."""
    return _load_ephys_csv(path, record_dtype)


def _build_stim_fr_vision_info(
    n_stim: int,
    allstim: np.ndarray | None,
    teststim: np.ndarray | None,
) -> dict:
    """Build the vision_info dict for a ChStimFr (stimulus-level) BaseData.

    Parameters
    ----------
    n_stim : int
        Total number of stimuli (``neuro.shape[0]``).
    allstim : np.ndarray or None
        All stimulus IDs stored in the HDF5 ``allstim`` dataset, or ``None``
        when the dataset is absent.
    teststim : np.ndarray or None
        Test-set stimulus IDs from ``teststim``, or ``None`` when absent.

    Returns
    -------
    dict
        ``{"n_stim": n_stim}`` plus optional ``"stim_ids"`` and ``"teststim"``
        keys when the corresponding arrays are provided.
    """
    info: dict = {"n_stim": n_stim}
    if allstim is not None:
        info["stim_ids"] = allstim.tolist()
    if teststim is not None:
        info["teststim"] = teststim.tolist()
    return info


# ── Ephys dtype loaders ───────────────────────────────────────────────────────


def _load_ephys_raster(path: EphysPath, level: str) -> BaseData:
    """Load TrialRaster / ChTrialRaster into an epochs-mode BaseData.

    The raster h5 is NOT read eagerly — a lazy loader is attached so
    the actual COO → dense conversion happens on first ``bd.neuro`` access.
    """
    record_dtype, prop_dtype = _ephys_level_dtypes(level)

    raster_fpath = path.fpath
    if not raster_fpath.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_fpath}")

    with h5py.File(raster_fpath, "r") as f:
        original_shape = tuple(int(x) for x in f.attrs["original_shape"])
        stored_dtype = str(f.attrs["dtype"])
        meta = dict(f["metadata"].attrs)

    if level == "unit":
        # (n_units, n_trials, n_timebins) → (n_trials, n_timebins, n_units)
        target_shape = (original_shape[1], original_shape[2], original_shape[0])
    else:
        # (n_trials, n_channels, n_timebins) → (n_trials, n_timebins, n_channels)
        target_shape = (original_shape[0], original_shape[2], original_shape[1])

    n_trials, n_timebins, n_chan = target_shape

    record_df = _load_ephys_record(path, record_dtype)
    _, ch_names = _load_ephys_prop(path, prop_dtype)

    visual_ids = record_df["stim_index"].values
    pre_onset = int(meta.get("pre_onset", meta.get("pre_stim_ms", 50)))
    post_onset = n_timebins - pre_onset
    sfreq = int(meta.get("sampling_rate", 1000))

    unique_ids = np.unique(visual_ids)
    is_str_ids = unique_ids.dtype.kind in ("U", "S", "O")
    if is_str_ids:
        visual = np.empty((n_trials, n_timebins), dtype=object)
        visual[:] = np.nan
    else:
        visual = np.full((n_trials, n_timebins), np.nan)
    for i, sid in enumerate(visual_ids):
        visual[i, pre_onset] = sid

    trial = np.empty((n_trials, n_timebins))
    for i in range(n_trials):
        trial[i, :] = i

    neuro_info = {
        "sfreq": sfreq,
        "ch_names": ch_names,
        "highpass": None,
        "lowpass": None,
        "source_file": str(path.fpath),
        "shape": target_shape,
    }

    bd = BaseData.for_epochs(
        neuro_info=neuro_info,
        stim_labels=visual,
        vision_info=_build_vision_info(unique_ids),
        trial=trial,
        trial_info={
            "baseline": [-pre_onset, 0],
            "trial_window": [-pre_onset, post_onset],
        },
        trial_starts=np.zeros(n_trials, dtype=int),
        trial_ends=np.full(n_trials, n_timebins, dtype=int),
        vision_onsets=np.full(n_trials, pre_onset, dtype=int),
        trial_meta=record_df,
    )
    bd.set_neuro_loader(LazyNeuroLoader(lambda: _coo_to_dense(raster_fpath, original_shape, stored_dtype, level)))

    logger.info(
        "Loaded ephys {} (lazy): {} trials, {} timebins, {} channels",
        path.dtype,
        n_trials,
        n_timebins,
        n_chan,
    )
    return bd


def _load_ephys_mean_fr(path: EphysPath, level: str) -> BaseData:
    """Load MeanFr / ChMeanFr into a trial-level BaseData."""
    record_dtype, prop_dtype = _ephys_level_dtypes(level)

    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"MeanFr file not found: {fpath}")

    with h5py.File(fpath, "r") as f:
        neuro = f["data"][:]

    record_df = _load_ephys_record(path, record_dtype)
    _, ch_names = _load_ephys_prop(path, prop_dtype)

    visual_ids = record_df["stim_index"].values
    unique_ids = np.unique(visual_ids)

    bd = BaseData(
        neuro=neuro,
        neuro_info={
            "sfreq": None,
            "ch_names": ch_names,
            "highpass": None,
            "lowpass": None,
            "source_file": str(path.fpath),
        },
        vision_info=_build_vision_info(unique_ids),
        trial_meta=record_df,
        data_mode="patterns",
    )

    logger.info("Loaded ephys {} (trial-level): shape {}", path.dtype, neuro.shape)
    return bd


def _load_ephys_stim_fr(path: EphysPath) -> BaseData:
    """Load ChStimFr into a stimulus-level BaseData."""
    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"ChStimFr file not found: {fpath}")

    with h5py.File(fpath, "r") as f:
        neuro = f["data"][:]
        allstim = f["allstim"][:] if "allstim" in f else None
        teststim = f["teststim"][:] if "teststim" in f else None

    _, ch_names = _load_ephys_prop(path, "ChProp")

    trial_meta = pd.DataFrame({"stim_index": allstim}) if allstim is not None else None

    bd = BaseData(
        neuro=neuro,
        neuro_info={
            "sfreq": None,
            "ch_names": ch_names,
            "highpass": None,
            "lowpass": None,
            "source_file": str(path.fpath),
        },
        vision_info=_build_stim_fr_vision_info(neuro.shape[0], allstim, teststim),
        trial_meta=trial_meta,
        data_mode="patterns",
    )

    logger.info("Loaded ephys ChStimFr (stimulus-level): shape {}", neuro.shape)
    return bd


# ======================================================================
# MNE / BIDS / H5 loading
# ======================================================================


def _build_bd_from_mne_raw(raw: Any, source_file: str) -> BaseData:
    """Build a lazy continuous :class:`BaseData` from an open MNE Raw object.

    Shared by :func:`_load_from_mne` and :func:`_load_from_bids`.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        An already-opened MNE Raw object (``preload=False``).
    source_file : str
        Absolute path string for the ``source_file`` entry in ``neuro_info``.

    Returns
    -------
    BaseData
        Lazy-loaded; ``neuro`` is not in memory until first access.
    """
    neuro_info = {
        "sfreq": raw.info["sfreq"],
        "ch_names": raw.info["ch_names"],
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "source_file": source_file,
        "shape": (len(raw.times), len(raw.ch_names)),
    }
    bd = BaseData.for_continuous(neuro_info=neuro_info)
    logger.info(
        "MNE-like metadata loaded: {} timepoints, {} channels, sfreq={} Hz",
        len(raw.times),
        len(raw.ch_names),
        raw.info["sfreq"],
    )
    return bd


def _load_from_mne(path: MNEPath) -> BaseData:
    """Load data from MNE raw file (lazy by default)."""
    try:
        import mne
    except ImportError as e:
        raise ImportError("mne is required for loading MNE data") from e

    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"MNE file not found: {fpath}")

    logger.info("Loading MNE metadata from {}", fpath)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw(fpath, preload=False, verbose=False)

    def _neuro_loader(_raw=raw) -> np.ndarray:
        logger.info("Reading MNE data into memory from {}", fpath)
        return _raw.get_data().T

    bd = _build_bd_from_mne_raw(raw, str(fpath))
    bd.set_neuro_loader(LazyNeuroLoader(_neuro_loader))
    return bd


def _load_from_bids(path: BIDSPath) -> BaseData:
    """Load data from BIDS dataset (lazy by default)."""
    try:
        from mne_bids import read_raw_bids
    except ImportError as e:
        raise ImportError("mne_bids is required for loading BIDS data") from e

    if path.bids_path is None:
        raise ValueError("BIDSPath not properly initialized")

    logger.info("Loading BIDS metadata from {}", path.fpath)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = read_raw_bids(path.bids_path, verbose=False)

    def _neuro_loader(_raw=raw) -> np.ndarray:
        logger.info("Reading BIDS data into memory from {}", path.fpath)
        _raw.load_data()
        return _raw.get_data().T

    bd = _build_bd_from_mne_raw(raw, str(path.fpath))
    bd.set_neuro_loader(LazyNeuroLoader(_neuro_loader))
    return bd


def _load_from_h5(fpath: Path) -> BaseData:
    """Load data from h5 file (saved VTK data)."""
    from vneurotk.io.h5_persistence import load_recording

    return load_recording(fpath)
