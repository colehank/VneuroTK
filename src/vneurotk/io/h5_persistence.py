"""HDF5 persistence for BaseData.

All knowledge of the on-disk schema lives here.  Both the write path
(``save_recording``) and the read path (``load_recording``) are co-located so
that schema changes require edits in exactly one file.

``BaseData.save()`` and ``io.loader._load_from_h5()`` are thin delegators
that call these two functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vneurotk.io._image_codec import _encode_image
from vneurotk.io.loader import LazyNeuroLoader

if TYPE_CHECKING:
    from vneurotk.core.recording import BaseData


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


def _is_sparse(arr: np.ndarray) -> bool:
    """Return True when *arr* is worth storing in COO sparse format.

    Samples up to 100 000 elements at random (seed fixed for reproducibility)
    and returns True when more than half of them are zero.  Only meaningful
    for 3-D arrays (epochs layout); callers should skip 2-D inputs.

    Parameters
    ----------
    arr : np.ndarray
        Array to test.  Must be 3-D.

    Returns
    -------
    bool
    """
    flat = arr.ravel()
    n_sample = min(100_000, flat.size)
    idx = np.random.default_rng(seed=0).integers(0, flat.size, size=n_sample)
    return bool((flat[idx] == 0).mean() > 0.5)


def save_recording(bd: BaseData, fpath: Path) -> None:
    """Serialize a configured :class:`BaseData` to an HDF5 file.

    Parameters
    ----------
    bd : BaseData
        Configured recording.  Must have ``bd.configured == True``.
    fpath : Path
        Destination file path.  Parent directories are created if needed.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(fpath, "w") as f:
        _write_neuro(f, bd)
        _write_stim_labels(f, bd)
        _write_trial_arrays(f, bd)
        _write_neuro_info(f, bd)
        _write_vision_info(f, bd)
        _write_trial_info(f, bd)
        _write_trial_meta(f, bd)
        _write_vision_store(f, bd)
        _write_stimuli_db(f, bd)

    logger.info("Saved BaseData to {}", fpath)


def _write_neuro(f: h5py.File, bd: BaseData) -> None:
    neuro_arr = bd.neuro.data
    use_coo = neuro_arr.ndim == 3 and _is_sparse(neuro_arr)

    if use_coo:
        from scipy.sparse import coo_matrix

        flat = neuro_arr.reshape(-1, neuro_arr.shape[-1])
        sparse = coo_matrix(flat)
        f.create_dataset("neuro_row", data=sparse.row)
        f.create_dataset("neuro_col", data=sparse.col)
        f.create_dataset("neuro_data", data=sparse.data)
        f.attrs["neuro_format"] = "coo"
        f.attrs["neuro_shape"] = list(neuro_arr.shape)
        f.attrs["neuro_dtype"] = str(neuro_arr.dtype)
    else:
        f.create_dataset("neuro", data=neuro_arr)
        f.attrs["neuro_format"] = "dense"


def _write_stim_labels(f: h5py.File, bd: BaseData) -> None:
    assert bd.stim_labels is not None
    sl_flat = bd.stim_labels.ravel()
    if sl_flat.dtype == object:
        sl_arr = np.array(
            [v if isinstance(v, str) else "" for v in sl_flat],
            dtype=h5py.string_dtype(),
        )
        f.create_dataset("stim_labels", data=sl_arr)
        f.attrs["stim_labels_is_str"] = True
    else:
        f.create_dataset("stim_labels", data=sl_flat)
        f.attrs["stim_labels_is_str"] = False
    f.attrs["stim_labels_shape"] = list(bd.stim_labels.shape)


def _write_trial_arrays(f: h5py.File, bd: BaseData) -> None:
    f.create_dataset("trial", data=bd.trial)
    f.create_dataset("trial_starts", data=bd.trial_starts)
    f.create_dataset("trial_ends", data=bd.trial_ends)
    f.create_dataset("vision_onsets", data=bd.vision_onsets)
    f.attrs["data_mode"] = bd.data_mode


def _write_neuro_info(f: h5py.File, bd: BaseData) -> None:
    ni = f.create_group("neuro_info")
    for k, v in bd.neuro_info.items():
        if v is None:
            continue
        if isinstance(v, list):
            ni.attrs[k] = np.array(v, dtype=h5py.string_dtype()) if all(isinstance(x, str) for x in v) else np.array(v)
        else:
            ni.attrs[k] = v


def _write_vision_info(f: h5py.File, bd: BaseData) -> None:
    assert bd.vision_info is not None
    vi = f.create_group("vision_info")
    vi.attrs["n_stim"] = bd.vision_info["n_stim"]
    stim_ids = bd.vision_info["stim_ids"]
    if stim_ids and isinstance(stim_ids[0], str):
        vi.create_dataset("stim_ids", data=np.array(stim_ids, dtype=h5py.string_dtype()))
    else:
        vi.create_dataset("stim_ids", data=np.array(stim_ids))


def _write_trial_info(f: h5py.File, bd: BaseData) -> None:
    assert bd.trial_info is not None
    ti = f.create_group("trial_info")
    ti.attrs["baseline"] = np.array(bd.trial_info["baseline"])
    ti.attrs["trial_window"] = np.array(bd.trial_info["trial_window"])


def _write_trial_meta(f: h5py.File, bd: BaseData) -> None:
    if bd.trial_meta is None:
        return
    tm = f.create_group("trial_meta")
    for col in bd.trial_meta.columns:
        vals = bd.trial_meta[col].to_numpy()
        if np.issubdtype(vals.dtype, np.number) or vals.dtype == np.bool_:
            tm.create_dataset(col, data=vals)
        else:
            tm.create_dataset(col, data=[str(v) for v in vals], dtype=h5py.string_dtype())


def _write_vision_store(f: h5py.File, bd: BaseData) -> None:
    if bd._vision_data is not None and bd._vision_data.has_visual_representations:
        bd._vision_data.dump(f)


def _write_stimuli_db(f: h5py.File, bd: BaseData) -> None:
    db = bd._vision_data.db if bd._vision_data is not None else None
    if db is None:
        return
    grp = f.create_group("stimuli_db")
    for stim_id, img in db.items():
        ds_key = str(stim_id)
        data, kind = _encode_image(img)
        grp.create_dataset(ds_key, data=data)
        grp[ds_key].attrs["kind"] = kind
        grp[ds_key].attrs["key_type"] = type(stim_id).__name__


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def _make_coo_loader(fpath: Path, shape: tuple, dtype: str):
    """Return a zero-argument callable that reconstructs a COO-sparse neuro array.

    Parameters
    ----------
    fpath : Path
        HDF5 file containing ``neuro_row``, ``neuro_col``, ``neuro_data``.
    shape : tuple
        Original 3-D shape ``(n_trials, n_timebins, n_chan)``.
    dtype : str
        NumPy dtype string stored in ``neuro_dtype`` attr.

    Returns
    -------
    Callable[[], np.ndarray]
    """

    def _loader(_fpath=fpath, _shape=shape, _dtype=dtype) -> np.ndarray:
        from scipy.sparse import coo_matrix as _coo

        logger.info("Lazy-loading COO data from {}", _fpath)
        with h5py.File(_fpath, "r") as fh:
            row = fh["neuro_row"][:]
            col = fh["neuro_col"][:]
            data = fh["neuro_data"][:]
        flat_shape = (_shape[0] * _shape[1], _shape[2])
        sparse = _coo((data, (row, col)), shape=flat_shape, dtype=_dtype)
        return sparse.toarray().reshape(_shape)

    return _loader


def _make_dense_loader(fpath: Path):
    """Return a zero-argument callable that loads the dense neuro dataset.

    Parameters
    ----------
    fpath : Path
        HDF5 file containing a ``neuro`` dataset.

    Returns
    -------
    Callable[[], np.ndarray]
    """

    def _loader(_fpath=fpath) -> np.ndarray:
        logger.info("Lazy-loading dense neuro from {}", _fpath)
        with h5py.File(_fpath, "r") as fh:
            return fh["neuro"][:]

    return _loader


def load_recording(fpath: Path) -> BaseData:
    """Deserialize a :class:`BaseData` from an HDF5 file.

    Parameters
    ----------
    fpath : Path
        Path to an HDF5 file written by :func:`save_recording`.

    Returns
    -------
    BaseData
        Neuro data is lazy-loaded (populated on first access to ``.neuro``).
    """
    from vneurotk.core.recording import BaseData

    if not fpath.exists():
        raise FileNotFoundError(f"H5 file not found: {fpath}")

    logger.info("Loading VTK data from {}", fpath)

    with h5py.File(fpath, "r") as f:
        neuro, neuro_shape, neuro_loader = _read_neuro(f, fpath)
        neuro_info = _read_neuro_info(f, neuro_shape)
        stim_labels = _read_stim_labels(f)
        vision_info = _read_vision_info(f)
        trial, trial_starts, trial_ends, vision_onsets = _read_trial_arrays(f)
        trial_info = _read_trial_info(f)
        trial_meta = _read_trial_meta(f)
        _has_vision_store = "vision_store" in f
        _has_stimuli_db = "stimuli_db" in f
        data_mode = str(f.attrs.get("data_mode", "continuous")) or "continuous"
        if data_mode == "continues":
            data_mode = "continuous"

    logger.info("Loaded VTK data (lazy): neuro shape {}", neuro_shape)

    bd = BaseData(
        neuro=neuro,
        neuro_info=neuro_info,
        stim_labels=stim_labels,
        vision_info=vision_info,
        trial=trial,
        trial_info=trial_info,
        trial_starts=trial_starts,
        trial_ends=trial_ends,
        vision_onsets=vision_onsets,
        trial_meta=trial_meta,
        data_mode=data_mode,
    )
    if neuro_loader is not None:
        bd.set_neuro_loader(neuro_loader)
    _read_vision_data(fpath, bd, _has_vision_store, _has_stimuli_db)
    return bd


def _read_neuro(f: h5py.File, fpath: Path) -> tuple[np.ndarray | None, tuple, Any]:
    """Read neuro data format and build a lazy loader if needed.

    Returns
    -------
    tuple[ndarray | None, tuple, callable | None]
        ``(neuro, neuro_shape, neuro_loader)`` — neuro is None when lazy.
    """
    neuro_format = str(f.attrs.get("neuro_format", "dense"))

    if neuro_format == "coo":
        neuro_shape = tuple(int(x) for x in f.attrs["neuro_shape"])
        neuro_dtype = str(f.attrs["neuro_dtype"])
        return None, neuro_shape, LazyNeuroLoader(_make_coo_loader(fpath, neuro_shape, neuro_dtype))

    neuro_shape = tuple(f["neuro"].shape)
    return None, neuro_shape, LazyNeuroLoader(_make_dense_loader(fpath))


def _read_trial_arrays(
    f: h5py.File,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Read trial, trial_starts, trial_ends, vision_onsets arrays.

    Symmetric with :func:`_write_trial_arrays`.

    Returns
    -------
    tuple
        ``(trial, trial_starts, trial_ends, vision_onsets)`` — each is None if absent.
    """
    trial = f["trial"][:] if "trial" in f else None
    trial_starts = f["trial_starts"][:] if "trial_starts" in f else None
    trial_ends = f["trial_ends"][:] if "trial_ends" in f else None
    vision_onsets = f["vision_onsets"][:] if "vision_onsets" in f else None
    return trial, trial_starts, trial_ends, vision_onsets


def _read_vision_data(
    fpath: Path,
    bd: BaseData,
    has_vision_store: bool,
    has_stimuli_db: bool,
) -> None:
    """Reconstruct and attach VisionData to *bd* if the file contains vision data.

    Symmetric with :func:`_write_vision_store` + :func:`_write_stimuli_db`.
    """
    if not has_vision_store and not has_stimuli_db:
        return

    from vneurotk.io.loader import LazyH5Dict
    from vneurotk.vision.data import VisionData

    loaded_stimuli = LazyH5Dict(fpath, "stimuli_db") if has_stimuli_db else None
    with h5py.File(fpath, "r") as f:
        store = VisionData.from_h5(
            f,
            output_order=bd.trial_stim_ids if bd.configured else np.array([]),
            vision_db=loaded_stimuli,
            fpath=fpath,
        )
    if store.has_visual_representations or store.db is not None:
        bd._restore_vision_data(store)


def _decode_attr(value: Any) -> Any:
    """Convert an HDF5 attribute value to a Python native type.

    Parameters
    ----------
    value : Any
        Raw value read from ``h5py.AttributeManager``.

    Returns
    -------
    Any
        ``list`` for ``np.ndarray``, ``int`` for ``np.integer``,
        ``float`` for ``np.floating``, original value otherwise.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _read_neuro_info(f: h5py.File, neuro_shape: tuple) -> dict:
    neuro_info: dict = {}
    if "neuro_info" in f:
        for key in f["neuro_info"].attrs:
            neuro_info[key] = _decode_attr(f["neuro_info"].attrs[key])
    if "shape" not in neuro_info:
        neuro_info["shape"] = list(neuro_shape)
    return neuro_info


def _read_stim_labels(f: h5py.File) -> np.ndarray | None:
    if "stim_labels" not in f:
        return None
    vis_is_str = bool(f.attrs.get("stim_labels_is_str", False))
    vis_shape = f.attrs.get("stim_labels_shape", None)
    raw_vis = f["stim_labels"][:]
    if vis_is_str:
        vision = np.empty(len(raw_vis), dtype=object)
        for i, v in enumerate(raw_vis):
            s = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            vision[i] = np.nan if s == "" else s
    else:
        vision = raw_vis
    if vis_shape is not None:
        vision = vision.reshape([int(x) for x in vis_shape])
    return vision


def _read_vision_info(f: h5py.File) -> dict:
    vision_info: dict = {}
    if "vision_info" not in f:
        return vision_info
    vi_grp = f["vision_info"]
    for key in vi_grp.attrs:
        vision_info[key] = _decode_attr(vi_grp.attrs[key])
    if "stim_ids" in vi_grp:
        vision_info["stim_ids"] = vi_grp["stim_ids"][:].tolist()
    return vision_info


def _read_trial_info(f: h5py.File) -> dict:
    trial_info: dict = {}
    if "trial_info" not in f:
        return trial_info
    for key in f["trial_info"].attrs:
        trial_info[key] = _decode_attr(f["trial_info"].attrs[key])
    return trial_info


def _read_trial_meta(f: h5py.File) -> pd.DataFrame | None:
    if "trial_meta" not in f:
        return None
    cols: dict = {}
    for col_name in f["trial_meta"]:
        vals = f["trial_meta"][col_name][:]
        if vals.dtype.kind in ("S", "O"):
            vals = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in vals])
        cols[col_name] = vals
    return pd.DataFrame(cols)
