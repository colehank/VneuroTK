"""HDF5 persistence for BaseData.

All knowledge of the on-disk schema lives here.  Both the write path
(``save_recording``) and the read path (``load_recording``) are co-located so
that schema changes require edits in exactly one file.

``BaseData.save()`` and ``io.loader._load_from_h5()`` are thin delegators
that call these two functions.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vneurotk.io._h5_codec import (
    FileIdentity,
    H5StorageOptions,
    dataset_kwargs,
    decode_text,
    file_identity,
    open_file_identity,
    read_scalar_sequence,
    verify_open_file_identity,
    write_scalar_sequence,
)
from vneurotk.io._image_codec import _encode_image
from vneurotk.io.loader import LazyNeuroLoader

if TYPE_CHECKING:
    from vneurotk.core.recording import BaseData


FORMAT_MAGIC_ATTR = "vneurotk_format"
FORMAT_MAGIC = "recording"
SCHEMA_VERSION_ATTR = "vneurotk_schema_version"
WRITER_VERSION_ATTR = "writer_version"
MIN_SUPPORTED_SCHEMA_VERSION = 0
CURRENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _FormatInfo:
    """Validated root format metadata used to select a schema reader."""

    schema_version: int
    writer_version: str | None


def _package_version() -> str:
    """Return the installed writer package version."""
    try:
        return version("vneurotk")
    except PackageNotFoundError:
        return "unknown"


def _decode_text_attr(value: Any) -> str:
    """Return an HDF5 text attribute as a Python string."""
    return decode_text(value)


def _detect_format(f: h5py.File, fpath: Path) -> _FormatInfo:
    """Validate root magic/version metadata and select the schema version.

    Files with neither format field are historical schema v0 recordings.
    A partially present header is corrupt rather than legacy.
    """
    has_magic = FORMAT_MAGIC_ATTR in f.attrs
    has_version = SCHEMA_VERSION_ATTR in f.attrs
    if not has_magic and not has_version:
        return _FormatInfo(schema_version=0, writer_version=None)
    if not has_magic or not has_version:
        missing = FORMAT_MAGIC_ATTR if not has_magic else SCHEMA_VERSION_ATTR
        raise ValueError(f"Invalid VneuroTK HDF5 file {fpath}: missing required root attribute {missing!r}.")

    magic = _decode_text_attr(f.attrs[FORMAT_MAGIC_ATTR])
    if magic != FORMAT_MAGIC:
        raise ValueError(
            f"Invalid VneuroTK HDF5 file {fpath}: {FORMAT_MAGIC_ATTR} must be {FORMAT_MAGIC!r}, got {magic!r}."
        )

    raw_version = f.attrs[SCHEMA_VERSION_ATTR]
    if isinstance(raw_version, (bool, np.bool_)) or not isinstance(raw_version, (int, np.integer)):
        raise ValueError(
            f"Invalid VneuroTK HDF5 file {fpath}: {SCHEMA_VERSION_ATTR} must be an integer, got {raw_version!r}."
        )
    schema_version = int(raw_version)
    if not MIN_SUPPORTED_SCHEMA_VERSION <= schema_version <= CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported VneuroTK HDF5 schema version {schema_version} in {fpath}; "
            f"supported range is {MIN_SUPPORTED_SCHEMA_VERSION}..{CURRENT_SCHEMA_VERSION}."
        )

    writer_version = f.attrs.get(WRITER_VERSION_ATTR)
    if schema_version >= 1 and writer_version is None:
        raise ValueError(
            f"Invalid VneuroTK HDF5 file {fpath}: missing required root attribute {WRITER_VERSION_ATTR!r}."
        )
    return _FormatInfo(
        schema_version=schema_version,
        writer_version=None if writer_version is None else _decode_text_attr(writer_version),
    )


def _normalize_v0(f: h5py.File) -> dict[str, str]:
    """Translate known schema-v0 spellings to current in-memory values."""
    data_mode = _decode_text_attr(f.attrs.get("data_mode", "continuous")) or "continuous"
    if data_mode == "continues":
        data_mode = "continuous"
    return {"data_mode": data_mode}


def _normalize_v1(f: h5py.File) -> dict[str, str]:
    """Read required schema-v1 values without legacy aliases."""
    if "data_mode" not in f.attrs:
        raise ValueError("Invalid schema-1 VneuroTK HDF5 file: missing required root attribute 'data_mode'.")
    data_mode = _decode_text_attr(f.attrs["data_mode"])
    if not data_mode:
        raise ValueError("Invalid schema-1 VneuroTK HDF5 file: root attribute 'data_mode' must be nonempty.")
    return {"data_mode": data_mode}


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


def save_recording(
    bd: BaseData,
    fpath: Path,
    *,
    compression: str | None = "gzip",
    compression_opts: Any = 4,
    chunk_target_bytes: int = 1024 * 1024,
) -> None:
    """Atomically serialize a configured :class:`BaseData` to an HDF5 file.

    The complete recording is written and flushed to a temporary file in the
    destination directory, reopened for basic schema validation, and only then
    installed with :func:`os.replace`. Existing targets are therefore untouched
    by encoding, I/O, validation, or replacement failures.
    """
    fpath.parent.mkdir(parents=True, exist_ok=True)
    options = H5StorageOptions(compression, compression_opts, chunk_target_bytes)
    destination_mode = stat.S_IMODE(fpath.stat().st_mode) if fpath.exists() else None
    fd, temp_name = tempfile.mkstemp(prefix=f".{fpath.name}.", suffix=".tmp", dir=fpath.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    if destination_mode is None:
        temp_path.unlink()
        fd = os.open(temp_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666)
        os.close(fd)
    else:
        os.chmod(temp_path, destination_mode)
    try:
        with h5py.File(temp_path, "w") as f:
            f.attrs[FORMAT_MAGIC_ATTR] = FORMAT_MAGIC
            f.attrs[SCHEMA_VERSION_ATTR] = CURRENT_SCHEMA_VERSION
            f.attrs[WRITER_VERSION_ATTR] = _package_version()
            neuro_shape = _write_neuro(f, bd, options)
            _write_stim_labels(f, bd)
            _write_trial_arrays(f, bd)
            _write_neuro_info(f, bd, neuro_shape)
            _write_vision_info(f, bd)
            _write_trial_info(f, bd)
            _write_trial_meta(f, bd)
            _write_vision_store(f, bd, options)
            _write_stimuli_db(f, bd, options)
            f.flush()
        _validate_written_file(temp_path)
        os.replace(temp_path, fpath)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass

    logger.info("Saved BaseData to {}", fpath)


def _validate_written_file(fpath: Path) -> None:
    """Perform basic header and required-layout validation before replacement."""
    with h5py.File(fpath, "r") as f:
        info = _detect_format(f, fpath)
        if info.schema_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(f"Temporary HDF5 file has unexpected schema version {info.schema_version}.")
        _read_neuro(f, fpath)
        if "data_mode" not in f.attrs:
            raise ValueError("Temporary HDF5 file is missing required data_mode metadata.")


def _write_neuro(f: h5py.File, bd: BaseData, options: H5StorageOptions) -> tuple[int, ...]:
    neuro_arr = bd.neuro.data
    neuro_shape = tuple(int(dim) for dim in neuro_arr.shape)
    declared_shape = bd.neuro_info.get("shape")
    if declared_shape is not None:
        try:
            normalized_declared_shape = tuple(int(dim) for dim in declared_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid neuro_info['shape'] {declared_shape!r}; expected an integer shape.") from exc
        if normalized_declared_shape != neuro_shape:
            raise ValueError(
                f"neuro_info['shape'] {normalized_declared_shape!r} contradicts actual neural data shape "
                f"{neuro_shape!r}."
            )
    use_coo = neuro_arr.ndim == 3 and _is_sparse(neuro_arr)

    if use_coo:
        from scipy.sparse import coo_matrix

        flat = neuro_arr.reshape(-1, neuro_arr.shape[-1])
        sparse = coo_matrix(flat)
        f.create_dataset("neuro_row", data=sparse.row, **dataset_kwargs(sparse.row, options))
        f.create_dataset("neuro_col", data=sparse.col, **dataset_kwargs(sparse.col, options))
        f.create_dataset("neuro_data", data=sparse.data, **dataset_kwargs(sparse.data, options))
        f.attrs["neuro_format"] = "coo"
        f.attrs["neuro_shape"] = list(neuro_arr.shape)
        f.attrs["neuro_dtype"] = str(neuro_arr.dtype)
    else:
        f.create_dataset("neuro", data=neuro_arr, **dataset_kwargs(neuro_arr, options))
        f.attrs["neuro_format"] = "dense"
    return neuro_shape


def _write_stim_labels(f: h5py.File, bd: BaseData) -> None:
    if bd.stim_labels is None:
        return
    sl_flat = bd.stim_labels.ravel()
    if sl_flat.dtype == object:
        write_scalar_sequence(
            f,
            "stim_labels",
            sl_flat,
            allow_missing=True,
            context="object stimulus label",
        )
        f.attrs["stim_labels_encoding"] = "typed_scalars"
    else:
        f.create_dataset("stim_labels", data=sl_flat)
        f.attrs["stim_labels_encoding"] = "native"
        # Keep this historical flag for readers predating typed scalar labels.
        f.attrs["stim_labels_is_str"] = False
    f.attrs["stim_labels_shape"] = list(bd.stim_labels.shape)


def _write_trial_arrays(f: h5py.File, bd: BaseData) -> None:
    for name, value in (
        ("trial", bd.trial),
        ("trial_starts", bd.trial_starts),
        ("trial_ends", bd.trial_ends),
        ("vision_onsets", bd.vision_onsets),
    ):
        if value is not None:
            f.create_dataset(name, data=value)
    f.attrs["data_mode"] = bd.data_mode


def _write_neuro_info(f: h5py.File, bd: BaseData, neuro_shape: tuple[int, ...]) -> None:
    ni = f.create_group("neuro_info")
    for k, v in bd.neuro_info.items():
        if k == "shape" or v is None:
            continue
        if isinstance(v, list):
            ni.attrs[k] = np.array(v, dtype=h5py.string_dtype()) if all(isinstance(x, str) for x in v) else np.array(v)
        else:
            ni.attrs[k] = v
    ni.attrs["shape"] = neuro_shape


def _write_vision_info(f: h5py.File, bd: BaseData) -> None:
    if not bd.vision_info:
        return
    vi = f.create_group("vision_info")
    for key, value in bd.vision_info.items():
        if key == "stim_ids":
            write_scalar_sequence(vi, "stim_ids", value, context="stimulus ID")
        elif value is not None:
            vi.attrs[key] = value


def _write_trial_info(f: h5py.File, bd: BaseData) -> None:
    if not bd.trial_info:
        return
    ti = f.create_group("trial_info")
    for key, value in bd.trial_info.items():
        if value is not None:
            ti.attrs[key] = np.array(value)


def _write_trial_meta(f: h5py.File, bd: BaseData) -> None:
    if bd.trial_meta is None:
        return
    if not isinstance(bd.trial_meta, pd.DataFrame):
        raise TypeError("trial_meta must be a pandas DataFrame for HDF5 serialization.")
    tm = f.create_group("trial_meta")
    tm.attrs["encoding"] = "pandas_v1"
    _write_pandas_index(tm.create_group("column_index"), bd.trial_meta.columns, "trial metadata columns")
    _write_pandas_index(tm.create_group("index"), bd.trial_meta.index, "trial metadata index")
    columns = tm.create_group("columns")
    for position, (_, series) in enumerate(bd.trial_meta.items()):
        _write_pandas_series(columns.create_group(str(position)), series, f"trial metadata column {series.name!r}")


def _write_pandas_index(group: h5py.Group, index: pd.Index, context: str) -> None:
    group.attrs["name_present"] = index.name is not None
    if index.name is not None:
        write_scalar_sequence(group, "name", [index.name], context=f"{context} name")
    if isinstance(index, pd.RangeIndex):
        group.attrs["kind"] = "range"
        group.attrs["start"] = index.start
        group.attrs["stop"] = index.stop
        group.attrs["step"] = index.step
        return
    group.attrs["kind"] = "values"
    _write_pandas_series(group.create_group("data"), pd.Series(index.array), context)


def _write_pandas_series(group: h5py.Group, series: pd.Series, context: str) -> None:
    dtype = series.dtype
    group.attrs["dtype"] = str(dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        group.attrs["kind"] = "categorical"
        group.attrs["ordered"] = dtype.ordered
        category_series = pd.Series(dtype.categories.array)
        _write_pandas_series(group.create_group("category_values"), category_series, f"{context} categories")
        codes = series.cat.codes.to_numpy(dtype=np.int64)
        group.create_dataset("codes", data=codes)
        return
    if isinstance(dtype, pd.DatetimeTZDtype):
        group.attrs["kind"] = "datetime_tz"
        group.attrs["tz"] = str(dtype.tz)
        group.attrs["unit"] = dtype.unit
        values = cast("Any", series.array).asi8
        group.create_dataset("values", data=values)
        return
    if pd.api.types.is_datetime64_dtype(dtype):
        group.attrs["kind"] = "datetime"
        dtype_text = str(dtype)
        unit = dtype_text[dtype_text.find("[") + 1 : -1]
        group.attrs["unit"] = unit
        group.create_dataset("values", data=np.asarray(series.array, dtype=f"datetime64[{unit}]").view(np.int64))
        return
    if pd.api.types.is_extension_array_dtype(dtype):
        supported_extension_dtypes = {
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "Float32",
            "Float64",
            "boolean",
            "string",
            "str",
        }
        if str(dtype) not in supported_extension_dtypes:
            raise TypeError(f"Unsupported {context} dtype {dtype!s}.")
        group.attrs["kind"] = "nullable"
        write_scalar_sequence(group, "values", series.array.tolist(), allow_missing=True, context=context)
        return
    values = series.to_numpy()
    if values.dtype.kind == "O":
        group.attrs["kind"] = "object_scalars"
        write_scalar_sequence(group, "values", values, allow_missing=True, context=context)
        return
    if values.dtype.kind in "biufcUS":
        group.attrs["kind"] = "numpy"
        if values.dtype.kind == "U":
            group.create_dataset("values", data=values.astype(h5py.string_dtype("utf-8")))
        else:
            group.create_dataset("values", data=values)
        return
    raise TypeError(f"Unsupported {context} dtype {dtype!s}.")


def _write_vision_store(f: h5py.File, bd: BaseData, options: H5StorageOptions) -> None:
    if bd._vision_data is not None and bd._vision_data.has_visual_representations:
        bd._vision_data.dump(f, storage_options=options)


def _write_stimuli_db(f: h5py.File, bd: BaseData, options: H5StorageOptions) -> None:
    db = bd._vision_data.db if bd._vision_data is not None else None
    if db is None:
        return
    grp = f.create_group("stimuli_db")
    grp.attrs["encoding"] = "ordered_entries_v1"
    for position, (stim_id, img) in enumerate(db.items()):
        entry = grp.create_group(str(position))
        write_scalar_sequence(entry, "id", [stim_id], context="stimulus ID")
        data, kind = _encode_image(img)
        kwargs = dataset_kwargs(data, options) if kind == "array" else {}
        entry.create_dataset("image", data=data, **kwargs)
        entry["image"].attrs["kind"] = kind


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def _make_coo_loader(fpath: Path, shape: tuple, dtype: str, identity: FileIdentity | None = None):
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

    identity = file_identity(fpath) if identity is None else identity

    def _loader(_fpath=fpath, _shape=shape, _dtype=dtype, _identity=identity) -> np.ndarray:
        from scipy.sparse import coo_matrix as _coo

        logger.info("Lazy-loading COO data from {}", _fpath)
        with h5py.File(_fpath, "r") as fh:
            verify_open_file_identity(fh, _identity, _fpath)
            row = fh["neuro_row"][:]
            col = fh["neuro_col"][:]
            data = fh["neuro_data"][:]
        flat_shape = (_shape[0] * _shape[1], _shape[2])
        sparse = _coo((data, (row, col)), shape=flat_shape, dtype=_dtype)
        return sparse.toarray().reshape(_shape)

    return _loader


def _make_dense_loader(fpath: Path, identity: FileIdentity | None = None):
    """Return a zero-argument callable that loads the dense neuro dataset.

    Parameters
    ----------
    fpath : Path
        HDF5 file containing a ``neuro`` dataset.

    Returns
    -------
    Callable[[], np.ndarray]
    """

    identity = file_identity(fpath) if identity is None else identity

    def _loader(_fpath=fpath, _identity=identity) -> np.ndarray:
        logger.info("Lazy-loading dense neuro from {}", _fpath)
        with h5py.File(_fpath, "r") as fh:
            verify_open_file_identity(fh, _identity, _fpath)
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
        file_token = open_file_identity(f)
        format_info = _detect_format(f, fpath)
        normalized = _normalize_v0(f) if format_info.schema_version == 0 else _normalize_v1(f)
        neuro, neuro_shape, neuro_loader = _read_neuro(f, fpath, file_token)
        neuro_info = _read_neuro_info(f, neuro_shape)
        stim_labels = _read_stim_labels(f)
        vision_info = _read_vision_info(f)
        trial, trial_starts, trial_ends, vision_onsets = _read_trial_arrays(f)
        trial_info = _read_trial_info(f)
        trial_meta = _read_trial_meta(f)
        _has_vision_store = "vision_store" in f
        _has_stimuli_db = "stimuli_db" in f
        data_mode_attr = normalized["data_mode"]
        if data_mode_attr not in ("continuous", "epochs", "patterns"):
            raise ValueError(f"Invalid data_mode {data_mode_attr!r} in HDF5 file {fpath}.")
        data_mode = cast("Literal['continuous', 'epochs', 'patterns']", data_mode_attr)

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
    if format_info.schema_version >= 1:
        try:
            bd._validate_state()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Invalid schema-1 VneuroTK HDF5 file {fpath}: {exc}") from exc
    _read_vision_data(fpath, bd, _has_vision_store, _has_stimuli_db, file_token)
    return bd


def _read_neuro(
    f: h5py.File, fpath: Path, identity: FileIdentity | None = None
) -> tuple[np.ndarray | None, tuple, Any]:
    """Read neuro data format and build a lazy loader if needed.

    Returns
    -------
    tuple[ndarray | None, tuple, callable | None]
        ``(neuro, neuro_shape, neuro_loader)`` — neuro is None when lazy.
    """
    neuro_format = _decode_text_attr(f.attrs.get("neuro_format", "dense"))
    identity = file_identity(fpath) if identity is None else identity

    if neuro_format == "coo":
        for attr in ("neuro_shape", "neuro_dtype"):
            if attr not in f.attrs:
                raise ValueError(f"Invalid COO neuro data in {fpath}: missing required root attribute {attr!r}.")
        missing = [name for name in ("neuro_row", "neuro_col", "neuro_data") if name not in f]
        if missing:
            raise ValueError(f"Invalid COO neuro data in {fpath}: missing required dataset(s): {', '.join(missing)}.")
        neuro_shape = tuple(int(x) for x in f.attrs["neuro_shape"])
        if len(neuro_shape) != 3:
            raise ValueError(f"Invalid COO neuro_shape {neuro_shape!r} in {fpath}; expected three dimensions.")
        neuro_dtype = _decode_text_attr(f.attrs["neuro_dtype"])
        return None, neuro_shape, LazyNeuroLoader(_make_coo_loader(fpath, neuro_shape, neuro_dtype, identity))

    if neuro_format != "dense":
        raise ValueError(f"Invalid neuro_format {neuro_format!r} in HDF5 file {fpath}; expected 'dense' or 'coo'.")
    if "neuro" not in f:
        raise ValueError(f"Invalid dense neuro data in {fpath}: missing required dataset 'neuro'.")
    neuro_shape = tuple(f["neuro"].shape)
    return None, neuro_shape, LazyNeuroLoader(_make_dense_loader(fpath, identity))


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
    identity: FileIdentity,
) -> None:
    """Reconstruct and attach VisionData to *bd* if the file contains vision data.

    Symmetric with :func:`_write_vision_store` + :func:`_write_stimuli_db`.
    """
    if not has_vision_store and not has_stimuli_db:
        return

    output_order: np.ndarray | None
    if bd.data_mode == "patterns":
        output_order = bd._pattern_stim_ids()
    else:
        output_order = bd.trial_stim_ids if bd.configured else None
    if output_order is None:
        raise ValueError("Saved vision data cannot be aligned because explicit stimulus IDs are missing.")

    from vneurotk.io.loader import LazyH5Dict
    from vneurotk.vision.data import VisionData

    loaded_stimuli = LazyH5Dict(fpath, "stimuli_db", identity=identity) if has_stimuli_db else None
    with h5py.File(fpath, "r") as f:
        verify_open_file_identity(f, identity, fpath)
        store = VisionData.from_h5(
            f,
            output_order=output_order,
            vision_db=loaded_stimuli,
            fpath=fpath,
            file_identity=identity,
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
    if "shape" in neuro_info:
        try:
            declared_shape = tuple(int(dim) for dim in neuro_info["shape"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid neuro_info shape {neuro_info['shape']!r} in HDF5 file.") from exc
        if declared_shape != tuple(neuro_shape):
            raise ValueError(
                f"Invalid neuro_info shape {declared_shape!r}; neural dataset shape is {tuple(neuro_shape)!r}."
            )
    neuro_info["shape"] = list(neuro_shape)
    return neuro_info


def _read_stim_labels(f: h5py.File) -> np.ndarray | None:
    if "stim_labels" not in f:
        return None
    vis_shape = f.attrs.get("stim_labels_shape", None)
    encoding = _decode_text_attr(f.attrs.get("stim_labels_encoding", "legacy"))
    if encoding == "typed_scalars":
        vision = np.asarray(read_scalar_sequence(f, "stim_labels", allow_nonfinite=True), dtype=object)
    else:
        vis_is_str = bool(f.attrs.get("stim_labels_is_str", False))
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
        if isinstance(vi_grp["stim_ids"], h5py.Group):
            vision_info["stim_ids"] = read_scalar_sequence(vi_grp, "stim_ids")
        else:
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
    tm = f["trial_meta"]
    if _decode_text_attr(tm.attrs.get("encoding", "legacy")) != "pandas_v1":
        cols: dict = {}
        for col_name in tm:
            vals = tm[col_name][:]
            if vals.dtype.kind in ("S", "O"):
                vals = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in vals])
            cols[col_name] = vals
        return pd.DataFrame(cols)

    if "column_index" in tm:
        column_index = _read_pandas_index(tm["column_index"])
        labels = list(column_index)
    else:
        labels = read_scalar_sequence(tm, "column_labels")
        column_index = pd.Index(labels)
    series = [_read_pandas_series(tm["columns"][str(position)]) for position in range(len(labels))]
    frame = pd.concat(series, axis=1) if series else pd.DataFrame()
    frame.columns = column_index
    frame.index = _read_pandas_index(tm["index"])
    return frame


def _read_pandas_index(group: h5py.Group) -> pd.Index:
    name = read_scalar_sequence(group, "name")[0] if bool(group.attrs.get("name_present", False)) else None
    kind = _decode_text_attr(group.attrs["kind"])
    if kind == "range":
        return pd.RangeIndex(int(group.attrs["start"]), int(group.attrs["stop"]), int(group.attrs["step"]), name=name)
    values = _read_pandas_series(group["data"])
    return pd.Index(values.array, name=name)


def _read_pandas_series(group: h5py.Group) -> pd.Series:
    kind = _decode_text_attr(group.attrs["kind"])
    dtype = _decode_text_attr(group.attrs["dtype"])
    if kind == "categorical":
        if "category_values" in group:
            categories = pd.Index(_read_pandas_series(group["category_values"]).array)
        else:
            categories = read_scalar_sequence(group, "categories")
        values = pd.Categorical.from_codes(
            group["codes"][:],
            categories=categories,
            ordered=bool(group.attrs["ordered"]),
        )
        return pd.Series(values)
    if kind == "datetime_tz":
        unit = _decode_text_attr(group.attrs.get("unit", "ns"))
        values = pd.to_datetime(group["values"][:], unit=unit, utc=True).tz_convert(
            _decode_text_attr(group.attrs["tz"])
        )
        return pd.Series(values).astype(dtype)
    if kind == "datetime":
        unit = _decode_text_attr(group.attrs.get("unit", "ns"))
        return pd.Series(group["values"][:].view(f"datetime64[{unit}]")).astype(dtype)
    if kind in ("nullable", "object_scalars"):
        values = read_scalar_sequence(group, "values")
        return pd.Series(values, dtype=dtype if kind == "nullable" else object)
    values = group["values"][:]
    if values.dtype.kind in ("S", "O"):
        values = np.asarray([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])
    return pd.Series(values, dtype=dtype)
