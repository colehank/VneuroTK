"""Safe scalar and dataset encoding helpers for VneuroTK HDF5 files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

SCALAR_ENCODING = "vneurotk_scalar_v1"

FileIdentity = tuple[int, int]


def decode_text(value: Any) -> str:
    """Return an HDF5 text value as a Python string."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def file_identity(path: Path | str) -> FileIdentity:
    """Return the device/inode identity of an HDF5 pathname."""
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino


def verify_file_identity(path: Path | str, expected: FileIdentity) -> None:
    """Raise when an atomic replacement changed a lazy loader's backing file."""
    try:
        actual = file_identity(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Lazy HDF5 backing file {Path(path)} is no longer available.") from exc
    if actual != expected:
        raise RuntimeError(
            f"Lazy HDF5 backing file {Path(path)} has changed since this object was loaded; "
            "reload the recording instead of mixing file snapshots."
        )


def open_file_identity(f: h5py.File) -> FileIdentity:
    """Return the device/inode identity of an already-open HDF5 handle."""
    try:
        handle = f.id.get_vfd_handle()
        fd = handle[0] if isinstance(handle, tuple) else handle
        stat = os.fstat(int(fd))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError("Cannot determine opened HDF5 file identity.") from exc
    return stat.st_dev, stat.st_ino


def verify_open_file_identity(f: h5py.File, expected: FileIdentity, path: Path | str) -> None:
    """Raise when an opened HDF5 handle is not the captured backing file."""
    try:
        actual = open_file_identity(f)
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot verify lazy HDF5 backing file identity for {Path(path)}.") from exc
    if actual != expected:
        raise RuntimeError(
            f"Lazy HDF5 backing file {Path(path)} has changed since this object was loaded; "
            "reload the recording instead of mixing file snapshots."
        )


_BOOL = 1
_INT = 2
_FLOAT = 3
_STRING = 4
_NONE = 5
_PD_NA = 6
_NAT = 7


@dataclass(frozen=True)
class H5StorageOptions:
    """Compression and chunk-size policy for large numerical datasets."""

    compression: str | None = "gzip"
    compression_opts: Any = 4
    chunk_target_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if isinstance(self.compression, bool) or (
            self.compression is not None and not isinstance(self.compression, str)
        ):
            raise TypeError("compression must be an HDF5 filter name or None.")
        if isinstance(self.chunk_target_bytes, bool) or self.chunk_target_bytes <= 0:
            raise ValueError("chunk_target_bytes must be a positive integer.")


def dataset_kwargs(data: Any, options: H5StorageOptions) -> dict[str, Any]:
    """Return HDF5 filter/chunk arguments for a non-scalar numerical array."""
    arr = np.asarray(data)
    if arr.ndim == 0 or arr.size == 0 or arr.dtype.kind == "O":
        return {}

    shape = [max(1, int(size)) for size in arr.shape]
    itemsize = max(1, int(arr.dtype.itemsize))
    while int(np.prod(shape, dtype=np.int64)) * itemsize > options.chunk_target_bytes:
        axis = max(range(len(shape)), key=lambda i: shape[i])
        if shape[axis] == 1:
            break
        shape[axis] = max(1, (shape[axis] + 1) // 2)

    kwargs: dict[str, Any] = {"chunks": tuple(shape)}
    if options.compression is not None:
        kwargs["compression"] = options.compression
        if options.compression_opts is not None and options.compression not in ("lzf",):
            kwargs["compression_opts"] = options.compression_opts
        if arr.dtype.kind in "biufc":
            kwargs["shuffle"] = True
    return kwargs


def _native_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def scalar_token(value: Any, *, allow_missing: bool = False, context: str = "value") -> tuple[int, str]:
    """Encode one supported scalar without relying on its string form as a path."""
    value = _native_scalar(value)
    if allow_missing:
        if value is None:
            return _NONE, ""
        if value is pd.NA:
            return _PD_NA, ""
        if value is pd.NaT:
            return _NAT, ""
    if isinstance(value, bool):
        return _BOOL, "1" if value else "0"
    if isinstance(value, int):
        return _INT, str(value)
    if isinstance(value, float):
        if not np.isfinite(value) and not (allow_missing and np.isnan(value)):
            raise ValueError(f"Non-finite float {context} {value!r} is not supported; use a finite float value.")
        return _FLOAT, value.hex()
    if isinstance(value, str):
        if "\x00" in value:
            raise ValueError(
                f"Unsupported {context} string containing an embedded NUL character; "
                "supported strings are valid UTF-8 text without NUL characters."
            )
        return _STRING, value
    supported = "bool, int, float, and str"
    missing = ", plus None/pandas missing values" if allow_missing else ""
    raise TypeError(f"Unsupported {context} type {type(value).__name__}; supported values are {supported}{missing}.")


def decode_scalar(type_code: int, payload: Any, *, allow_nonfinite: bool = False) -> Any:
    """Decode one scalar written by :func:`scalar_token`."""
    text = decode_text(payload)
    if type_code == _BOOL:
        return text == "1"
    if type_code == _INT:
        return int(text)
    if type_code == _FLOAT:
        value = float.fromhex(text)
        if not allow_nonfinite and not np.isfinite(value):
            raise ValueError(f"Invalid non-finite float in typed scalar encoding: {value!r}.")
        return value
    if type_code == _STRING:
        return text
    if type_code == _NONE:
        return None
    if type_code == _PD_NA:
        return pd.NA
    if type_code == _NAT:
        return pd.NaT
    raise ValueError(f"Unknown VneuroTK scalar type code {type_code}.")


def write_scalar(group: h5py.Group, name: str, value: Any, *, context: str = "value") -> None:
    """Write one typed scalar into a child group."""
    type_code, payload = scalar_token(value, context=context)
    target = group.create_group(name)
    target.attrs["encoding"] = SCALAR_ENCODING
    target.attrs["type"] = type_code
    target.create_dataset("value", data=payload, dtype=h5py.string_dtype("utf-8"))


def read_scalar(group: h5py.Group, name: str) -> Any:
    """Read one typed scalar child group."""
    target = group[name]
    if decode_text(target.attrs.get("encoding", "")) != SCALAR_ENCODING:
        raise ValueError(f"Unsupported scalar encoding in HDF5 group {target.name!r}.")
    return decode_scalar(int(target.attrs["type"]), target["value"][()])


def write_scalar_sequence(
    group: h5py.Group,
    name: str,
    values: Any,
    *,
    allow_missing: bool = False,
    context: str = "value",
) -> h5py.Group:
    """Write an ordered, type-preserving sequence of supported scalars."""
    tokens = [scalar_token(value, allow_missing=allow_missing, context=context) for value in values]
    target = group.create_group(name)
    target.attrs["encoding"] = SCALAR_ENCODING
    target.create_dataset("types", data=np.asarray([token[0] for token in tokens], dtype=np.uint8))
    target.create_dataset(
        "values",
        data=np.asarray([token[1] for token in tokens], dtype=h5py.string_dtype("utf-8")),
    )
    return target


def read_scalar_sequence(group: h5py.Group, name: str, *, allow_nonfinite: bool = False) -> list[Any]:
    """Read a sequence written by :func:`write_scalar_sequence`."""
    target = group[name]
    if not isinstance(target, h5py.Group) or decode_text(target.attrs.get("encoding", "")) != SCALAR_ENCODING:
        raise ValueError(f"Unsupported scalar sequence encoding in HDF5 object {target.name!r}.")
    types = target["types"][:]
    values = target["values"][:]
    if len(types) != len(values):
        raise ValueError(f"Invalid scalar sequence {target.name!r}: type/value lengths differ.")
    return [
        decode_scalar(int(type_code), payload, allow_nonfinite=allow_nonfinite)
        for type_code, payload in zip(types, values, strict=True)
    ]
