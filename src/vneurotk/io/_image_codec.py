"""Image serialization codec for HDF5 storage.

Both encode and decode live here so the *kind* string contract has exactly
one authoritative location.  Any change to the supported format set (adding
JPEG compression, removing path references, etc.) requires edits in this
file only.

``_encode_image`` is used by the write path (``h5_persistence``).
``_decode_image`` is used by the read path (``loader.LazyH5Dict``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _encode_image(img: Any) -> tuple[np.ndarray, str]:
    """Encode an image value to a ``(data_array, kind)`` pair for HDF5 storage.

    Parameters
    ----------
    img : str, Path, np.ndarray, or array-like
        Image to encode.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(data, kind)`` where *kind* is one of ``"image_bytes"`` or
        ``"array"``.
    """
    if isinstance(img, (str, Path)):
        return np.frombuffer(Path(img).read_bytes(), dtype=np.uint8), "image_bytes"
    return np.asarray(img), "array"


def _decode_image(data: np.ndarray, kind: str) -> Any:
    """Decode a raw HDF5 array to a numpy image array based on *kind*.

    Symmetric with :func:`_encode_image`.

    Parameters
    ----------
    data : np.ndarray
        Raw data as loaded from the HDF5 dataset (``ds[:]`` or ``ds[()]``).
    kind : str
        Storage kind written by ``_encode_image``: ``"path"``, ``"image_bytes"``,
        or ``"array"``.

    Returns
    -------
    np.ndarray
        RGB image array for image kinds; ``data`` unchanged for ``"array"``.
    """
    if kind == "path":
        path_str = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        try:
            from PIL import Image  # type: ignore[import]

            return np.asarray(Image.open(path_str).convert("RGB"))
        except ImportError as exc:
            raise ImportError("Pillow is required to load images saved as paths.  Install with: uv add Pillow") from exc
    if kind == "image_bytes":
        try:
            import io

            from PIL import Image  # type: ignore[import]

            return np.asarray(Image.open(io.BytesIO(data.tobytes())).convert("RGB"))
        except ImportError as exc:
            raise ImportError("Pillow is required to load image_bytes.  Install with: uv add Pillow") from exc
    return data
