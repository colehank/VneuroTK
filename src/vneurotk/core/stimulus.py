"""Stimulus Set — image database for a Joint Data Object."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

StimImage = Any  # PIL.Image | np.ndarray | Path | str


def _norm_key(sid: Any) -> Any:
    """Convert numpy scalars to Python natives for dict keys."""
    return sid.item() if hasattr(sid, "item") else sid


def _unique_ordered_keys(stim_ids: Any) -> list:
    """Return unique values from *stim_ids* in first-appearance order."""
    seen: dict = {}
    result: list = []
    for sid in stim_ids:
        key = _norm_key(sid)
        if key not in seen:
            seen[key] = len(result)
            result.append(key)
    return result


class StimulusSet:
    """Container linking per-onset stimulus IDs to per-stimulus images.

    Parameters
    ----------
    stim_ids : array-like, shape (n_onsets,)
        Stimulus ID for each onset.
    stimuli : dict, list, np.ndarray, or None
        Image source for each unique stimulus.

        ``dict``
            ``{stim_id: image}`` — explicit mapping.
        ``list`` / ``np.ndarray`` of length *n_unique*
            Auto-assigned in first-appearance order from *stim_ids*.
        ``list`` / ``np.ndarray`` of length *n_onsets*
            Aggregated per stim_id (first occurrence wins per id).
        ``None``
            Only stim IDs are stored; no image data.

    Notes
    -----
    Supported image types: ``PIL.Image``, ``np.ndarray``, ``Path``, ``str``.
    ``Path`` / ``str`` entries are **lazily loaded** as ``PIL.Image`` on
    ``__getitem__`` access.

    Examples
    --------
    >>> ss = StimulusSet(stim_ids=[0, 1, 0], stimuli={0: arr0, 1: arr1})
    >>> ss[0]   # returns arr0
    """

    def __init__(
        self,
        stim_ids: np.ndarray | list,
        stimuli: dict | list | np.ndarray | None = None,
    ) -> None:
        self._vision_id = np.asarray(stim_ids)
        if self._vision_id.ndim != 1:
            raise ValueError(f"stim_ids must be 1-D, got shape {self._vision_id.shape}")

        self._stimuli: Mapping[Any, StimImage] | None = None
        if stimuli is not None:
            self._stimuli = self._build_stimuli(self._vision_id, stimuli)

    # --- properties -------------------------------------------------------

    @property
    def stim_ids(self) -> np.ndarray:
        """Per-onset stimulus IDs, shape ``(n_onsets,)``."""
        return self._vision_id

    @property
    def stimuli(self) -> Mapping[Any, StimImage] | None:
        """Dict ``{stim_id: image}`` or ``None`` if not provided."""
        return self._stimuli

    @property
    def unique_ids(self) -> list:
        """Unique stimulus IDs in sorted order."""
        return np.unique(self._vision_id).tolist()

    # --- item access ------------------------------------------------------

    def __getitem__(self, stim_id: Any) -> StimImage:
        """Return the image for *stim_id*, lazily loading Path / str entries."""
        if self._stimuli is None:
            raise KeyError("StimulusSet has no stimuli data (vision_db=None)")
        key = _norm_key(stim_id)
        img = self._stimuli[key]
        if isinstance(img, (str, Path)):
            try:
                from PIL import Image  # type: ignore[import]

                return Image.open(img)
            except ImportError as exc:
                raise ImportError("Pillow is required to load images from paths.  Install with: uv add Pillow") from exc
        return img

    def __len__(self) -> int:
        return len(self._vision_id)

    def __contains__(self, stim_id: Any) -> bool:
        """Return True if *stim_id* has an associated image in this StimulusSet."""
        if self._stimuli is None:
            return False
        return _norm_key(stim_id) in self._stimuli

    def items(self):
        """Yield ``(stim_id, image)`` pairs for all unique stimuli."""
        if self._stimuli is None:
            return iter([])
        return iter(self._stimuli.items())

    def __repr__(self) -> str:
        has_db = self._stimuli is not None
        return f"StimulusSet(n_onsets={len(self._vision_id)}, n_unique={len(self.unique_ids)}, has_stimuli={has_db})"

    # --- explicit classmethods -------------------------------------------

    @classmethod
    def from_dict(cls, stim_ids: np.ndarray, stimuli: dict) -> StimulusSet:
        """Build from an explicit ``{stim_id: image}`` mapping.

        Parameters
        ----------
        stim_ids : array-like, shape (n_onsets,)
            Stimulus ID per onset.
        stimuli : dict
            Complete ``{stim_id: image}`` mapping; every ID in *stim_ids*
            must have a corresponding entry.

        Returns
        -------
        StimulusSet
        """
        ss: StimulusSet = cls.__new__(cls)
        ss._vision_id = np.asarray(stim_ids)
        if ss._vision_id.ndim != 1:
            raise ValueError(f"stim_ids must be 1-D, got shape {ss._vision_id.shape}")
        ss._stimuli = {_norm_key(k): v for k, v in stimuli.items()}
        return ss

    @classmethod
    def from_unique_list(cls, stim_ids: np.ndarray, images: list) -> StimulusSet:
        """Build from a list of images aligned with unique stim_ids.

        *images* must have exactly as many entries as there are unique
        stimulus IDs, ordered by first appearance in *stim_ids*.

        Parameters
        ----------
        stim_ids : array-like, shape (n_onsets,)
            Stimulus ID per onset.
        images : list
            One image per unique stimulus, in first-appearance order.

        Returns
        -------
        StimulusSet

        Raises
        ------
        ValueError
            If ``len(images)`` does not equal the number of unique stim IDs.
        """
        ids = np.asarray(stim_ids)
        unique_ordered = _unique_ordered_keys(ids)
        if len(unique_ordered) != len(images):
            raise ValueError(f"images length {len(images)} != {len(unique_ordered)} unique stim IDs")
        return cls.from_dict(ids, dict(zip(unique_ordered, images, strict=True)))

    @classmethod
    def from_h5(cls, stim_ids: np.ndarray, lazy_dict: Any) -> StimulusSet:
        """Build from a :class:`LazyH5Dict` or any Mapping for lazy HDF5 access.

        Parameters
        ----------
        stim_ids : array-like, shape (n_onsets,)
            Stimulus ID per onset.
        lazy_dict : Mapping
            Any mapping conforming to :class:`~vneurotk.vision.image_source.ImageSource`
            (e.g. :class:`~vneurotk.io.loader.LazyH5Dict`).

        Returns
        -------
        StimulusSet
        """
        ss: StimulusSet = cls.__new__(cls)
        ss._vision_id = np.asarray(stim_ids)
        if ss._vision_id.ndim != 1:
            raise ValueError(f"stim_ids must be 1-D, got shape {ss._vision_id.shape}")
        ss._stimuli = lazy_dict
        return ss

    # --- private helpers --------------------------------------------------

    @staticmethod
    def _infer_stimuli_mode(n_seq: int, n_unique: int, n_onsets: int) -> str:
        """Infer whether a sequence maps images by unique ID or by onset.

        Parameters
        ----------
        n_seq : int
            Length of the provided image sequence.
        n_unique : int
            Number of unique stimulus IDs.
        n_onsets : int
            Total number of onsets (trials).

        Returns
        -------
        str
            ``"by_unique"`` if *n_seq* == *n_unique*;
            ``"by_onset"`` if *n_seq* == *n_onsets* and *n_seq* != *n_unique*.

        Raises
        ------
        ValueError
            If *n_seq* matches neither *n_unique* nor *n_onsets*.
        """
        if n_seq == n_unique:
            return "by_unique"
        if n_seq == n_onsets:
            return "by_onset"
        raise ValueError(
            f"vision_db length {n_seq} does not match "
            f"n_unique={n_unique} or n_onsets={n_onsets}.  "
            "Provide a dict, or a list/array of length n_unique or n_onsets."
        )

    @staticmethod
    def _build_stimuli(
        stim_ids: np.ndarray,
        stimuli: dict | list | np.ndarray,
    ) -> Mapping[Any, Any]:
        from collections.abc import Mapping as _M

        if isinstance(stimuli, _M):
            if isinstance(stimuli, dict):
                return {_norm_key(k): v for k, v in stimuli.items()}
            return stimuli  # LazyH5Dict or other Mapping — preserve for lazy access

        seq: list = stimuli if isinstance(stimuli, list) else list(stimuli)
        n_onsets = len(stim_ids)
        unique_ordered = _unique_ordered_keys(stim_ids)
        n_unique = len(unique_ordered)

        mode = StimulusSet._infer_stimuli_mode(len(seq), n_unique, n_onsets)
        if mode == "by_unique":
            logger.debug("StimulusSet: auto-assigning {} images by unique-id order", n_unique)
            return {uid: seq[i] for i, uid in enumerate(unique_ordered)}
        else:
            logger.debug(
                "StimulusSet: aggregating {} onset images into {} unique ids",
                n_onsets,
                n_unique,
            )
            result: dict[Any, StimImage] = {}
            for sid, img in zip(stim_ids, seq, strict=True):
                key = _norm_key(sid)
                if key not in result:
                    result[key] = img
            return result
