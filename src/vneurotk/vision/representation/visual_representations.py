"""DNN activation containers — atomic and collection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

import numpy as np
import pandas as pd

__all__ = ["VisualRepresentation", "VisualRepresentations"]


class VisualRepresentation:
    """Atomic activation record: one model × one module.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"facebook/dinov2-base"``.
    module_name : str
        Module name as from ``named_modules()``, e.g. ``"encoder.layer.11"``.
    module_type : str
        Class name of the module, e.g. ``"Dinov2Layer"``.
    stim_ids : list
        Ordered stimulus identifiers corresponding to the first axis of *array*.
    array : np.ndarray or None
        Activation array of shape ``(n_stim, ...)``.  Mutually exclusive with
        *array_loader*; one of the two must be provided.
    array_loader : callable or None
        Zero-argument callable that returns the activation array on first access.
        Used for lazy loading from HDF5.  Mutually exclusive with *array*.
    shape : tuple or None
        Pre-computed shape to return from :attr:`shape` without triggering array
        loading.  Required when *array_loader* is given; ignored otherwise.
    """

    def __init__(
        self,
        model: str,
        module_name: str,
        module_type: str,
        stim_ids: list,
        array: np.ndarray | None = None,
        *,
        array_loader: Callable[[], np.ndarray] | None = None,
        shape: tuple | None = None,
    ) -> None:
        if array is None and array_loader is None:
            raise ValueError("Either array or array_loader must be provided.")
        self.model = model
        self.module_name = module_name
        self.module_type = module_type
        self.stim_ids: tuple = tuple(stim_ids)
        self._array: np.ndarray | None = np.asarray(array) if array is not None else None
        self._array_loader: Callable[[], np.ndarray] | None = array_loader
        self._shape: tuple | None = shape if self._array is None else None
        self._id_to_idx: dict[Any, int] = {sid: i for i, sid in enumerate(self.stim_ids)}

    @property
    def array(self) -> np.ndarray:
        """Activation array, loaded lazily from HDF5 if constructed with *array_loader*."""
        if self._array is None:
            self._array = self._array_loader()  # ty: ignore[call-non-callable]
            self._array_loader = None
            self._shape = None
        return self._array

    @array.setter
    def array(self, value: np.ndarray) -> None:
        self._array = np.asarray(value)
        self._shape = None

    @property
    def n_stim(self) -> int:
        """Number of stimuli."""
        return len(self.stim_ids)

    @property
    def shape(self) -> tuple:
        """Shape of the activation array ``(n_stim, ...)``."""
        if self._shape is not None:
            return self._shape
        return self.array.shape

    def select(self, ids: list | np.ndarray) -> VisualRepresentation:
        """Return a subset of stimuli by their IDs.

        Parameters
        ----------
        ids : list or np.ndarray
            Stimulus IDs to select.

        Returns
        -------
        VisualRepresentation
        """
        ids_list = list(ids)
        indices = np.array([self._id_to_idx[sid] for sid in ids_list])
        return VisualRepresentation(
            model=self.model,
            module_name=self.module_name,
            module_type=self.module_type,
            stim_ids=ids_list,
            array=self.array[indices],
        )

    def __repr__(self) -> str:
        return f"VisualRepresentation(model={self.model!r}, module={self.module_name!r}, shape={self.shape})"


class VisualRepresentations:
    """Collection of :class:`VisualRepresentation` objects.

    Returned by :meth:`~vneurotk.vision.model.base.VisionModel.extract`.
    Supports DataFrame-style filtering via boolean masks on :attr:`meta`.

    Parameters
    ----------
    representations : list[VisualRepresentation]
        Ordered list of atomic activation records.

    Examples
    --------
    >>> visual_representations = model.extract(images)
    >>> meta = visual_representations.meta
    >>> subset = visual_representations[meta["module_type"] == "Dinov2Layer"]
    """

    def __init__(self, representations: list[VisualRepresentation]) -> None:
        self._visual_representations: list[VisualRepresentation] = list(representations)
        self._assert_shared_stim_ids(self._visual_representations)
        self._meta: pd.DataFrame = self._build_meta(self._visual_representations)

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_shared_stim_ids(visual_representations: list[VisualRepresentation]) -> None:
        if len(visual_representations) > 1:
            ref = visual_representations[0].stim_ids
            for vr in visual_representations[1:]:
                if vr.stim_ids != ref:
                    raise ValueError(
                        f"All VisualRepresentations must share the same stim_ids. "
                        f"Mismatch between '{visual_representations[0].module_name}' and '{vr.module_name}'."
                    )

    @staticmethod
    def _build_meta(visual_representations: list[VisualRepresentation]) -> pd.DataFrame:
        if not visual_representations:
            return pd.DataFrame(columns=["model", "module_type", "module_name", "shape"])
        return pd.DataFrame(
            [
                {
                    "model": vr.model,
                    "module_type": vr.module_type,
                    "module_name": vr.module_name,
                    "shape": vr.shape,
                }
                for vr in visual_representations
            ]
        )

    @property
    def meta(self) -> pd.DataFrame:
        """DataFrame with columns ``model``, ``module_type``, ``module_name``, ``shape``."""
        return self._meta

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._visual_representations)

    def __iter__(self):
        return iter(self._visual_representations)

    @overload
    def __getitem__(self, key: str) -> VisualRepresentation: ...

    @overload
    def __getitem__(self, key: int | np.integer) -> VisualRepresentation: ...

    @overload
    def __getitem__(self, key: pd.Series | np.ndarray) -> VisualRepresentations: ...

    def __getitem__(
        self, key: pd.Series | np.ndarray | int | np.integer | str
    ) -> VisualRepresentations | VisualRepresentation:
        """Filter or index into the collection.

        Parameters
        ----------
        key : pd.Series or np.ndarray of bool, int, or str
            - ``str`` → look up by module name, returns :class:`VisualRepresentation`.
            - ``int`` → positional index, returns :class:`VisualRepresentation`.
            - 1-D bool array/Series aligned to :attr:`meta` → returns filtered
              :class:`VisualRepresentations`.

        Raises
        ------
        TypeError
            If *key* is a 0-d boolean (e.g. the result of a scalar comparison like
            ``'module_type' == 'Dinov2Layer'``).
        """
        if isinstance(key, str):
            return self.by_module(key)
        if isinstance(key, (int, np.integer)):
            return self._visual_representations[int(key)]
        arr = np.asarray(key)
        if arr.ndim == 0:
            raise TypeError(
                "Boolean index must be a 1-D array aligned to meta rows. "
                "Got a scalar — did you mean meta['col'] == value instead of 'col' == value?"
            )
        return self.filter(arr.astype(bool))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_stim(self) -> int:
        """Number of stimuli (from first record, or 0 if empty)."""
        return self._visual_representations[0].n_stim if self._visual_representations else 0

    @property
    def stim_ids(self) -> tuple:
        """Stimulus IDs shared by all records."""
        return self._visual_representations[0].stim_ids if self._visual_representations else ()

    @property
    def module_names(self) -> list[str]:
        """Module names of all contained records."""
        return [vr.module_name for vr in self._visual_representations]

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def numpy(self, layer: str) -> np.ndarray:
        """Return activation array for *layer*, shape ``(n_stim, ...)``.

        Parameters
        ----------
        layer : str
            Module name.
        """
        return self.by_module(layer).array

    def to_tensor(self, layer: str) -> Any:
        """Return activations for *layer* as a PyTorch tensor.

        Parameters
        ----------
        layer : str
            Module name.
        """
        try:
            import torch  # type: ignore

            return torch.from_numpy(np.asarray(self.by_module(layer).array))
        except ImportError as exc:
            raise ImportError("torch is required for to_tensor()") from exc

    def select(self, ids: list | np.ndarray) -> VisualRepresentations:
        """Return a subset of stimuli by their IDs across all records.

        Parameters
        ----------
        ids : list or np.ndarray
            Stimulus IDs to keep.
        """
        return VisualRepresentations([vr.select(ids) for vr in self._visual_representations])

    def select_by_index(self, indices: list | np.ndarray) -> VisualRepresentations:
        """Return a subset of stimuli by positional index across all records.

        Parameters
        ----------
        indices : list or np.ndarray
            Integer indices.
        """
        ids = [self.stim_ids[i] for i in indices]
        return self.select(ids)

    def by_module(self, name: str, model: str | None = None) -> VisualRepresentation:
        """Return the :class:`VisualRepresentation` for *name*.

        Parameters
        ----------
        name : str
            Module name.
        model : str or None
            Model identifier to disambiguate when multiple records share the same
            module name.

        Raises
        ------
        KeyError
            If *name* is not found, or is ambiguous and *model* was not given.
        """
        if model is not None:
            for vr in self._visual_representations:
                if vr.model == model and vr.module_name == name:
                    return vr
            raise KeyError(f"Module {name!r} for model {model!r} not found.")

        matches = [vr for vr in self._visual_representations if vr.module_name == name]
        if not matches:
            raise KeyError(f"Layer {name!r} not found in VisualRepresentations.")
        if len(matches) > 1:
            models = [vr.model for vr in matches]
            raise KeyError(f"Module {name!r} found in {len(matches)} models: {models}. Specify model= to disambiguate.")
        return matches[0]

    def filter(self, mask: pd.Series | np.ndarray) -> VisualRepresentations:
        """Return a subset filtered by a 1-D boolean mask over :attr:`meta` rows.

        Parameters
        ----------
        mask : pd.Series or np.ndarray of bool
            Aligned to :attr:`meta` rows.
        """
        bool_mask = np.asarray(mask, dtype=bool)
        return VisualRepresentations(
            [vr for vr, keep in zip(self._visual_representations, bool_mask, strict=True) if keep]
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VisualRepresentations("
            f"{self.n_stim} stimuli x {len(self._visual_representations)} modules, "
            f"models={list({vr.model for vr in self._visual_representations})})"
        )
