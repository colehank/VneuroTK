"""Vision storage and view layers for VneuroTK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger

from vneurotk.vision.representation.visual_representations import (
    VisualRepresentation,
    VisualRepresentations,
)

if TYPE_CHECKING:
    import h5py


class VisionData:
    """Aligned storage and view for Visual Representations within a Recording.

    Stores unique-stimulus activations internally as a
    ``dict[(model_id, module_name), VisualRepresentation]`` and re-indexes
    to ``output_order`` (typically ``BaseData.trial_stim_ids``) at read time.

    All storage, HDF5 persistence, and trial-aligned view logic live here;
    there is no separate inner storage class.

    Attributes
    ----------
    db : Any
        Original stimulus image database.
    output_order : np.ndarray
        Sequence of stimulus IDs defining the desired output ordering.
    meta : pd.DataFrame
        Metadata for all stored records: ``model``, ``module_type``,
        ``module_name``, ``shape``.
    """

    def __init__(self, output_order: np.ndarray, vision_db: Any = None) -> None:
        self._records: dict[tuple[str, str], VisualRepresentation] = {}
        self._output_order = np.asarray(output_order)
        self._vision_db = vision_db
        self._align_cache: dict[tuple[str, str], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    @property
    def db(self) -> Any:
        """Original stimulus image database."""
        return self._vision_db

    @property
    def has_visual_representations(self) -> bool:
        """Whether any VisualRepresentations have been stored."""
        return bool(self._records)

    @property
    def meta(self) -> pd.DataFrame:
        """DataFrame with columns ``model``, ``module_type``, ``module_name``, ``shape``."""
        return VisualRepresentations._build_meta(list(self._records.values()))

    def attach_db(self, db: Any) -> None:
        """Set or replace the image database attached to this VisionData.

        Parameters
        ----------
        db : Any
        """
        self._vision_db = db

    def _store_records(self, visual_representations: VisualRepresentations, overwrite: bool) -> None:
        """Write records into ``_records``, respecting *overwrite*."""
        for vr in visual_representations:
            key = (vr.model, vr.module_name)
            if key in self._records:
                if overwrite:
                    self._records[key] = vr
                    self._align_cache.pop(key, None)
                    logger.info("Overwriting: model={}, module={}", vr.model, vr.module_name)
                else:
                    logger.info(
                        "Skipping existing (overwrite=False): model={}, module={}",
                        vr.model,
                        vr.module_name,
                    )
            else:
                self._records[key] = vr

    @staticmethod
    def _assert_stim_ids_cover_output_order(vr: Any, output_order: np.ndarray) -> None:
        vr_ids = set(vr.stim_ids)
        missing = [sid for sid in output_order if sid not in vr_ids]
        if missing:
            raise ValueError(
                f"VR '{vr.module_name}' (model='{vr.model}') stim_ids do not cover "
                f"output_order. Missing {len(missing)} ID(s): {missing[:5]}"
                f"{'…' if len(missing) > 5 else ''}."
            )

    def add(self, visual_representations: VisualRepresentations, overwrite: bool = False) -> None:
        """Add records — validates stim_id coverage then stores.

        Parameters
        ----------
        visual_representations : VisualRepresentations
        overwrite : bool
        """
        for vr in visual_representations:
            self._assert_stim_ids_cover_output_order(vr, self._output_order)
        self._store_records(visual_representations, overwrite)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_from(
        self,
        model: Any,
        vision_db: Any = None,
        *,
        batch_size: int = 32,
        overwrite: bool = False,
    ) -> None:
        """Extract DNN features and store them.

        Accepts either a pre-built :class:`~vneurotk.vision.model.base.VisionModel`
        or a model-id string (in which case a VisionModel is built internally with
        default settings — use ``VisionModel`` directly for custom backends/selectors).

        Parameters
        ----------
        model : VisionModel or str
            A configured :class:`~vneurotk.vision.model.base.VisionModel`, or a
            model-id string like ``"facebook/dinov2-base"`` / ``"resnet50"``.
            When a string is given, backend defaults to ``"transformers"``, device
            to ``"cpu"``, and the default :class:`BlockLevelSelector` is used.
        vision_db : Any, optional
            Image source ``{stim_id: image}``.  Uses the already-attached db
            if ``None``.
        batch_size : int
            Images per forward pass.  Default 32.
        overwrite : bool
            Replace existing records for the same ``(model_id, module_name)`` key.

        Raises
        ------
        RuntimeError
            If no image source is available.
        """
        from vneurotk.vision.model.base import VisionModel

        if isinstance(model, str):
            model = VisionModel(model)

        if vision_db is not None:
            if self._vision_db is not None:
                logger.warning("extract_from: replacing existing image database.")
            self._vision_db = vision_db

        if self._vision_db is None:
            raise RuntimeError("No image source available. Pass vision_db= to extract_from().")

        self._run_extraction(model, self._vision_db, batch_size=batch_size, overwrite=overwrite)

    def _run_extraction(
        self,
        model: Any,
        stimuli: Any,
        batch_size: int,
        overwrite: bool,
    ) -> None:
        """Internal: run extraction via *model* and store results."""
        images = self._relevant_images(stimuli)
        if not images:
            logger.warning("extract_from: no images found for any output_order ID — skipping.")
            return

        pending = self._pending_modules(model.model_id, model.module_names, overwrite)
        if not pending:
            logger.info(
                "extract_from: all {} modules already extracted for '{}', skipping.",
                len(model.module_names),
                model.model_id,
            )
            return

        if len(pending) < len(model.module_names):
            logger.info(
                "extract_from: {}/{} modules missing for '{}', extracting subset.",
                len(pending),
                len(model.module_names),
                model.model_id,
            )
        else:
            logger.info(
                "extract_from: {} stimuli, {} modules, batch_size={}",
                len(images),
                len(model.module_names),
                batch_size,
            )
        visual_representations = model.extract_for_modules(images, pending, batch_size=batch_size)

        self._store_records(visual_representations, overwrite=overwrite)
        logger.info("extract_from done: {} modules stored.", len(self._records))

    def _pending_modules(self, model_id: str, module_names: list[str], overwrite: bool) -> list[str]:
        if overwrite or not self._records:
            return list(module_names)
        return [m for m in module_names if (model_id, m) not in self._records]

    def _relevant_images(self, stimuli: Any) -> dict:
        unique_ids: list = list(dict.fromkeys(self._output_order.tolist()))
        return {sid: stimuli[sid] for sid in unique_ids if sid in stimuli}

    # ------------------------------------------------------------------
    # HDF5 persistence
    # ------------------------------------------------------------------

    def dump(self, f: h5py.File, group_name: str = "vision_store") -> None:
        """Serialize stored records to an HDF5 group.

        Parameters
        ----------
        f : h5py.File
        group_name : str
        """
        import h5py as _h5py

        vsg = f.create_group(group_name)
        for i, vr in enumerate(self._records.values()):
            grp = vsg.create_group(str(i))
            grp.attrs["model"] = vr.model
            grp.attrs["module_name"] = vr.module_name
            grp.attrs["module_type"] = vr.module_type
            sid = vr.stim_ids
            if sid and isinstance(sid[0], str):
                grp.create_dataset("stim_ids", data=np.array(sid, dtype=_h5py.string_dtype()))
            else:
                grp.create_dataset("stim_ids", data=np.array(sid))
            grp.create_dataset("array", data=vr.array)

    @classmethod
    def from_h5(
        cls,
        f: h5py.File,
        output_order: np.ndarray,
        group_name: str = "vision_store",
        vision_db: Any = None,
        fpath: Any = None,
    ) -> VisionData:
        """Reconstruct from an HDF5 group.

        Parameters
        ----------
        f : h5py.File
        output_order : np.ndarray
        group_name : str
        vision_db : Any, optional
        fpath : Path or str, optional
            File path used to create lazy array loaders.  When provided, activation
            arrays are not loaded into memory until first access.

        Returns
        -------
        VisionData
        """
        from pathlib import Path

        vd = cls(output_order, vision_db=vision_db)
        if group_name not in f:
            return vd
        vsg = f[group_name]
        records = []
        for key in sorted(vsg.keys(), key=lambda x: int(x)):
            grp = vsg[key]
            raw_sids = grp["stim_ids"][:]
            if raw_sids.dtype.kind in ("S", "O"):
                sids: list = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw_sids]
            else:
                sids = raw_sids.tolist()

            if fpath is not None:
                arr_shape = tuple(grp["array"].shape)
                _p, _g, _k = str(Path(fpath)), group_name, key

                def _loader(p: str = _p, g: str = _g, k: str = _k) -> np.ndarray:
                    import h5py as _h5py

                    with _h5py.File(p, "r") as _f:
                        return _f[g][k]["array"][:]

                records.append(
                    VisualRepresentation(
                        model=str(grp.attrs["model"]),
                        module_name=str(grp.attrs["module_name"]),
                        module_type=str(grp.attrs["module_type"]),
                        stim_ids=sids,
                        array_loader=_loader,
                        shape=arr_shape,
                    )
                )
            else:
                records.append(
                    VisualRepresentation(
                        model=str(grp.attrs["model"]),
                        module_name=str(grp.attrs["module_name"]),
                        module_type=str(grp.attrs["module_type"]),
                        stim_ids=sids,
                        array=grp["array"][:],
                    )
                )
        if records:
            vd._store_records(VisualRepresentations(records), overwrite=False)
        return vd

    # ------------------------------------------------------------------
    # View layer — output_order-aligned indexing
    # ------------------------------------------------------------------

    @property
    def output_order(self) -> np.ndarray:
        """Sequence of stimulus IDs defining the desired output ordering."""
        return self._output_order

    @output_order.setter
    def output_order(self, value: np.ndarray) -> None:
        self._output_order = np.asarray(value)
        self._align_cache.clear()

    def by_module(self, name: str, model: str | None = None) -> np.ndarray:
        """Return the output-order-aligned activation array for *name*.

        Parameters
        ----------
        name : str
        model : str or None
            Disambiguates when multiple models share the same module name.

        Returns
        -------
        np.ndarray
            Shape ``(n_output_order_items, ...)``.
        """
        if model is not None:
            vr = self._records.get((model, name))
            if vr is None:
                raise KeyError(f"Module {name!r} for model {model!r} not found.")
            return self._align_vr(vr)

        matches = [(k[0], vr) for k, vr in self._records.items() if k[1] == name]
        if not matches:
            raise KeyError(f"Module {name!r} not found.")
        if len(matches) > 1:
            models = [m for m, _ in matches]
            raise KeyError(f"Module {name!r} found in {len(matches)} models: {models}. Specify model= to disambiguate.")
        return self._align_vr(matches[0][1])

    def __getitem__(self, key: pd.Series | np.ndarray | str | int) -> np.ndarray | VisualRepresentations:
        """Output-order-aligned indexing.

        Parameters
        ----------
        key : pd.Series, np.ndarray of bool, str, or int
            - ``str``  → look up by module name, return aligned ndarray.
            - ``int``  → positional index, return aligned ndarray.
            - 1-D bool mask → single match returns ndarray; multiple matches
              return :class:`VisualRepresentations` with arrays aligned to
              ``output_order``.

        Raises
        ------
        TypeError
            If *key* is a 0-d boolean scalar (e.g. ``'col' == value``).
        """
        records = list(self._records.values())

        if isinstance(key, str):
            return self.by_module(key)
        if isinstance(key, (int, np.integer)):
            return self._align_vr(records[int(key)])

        arr = np.asarray(key)
        if arr.ndim == 0:
            raise TypeError(
                "Boolean index must be a 1-D array aligned to meta rows. "
                "Got a scalar — did you mean meta['col'] == value instead of 'col' == value?"
            )
        bool_mask = arr.astype(bool)
        filtered = [vr for vr, keep in zip(records, bool_mask, strict=True) if keep]
        if len(filtered) == 1:
            return self._align_vr(filtered[0])

        aligned = [
            VisualRepresentation(
                model=vr.model,
                module_name=vr.module_name,
                module_type=vr.module_type,
                stim_ids=list(self._output_order),
                array=self._align_vr(vr),
            )
            for vr in filtered
        ]
        return VisualRepresentations(aligned)

    def _align_vr(self, vr: VisualRepresentation) -> np.ndarray:
        """Re-index *vr*'s array to match ``output_order``.

        Uses ``vr._id_to_idx`` (built once at construction time) rather than
        rebuilding the mapping on every cache miss.
        """
        key = (vr.model, vr.module_name)
        if key not in self._align_cache:
            indices = np.array([vr._id_to_idx[sid] for sid in self._output_order])
            self._align_cache[key] = vr.array[indices]
        return self._align_cache[key]

    def __repr__(self) -> str:
        return f"VisionData({len(self._output_order)} stimuli x {len(self._records)} modules)"
