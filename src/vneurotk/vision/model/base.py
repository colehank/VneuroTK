"""VisionModel — unified entry point for DNN feature extraction."""

from __future__ import annotations

import contextlib
import importlib
from typing import Any

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

from vneurotk.vision.model.backend.base import BaseBackend
from vneurotk.vision.model.selector import BlockLevelSelector, CustomSelector, ModuleSelector
from vneurotk.vision.representation.visual_representations import (
    VisualRepresentation,
    VisualRepresentations,
)

__all__ = ["VisionModel", "print_modules"]

DEFAULT_BATCH_SIZE = 32

_BACKEND_REGISTRY: dict[str, str | type[BaseBackend]] = {
    "timm": "vneurotk.vision.model.backend.timm_backend:TimmBackend",
    "transformers": "vneurotk.vision.model.backend.transformers_backend:TransformersBackend",
    "thingsvision": "vneurotk.vision.model.backend.thingsvision_backend:ThingsVisionBackend",
}


class VisionModel:
    """Unified interface for extracting DNN activations from images.

    Composes a :class:`~vneurotk.vision.model.backend.base.BaseBackend`
    and a :class:`~vneurotk.vision.model.selector.ModuleSelector`.
    Activations are returned as-is; any further processing (pooling,
    embedding, etc.) is left to the user.

    Parameters
    ----------
    model_id : str
        Model identifier passed directly to the backend, e.g.
        ``"facebook/dinov2-base"`` (transformers) or ``"resnet50"`` (timm).
    backend : str
        Backend to use: ``"transformers"`` (default), ``"timm"``, or
        ``"thingsvision"``.
    selector : ModuleSelector or None
        Layer selection strategy.  Defaults to
        :class:`~vneurotk.vision.model.selector.BlockLevelSelector`.
    device : str
        Inference device (default ``"cpu"``).
    pretrained : bool
        Load pretrained weights (default ``True``).
    """

    def __init__(
        self,
        model_id: str,
        backend: str = "transformers",
        selector: ModuleSelector | None = None,
        device: str = "cpu",
        pretrained: bool = True,
    ) -> None:
        self._selector = selector or BlockLevelSelector()

        self._backend = self._build_backend(backend, device)
        self._backend.load(model_id, pretrained=pretrained)

        self._bind_selector()

        logger.info(
            "VisionModel ready | model={} | backend={} | modules={}",
            model_id,
            backend,
            len(self._module_names),
        )

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Any,
        backend: BaseBackend,
        selector: ModuleSelector | None = None,
    ) -> VisionModel:
        """Build a VisionModel from an already-loaded model.

        Parameters
        ----------
        model : nn.Module
            Pre-loaded PyTorch model.
        backend : BaseBackend
            Backend instance with *model* already assigned.
        selector : ModuleSelector or None
            Defaults to :class:`BlockLevelSelector`.

        Returns
        -------
        VisionModel
        """
        inst = object.__new__(cls)
        inst._selector = selector or BlockLevelSelector()
        inst._backend = backend
        inst._backend.model = model

        inst._bind_selector()
        return inst

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _bind_selector(self) -> None:
        """Select modules, register hooks, and build the module-type map.

        Called by ``__init__``, ``from_model``, and ``set_selector`` so the
        wiring logic lives in exactly one place.
        """
        module_names = self._selector.select(self._backend.enumerate_modules())
        self._backend.register_hooks(module_names)
        self._module_names = module_names
        self._module_type_map: dict[str, str] = {m.name: m.module_type for m in self._backend.enumerate_modules()}

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_images(image: Any) -> dict:
        """Normalise any supported input type to a ``{stim_id: image}`` dict.

        Parameters
        ----------
        image : dict or single image
            ``dict`` → returned as-is.
            Single image (PIL, ndarray, str, Path) → ``{0: image}``.

        Returns
        -------
        dict

        Raises
        ------
        TypeError
            If *image* is a ``list``.  Use ``{0: img0, 1: img1, …}`` or any
            mapping with meaningful stim IDs instead — a list would silently
            assign integer stim IDs that won't align with
            ``BaseData.trial_stim_ids``.
        """
        if isinstance(image, list):
            raise TypeError(
                "batch input requires a dict mapping stim_id → image; "
                "got a list.  Use {0: img0, 1: img1, …} for integer stim IDs, "
                "or {name: img, …} for named stimuli."
            )
        if isinstance(image, dict):
            return image
        return {0: image}

    def extract(
        self,
        image: Any,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = True,
    ) -> VisualRepresentations:
        """Extract DNN activations for one image or a collection of stimuli.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or str or Path or dict
            Single image → ``n_sample=1``, ``stim_id=0``.
            ``dict`` mapping stim_ids to images → ``n_sample=len(dict)``.
            String / ``pathlib.Path`` values are opened automatically.
            ``list`` is **not** accepted — use a ``dict`` with explicit
            stim IDs to preserve alignment with ``BaseData.trial_stim_ids``.
        batch_size : int
            Number of images per GPU forward pass.  Default 32.
            Ignored for single-image input.
        show_progress : bool
            Display a tqdm progress bar over batches.  Automatically
            suppressed for single-image input.  Default ``True``.

        Returns
        -------
        VisualRepresentations
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        images = self._normalize_images(image)
        is_single = not isinstance(image, dict)
        return self.extract_for_modules(
            images,
            self._module_names,
            batch_size=1 if is_single else batch_size,
            show_progress=False if is_single else show_progress,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Model identifier (e.g. ``'facebook/dinov2-base'``, ``'resnet50'``)."""
        return self._backend.get_model_meta().model_id

    @property
    def module_names(self) -> list[str]:
        """Names of all currently hooked modules."""
        return list(self._module_names)

    @property
    def module_list(self) -> list:
        """All modules available in the loaded model.

        Returns
        -------
        list[ModuleInfo]
            One entry per named module, ordered as ``model.named_modules()``.
            Each entry exposes ``.name``, ``.module_type``, ``.depth``,
            and ``.n_params``.
        """
        return self._backend.enumerate_modules()

    # ------------------------------------------------------------------
    # Inspection & reconfiguration
    # ------------------------------------------------------------------

    def print_modules(self, max_depth: int | None = None, console: Any = None) -> None:
        """Print a tree-style summary of all model modules.

        Parameters
        ----------
        max_depth : int or None
            Maximum nesting depth to display.  ``None`` shows all levels.
        console : rich.console.Console or None
            Rich console to use for output.  Pass ``Console(record=True)`` to
            capture the output for SVG / HTML export.
        """
        _print_modules(self.module_list, max_depth=max_depth, console=console)

    @staticmethod
    def _filter_modules(
        module_list: list,
        types: set[str],
        names: set[str],
    ) -> list:
        """Filter *module_list* by module type and/or exact name.

        Returns all modules whose ``module_type`` is in *types* OR whose
        ``name`` is in *names*.  Both sets may be empty; an entry matches if
        it satisfies at least one non-empty criterion.

        Parameters
        ----------
        module_list : list[ModuleInfo]
            Full module list from :attr:`module_list`.
        types : set[str]
            ``module_type`` values to include.
        names : set[str]
            Exact ``name`` values to include.

        Returns
        -------
        list[ModuleInfo]
            Ordered subset of *module_list* (preserves original order).
        """
        return [m for m in module_list if (types and m.module_type in types) or (names and m.name in names)]

    def set_selector(
        self,
        selector: ModuleSelector | list | None = None,
        *,
        module_type: str | list[str] | None = None,
        module_name: str | list[str] | None = None,
    ) -> None:
        """Replace the layer selector and re-register hooks.

        Accepts the following forms (combinable):

        - ``set_selector(BlockLevelSelector())`` — explicit selector object
        - ``set_selector(["layer.0", "layer.1"])`` — list of module names / ModuleInfo
        - ``set_selector(module_type="Dinov2Layer")`` — all modules of that type
        - ``set_selector(module_type=["Dinov2Layer", "LayerNorm"])`` — multiple types
        - ``set_selector(module_name="encoder.layer.3")`` — single module by name
        - ``set_selector(module_name=["enc.0", "enc.6"])`` — multiple names
        - ``set_selector(module_type="Dinov2Layer", module_name="layernorm")``
          — union of both filters

        *selector* and (*module_type* / *module_name*) are mutually exclusive.

        Parameters
        ----------
        selector : ModuleSelector, list, or None
            Explicit selector object or list of module names / ModuleInfo objects.
        module_type : str, list[str], or None
            Hook all modules whose ``module_type`` is in this set.
        module_name : str, list[str], or None
            Hook modules whose ``name`` is in this set (exact match).

        Raises
        ------
        ValueError
            If no arguments are supplied, *selector* is combined with filters,
            or the resulting module list is empty.
        """
        using_filters = module_type is not None or module_name is not None
        if selector is None and not using_filters:
            raise ValueError("Provide selector, module_type, or module_name.")
        if selector is not None and using_filters:
            raise ValueError("selector is mutually exclusive with module_type/module_name.")

        if using_filters:
            types = {module_type} if isinstance(module_type, str) else set(module_type or [])
            names = {module_name} if isinstance(module_name, str) else set(module_name or [])
            matched = self._filter_modules(self.module_list, types, names)
            if not matched:
                raise ValueError(
                    f"No modules matched module_type={module_type!r}, "
                    f"module_name={module_name!r}. "
                    f"Available types: {sorted({m.module_type for m in self.module_list})}"
                )
            selector = CustomSelector(matched)
        elif isinstance(selector, list):
            selector = CustomSelector(selector)

        assert selector is not None
        self._selector = selector
        self._bind_selector()
        logger.info("Selector updated | modules={}", len(self._module_names))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def extract_for_modules(
        self,
        images: dict,
        module_names: list[str],
        batch_size: int,
        show_progress: bool = True,
    ) -> VisualRepresentations:
        """Extract activations for a subset of modules without altering state.

        Temporarily re-registers hooks for *module_names*, runs extraction,
        then restores the original hook configuration.

        Parameters
        ----------
        images : dict
            ``{stim_id: image}`` mapping.
        module_names : list[str]
            Subset of module names to extract.
        batch_size : int
            Images per forward pass.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        VisualRepresentations
        """
        with self._hooked_for(module_names):
            return self._extract_batch(images, batch_size=batch_size, show_progress=show_progress)

    @contextlib.contextmanager
    def _hooked_for(self, module_names: list[str]):
        """Temporarily re-register hooks for *module_names*, then restore."""
        original = self._module_names
        self._backend.register_hooks(module_names)
        self._module_names = module_names
        try:
            yield
        finally:
            self._backend.register_hooks(original)
            self._module_names = original

    def _build_vr_list(
        self,
        stim_ids: list,
        features: dict[str, np.ndarray],
    ) -> list[VisualRepresentation]:
        """Assemble VisualRepresentation objects from batched features.

        Pairs each layer's activation array with *stim_ids*, using the
        current model metadata and module-type map.  The caller guarantees
        that rows in every array are aligned to *stim_ids* in the same order.

        Parameters
        ----------
        stim_ids : list
            Ordered stimulus IDs produced by :meth:`_prepare_images`.
        features : dict[str, np.ndarray]
            Layer name → activation array of shape ``(n_stim, ...)``.

        Returns
        -------
        list[VisualRepresentation]
        """
        model_meta = self._backend.get_model_meta()
        return [
            VisualRepresentation(
                model=model_meta.model_id,
                module_name=layer,
                module_type=self._module_type_map.get(layer, ""),
                stim_ids=stim_ids,
                array=arr,
            )
            for layer, arr in features.items()
        ]

    def _extract_batch(self, images: dict, batch_size: int, show_progress: bool = False) -> VisualRepresentations:
        stim_ids, loaded = self._prepare_images(images)
        features = self._run_batches(loaded, batch_size, show_progress)
        vr_list = self._build_vr_list(stim_ids, features)
        logger.info(
            "Extracted | n={} | batch_size={} | modules={}",
            len(stim_ids),
            batch_size,
            len(vr_list),
        )
        return VisualRepresentations(vr_list)

    def _run_batches(
        self,
        loaded: list,
        batch_size: int,
        show_progress: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run batched forward passes and concatenate activations per layer.

        Parameters
        ----------
        loaded : list
            Pre-loaded images (PIL / ndarray / Tensor).
        batch_size : int
            Images per forward pass.
        show_progress : bool
            Show tqdm progress bar.

        Returns
        -------
        dict[str, np.ndarray]
            Layer name → concatenated activation array of shape ``(n_images, ...)``.
        """
        all_features: dict[str, list[np.ndarray]] = {}
        n_chunks = (len(loaded) + batch_size - 1) // batch_size
        iterator = tqdm(
            range(0, len(loaded), batch_size),
            total=n_chunks,
            desc="VisionModel",
            unit="batch",
            disable=not show_progress or n_chunks <= 1,
        )
        for start in iterator:
            chunk = loaded[start : start + batch_size]
            chunk_feats = self._forward_chunk(chunk)
            for layer, arr in chunk_feats.items():
                all_features.setdefault(layer, []).append(arr)
        return {layer: np.concatenate(arrs, axis=0) for layer, arrs in all_features.items()}

    @staticmethod
    def _prepare_images(images: dict) -> tuple[list[Any], list[Any]]:
        """Resolve image sources and return (stim_ids, loaded_images).

        Handles string and Path entries by opening them with PIL.
        Already-loaded images (PIL, ndarray, Tensor) pass through unchanged.

        Parameters
        ----------
        images : dict
            ``{stim_id: image_or_path}`` mapping.

        Returns
        -------
        tuple[list, list]
            ``(stim_ids, loaded_images)`` in the same order as *images*.
        """
        from pathlib import Path

        from PIL import Image as PILImage

        stim_ids = list(images.keys())
        loaded: list[Any] = []
        for sid in stim_ids:
            img = images[sid]
            if isinstance(img, (str, Path)):
                img = PILImage.open(img).convert("RGB")
            loaded.append(img)
        return stim_ids, loaded

    def _forward_chunk(self, images: list[Any]) -> dict[str, np.ndarray]:
        """Run one forward pass and return batched activations per layer.

        Parameters
        ----------
        images : list
            Batch of images (PIL / ndarray / str / Path).

        Returns
        -------
        dict[str, np.ndarray]
            Layer name → array of shape ``(B, ...)``.
        """
        inputs = self._backend.preprocess(images)
        with self._backend.collecting() as collect:
            self._backend.forward(inputs)
            activations = collect()
        return {name: act.numpy() for name, act in activations.items()}

    @classmethod
    def register_backend(cls, name: str, backend_cls: type[BaseBackend]) -> None:
        """Register a custom backend class under *name*.

        Parameters
        ----------
        name : str
            Key used in the ``backend=`` argument of :class:`VisionModel`.
        backend_cls : type[BaseBackend]
            Backend class to register.  Must be a concrete subclass of
            :class:`~vneurotk.vision.model.backend.base.BaseBackend`.
        """
        _BACKEND_REGISTRY[name] = backend_cls

    @staticmethod
    def _build_backend(backend: str, device: str) -> BaseBackend:
        if backend not in _BACKEND_REGISTRY:
            raise ValueError(f"Unknown backend {backend!r}. Available: {sorted(_BACKEND_REGISTRY)}")
        entry = _BACKEND_REGISTRY[backend]
        if isinstance(entry, str):
            module_path, class_name = entry.rsplit(":", 1)
            BackendClass: type[BaseBackend] = getattr(importlib.import_module(module_path), class_name)
        else:
            BackendClass = entry
        return BackendClass(device=device)


def print_modules(layers: list, max_depth: int | None = None, console: Any = None) -> None:
    """Print a tree-style summary of model layers.

    Parameters
    ----------
    layers : list[ModuleInfo]
        Module list returned by :attr:`VisionModel.module_list`.
    max_depth : int or None
        Maximum nesting depth to display.  ``None`` shows all levels.
    console : rich.console.Console or None
        Rich console to use for output.  Pass ``Console(record=True)`` to
        capture the output for SVG / HTML export.
    """
    _print_modules(layers, max_depth=max_depth, console=console)


def _print_modules(layers: list, max_depth: int | None = None, console: Any = None) -> None:
    from rich.console import Console as _Console
    from rich.table import Table
    from rich.text import Text

    if console is None:
        console = _Console()

    _COLORS = [
        "red",
        "green",
        "yellow",
        "blue",
        "magenta",
        "cyan",
        "bright_red",
        "bright_green",
        "bright_yellow",
        "bright_blue",
    ]
    type_color_map: dict[str, str] = {}

    def get_color(module_type: str) -> str:
        if module_type not in type_color_map:
            type_color_map[module_type] = _COLORS[len(type_color_map) % len(_COLORS)]
        return type_color_map[module_type]

    filtered = [lay for lay in layers if max_depth is None or lay.depth <= max_depth]
    max_params = max((lay.n_params for lay in filtered), default=1) or 1
    BAR_WIDTH = 8

    def is_last(idx: int) -> bool:
        depth = filtered[idx].depth
        for j in range(idx + 1, len(filtered)):
            if filtered[j].depth < depth:
                return True
            if filtered[j].depth == depth:
                return False
        return True

    def get_prefix(idx: int) -> str:
        depth = filtered[idx].depth
        parts = []
        for d in range(1, depth):
            has_more = False
            for j in range(idx + 1, len(filtered)):
                if filtered[j].depth == d:
                    has_more = True
                    break
                if filtered[j].depth < d:
                    break
            parts.append("│  " if has_more else "   ")
        parts.append("└─ " if is_last(idx) else "├─ ")
        return "".join(parts)

    def fmt_size(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1e6:.2f}M"
        if n >= 1_000:
            return f"{n / 1e3:.1f}K"
        if n > 0:
            return str(n)
        return "-"

    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Module  (⊡ container   • leaf)", min_width=48, no_wrap=True)
    table.add_column("Type", min_width=20)
    table.add_column("Param", justify="right")

    for i, layer in enumerate(filtered):
        color = get_color(layer.module_type)
        short_name = layer.name.split(".")[-1]
        prefix = get_prefix(i)
        label = "• " if layer.is_leaf else "⊡ "

        name_text = Text(label + prefix + short_name)
        if layer.is_leaf and layer.param_shapes:
            shapes = "  ".join(f"{k}:{list(v)}" for k, v in layer.param_shapes.items())
            name_text.append(f"  {shapes}", style="dim")

        bar_len = int(layer.n_params / max_params * BAR_WIDTH)
        param_text = Text()
        param_text.append("█" * bar_len + "░" * (BAR_WIDTH - bar_len), style=color)
        param_text.append(f" {fmt_size(layer.n_params):>6}")

        table.add_row(name_text, Text(layer.module_type, style=color), param_text)

    console.print()
    console.print(table)

    # type legend
    legend_items = [f"[{color}]██[/{color}] {mt}" for mt, color in type_color_map.items()]
    console.print("\nType legend:")
    console.print("  " + "   ".join(legend_items))
