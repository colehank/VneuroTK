"""Abstract base class for vision model backends.

A backend encapsulates:
1. Loading a model from its native library.
2. Preprocessing images into the required input format.
3. Running the forward pass.
4. Enumerating available layers.
5. Managing forward hooks to capture intermediate activations.
"""

from __future__ import annotations

import contextlib
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from vneurotk.vision.meta import ModelInfo, ModuleInfo

__all__ = ["BaseBackend", "ModuleInfo"]


class BaseBackend(ABC):
    """Abstract base for all feature-extraction backends.

    Subclasses implement :meth:`load`, :meth:`preprocess`,
    :meth:`forward`, and :meth:`get_model_meta`.  Hook management and
    module enumeration are provided here and shared.

    Parameters
    ----------
    device : str or torch.device
        Device for inference (default ``"cpu"``).
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self._hooks: list[Any] = []
        self._activations: OrderedDict[str, Tensor] = OrderedDict()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load the model into memory.

        Parameters
        ----------
        model_name : str
            Library-specific model identifier.
        pretrained : bool
            Whether to load pretrained weights.
        """

    @abstractmethod
    def preprocess(self, image: Any) -> dict[str, Any]:
        """Convert a PIL Image or Tensor to the model's input format.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or Tensor
            Raw input image.

        Returns
        -------
        dict[str, Any]
            Keyword arguments ready for ``forward()``.
        """

    @abstractmethod
    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run a forward pass and return the raw model output.

        Parameters
        ----------
        inputs : dict[str, Any]
            Preprocessed inputs from :meth:`preprocess`.

        Returns
        -------
        Any
            Raw model output (Tensor or HuggingFace ModelOutput).
        """

    def enumerate_modules(self) -> list[ModuleInfo]:
        """Return metadata for all named modules in :attr:`hookable_model`.

        Returns
        -------
        list[ModuleInfo]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        result = []
        for name, module in self.hookable_model.named_modules():
            if not name:
                continue
            result.append(
                ModuleInfo(
                    name=name,
                    module_type=type(module).__name__,
                    depth=name.count(".") + 1,
                    n_params=sum(p.numel() for p in module.parameters()),
                    is_leaf=len(list(module.children())) == 0,
                    param_shapes={n: tuple(p.shape) for n, p in module.named_parameters(recurse=False)},
                )
            )
        return result

    @abstractmethod
    def get_model_meta(self) -> ModelInfo:
        """Return model-level metadata.

        Returns
        -------
        ModelInfo
        """

    # ------------------------------------------------------------------
    # Image loading (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_images(image: Any) -> list:
        """Convert any supported input to a list of PIL Images.

        Parameters
        ----------
        image : PIL.Image.Image, np.ndarray, str, Path, or list of these
            Single image or batch.  String / ``Path`` values are opened from
            disk.  ``np.ndarray`` values are converted via
            ``PIL.Image.fromarray``.  An existing ``PIL.Image`` is passed
            through unchanged.

        Returns
        -------
        list[PIL.Image.Image]
        """
        from pathlib import Path

        import numpy as np
        from PIL import Image as PILImage

        images = image if isinstance(image, list) else [image]
        result = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = PILImage.open(img)
            elif isinstance(img, np.ndarray):
                img = PILImage.fromarray(img)
            result.append(img.convert("RGB"))
        return result

    # ------------------------------------------------------------------
    # Hookable model (overridable)
    # ------------------------------------------------------------------

    @property
    def hookable_model(self) -> Any:
        """The sub-model to attach hooks to.

        Returns :attr:`model` by default.  Override in subclasses that wrap
        a larger model container (e.g. a CLIP model whose hooks should be
        attached to its vision encoder rather than the top-level object).
        """
        return self.model

    # ------------------------------------------------------------------
    # Hook management (shared implementation)
    # ------------------------------------------------------------------

    def register_hooks(self, layer_names: list[str]) -> None:
        """Attach forward hooks to the specified layers.

        Hooks store a detached CPU copy of each layer output in
        :attr:`_activations`.  Call :meth:`remove_hooks` when done.

        Parameters
        ----------
        layer_names : list[str]
            Module names to hook.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        ValueError
            If any name is not found in the model.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before registering hooks.")

        self.remove_hooks()
        self._activations.clear()

        named = dict(self.hookable_model.named_modules())
        missing = [n for n in layer_names if n not in named]
        if missing:
            raise ValueError(f"Layer(s) not found in model: {missing}")

        for name in layer_names:
            module = named[name]

            def _hook(mod: nn.Module, inp: Any, output: Any, _n: str = name) -> None:  # noqa: ARG001
                act = output[0] if isinstance(output, tuple) else output
                self._activations[_n] = act.detach().cpu()

            handle = module.register_forward_hook(_hook)
            self._hooks.append(handle)

        logger.debug("Registered {} hooks", len(layer_names))

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all hooks")

    def collect_activations(self) -> OrderedDict[str, Tensor]:
        """Return captured activations and clear the buffer.

        Returns
        -------
        OrderedDict[str, Tensor]
        """
        result = OrderedDict(self._activations)
        self._activations.clear()
        return result

    @contextlib.contextmanager
    def collecting(self):
        """Context manager that clears the activation buffer on enter and exit.

        Yields the :meth:`collect_activations` callable so the caller can
        read activations after the forward pass without worrying about
        buffer state leaking between calls.

        Example
        -------
        >>> with backend.collecting() as collect:
        ...     backend.forward(inputs)
        ...     acts = collect()
        """
        self._activations.clear()
        try:
            yield self.collect_activations
        finally:
            self._activations.clear()

    # ------------------------------------------------------------------
    # Layer name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_layer_name(raw_name: str) -> str:
        """Convert a raw module name to a normalized identifier.

        Parameters
        ----------
        raw_name : str
            E.g. ``"blocks.11.attn.proj"``.

        Returns
        -------
        str
            E.g. ``"blocks_11_attn_proj"``.
        """
        name = raw_name.replace(".", "_")
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _move_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move all Tensor values in *inputs* to :attr:`device`."""
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
