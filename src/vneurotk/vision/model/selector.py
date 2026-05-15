"""Layer selection strategies.

Three concrete selectors are provided:

- :class:`BlockLevelSelector` — major blocks (ViT blocks, ResNet layers)
- :class:`AllLeafSelector`    — all leaf modules (no children)
- :class:`CustomSelector`     — explicit user-supplied list
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn  # type: ignore

from vneurotk.vision.meta import ModuleInfo

__all__ = [
    "ModuleSelector",
    "BlockLevelSelector",
    "AllLeafSelector",
    "CustomSelector",
]


class ModuleSelector(ABC):
    """Abstract base class for layer selection strategies.

    Subclasses implement :meth:`select`, which receives a list of
    :class:`~vneurotk.vision.meta.ModuleInfo` objects and returns an
    ordered list of module name strings to hook.
    """

    @abstractmethod
    def select(self, modules: list[ModuleInfo]) -> list[str]:
        """Return layer names to hook.

        Parameters
        ----------
        modules : list[ModuleInfo]
            All named modules enumerated by the backend, as returned by
            :meth:`~vneurotk.vision.model.backend.base.BaseBackend.enumerate_modules`.

        Returns
        -------
        list[str]
            Ordered module names to register hooks on.
        """


class BlockLevelSelector(ModuleSelector):
    """Select major block-level modules appropriate for the architecture.

    Uses regex patterns matched against module names.  Architecture
    patterns are tried in order; the first match wins.  Falls back to
    top-level children (depth == 1) if no pattern matches.

    Parameters
    ----------
    max_depth : int
        Maximum nesting depth to include (default 2).  Controls how
        deeply nested sub-blocks are included.
    include_patterns : list[str] or None
        Additional regex patterns to include alongside defaults.
    """

    _ARCH_PATTERNS: list[tuple[str, int]] = [
        (r"^blocks\.\d+$", 2),  # timm ViT
        (r"^encoder\.layers\.\d+$", 3),  # HF ViT (plural)
        (r"^encoder\.layer\.\d+$", 3),  # HF DINOv2 (singular)
        (r"^model\.layer\.\d+$", 3),  # HF DINOv3
        (r"^layer\d+\.\d+$", 3),  # ResNet
        (r"^features\.\d+$", 2),  # VGG / EfficientNet
        (r"^stages\.\d+$", 2),  # ConvNeXt
        (r"^layers\.\d+$", 2),  # Swin / generic
        (r"^vision_model\.encoder\.layers\.\d+$", 4),  # SigLIP / SigLIP2
    ]

    def __init__(
        self,
        max_depth: int = 2,
        include_patterns: list[str] | None = None,
        arch_patterns: list[tuple[str, int]] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self._extra = [re.compile(p) for p in (include_patterns or [])]
        raw = arch_patterns if arch_patterns is not None else self._ARCH_PATTERNS
        self._compiled_patterns: list[tuple[re.Pattern, int]] = [(re.compile(p), d) for p, d in raw]

    @classmethod
    def default_patterns(cls) -> list[tuple[str, int]]:
        """Return a copy of the built-in architecture patterns.

        Returns
        -------
        list[tuple[str, int]]
            Each element is ``(regex_pattern, max_depth)``.
            Mutating the returned list does not affect the class default.
        """
        return list(cls._ARCH_PATTERNS)

    @staticmethod
    def _module_depth(name: str) -> int:
        """Return the nesting depth of a module given its dotted name.

        Parameters
        ----------
        name : str
            Module name as produced by ``model.named_modules()``,
            e.g. ``"encoder.layer.3"`` → depth 3.

        Returns
        -------
        int
            ``name.count(".") + 1``.  Empty string returns ``0``.
        """
        return name.count(".") + 1 if name else 0

    def select(self, modules: list[ModuleInfo]) -> list[str]:
        """Select block-level layers from *modules*.

        Parameters
        ----------
        modules : list[ModuleInfo]

        Returns
        -------
        list[str]
        """
        selected: list[str] = []

        for m in modules:
            matched = any(pat.match(m.name) and m.depth <= max_d for pat, max_d in self._compiled_patterns)
            if not matched:
                matched = any(pat.search(m.name) for pat in self._extra)

            if matched:
                selected.append(m.name)

        if not selected:
            selected = [m.name for m in modules if m.depth == 1]

        return selected


class AllLeafSelector(ModuleSelector):
    """Select all leaf modules (modules with no children).

    Parameters
    ----------
    exclude_types : tuple[type, ...] or None
        Module types to skip.  Defaults to activation and regularization
        layers that carry no representational content.
    """

    _DEFAULT_EXCLUDE = (
        nn.Dropout,
        nn.Identity,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Softmax,
    )

    def __init__(self, exclude_types: tuple | None = None) -> None:
        raw = exclude_types if exclude_types is not None else self._DEFAULT_EXCLUDE
        self._exclude_names: frozenset[str] = frozenset(t.__name__ for t in raw)

    def select(self, modules: list[ModuleInfo]) -> list[str]:
        """Return names of all non-excluded leaf modules.

        Parameters
        ----------
        modules : list[ModuleInfo]

        Returns
        -------
        list[str]
        """
        return [m.name for m in modules if m.is_leaf and m.module_type not in self._exclude_names]


class CustomSelector(ModuleSelector):
    """Use an explicit user-supplied list of layer names.

    Parameters
    ----------
    layer_names : list[str] or list[ModuleInfo]
        Exact module names as they appear in ``model.named_modules()``,
        or :class:`~vneurotk.vision.meta.ModuleInfo`
        objects as returned by :attr:`VisionModel.module_list`.

    Raises
    ------
    ValueError
        During :meth:`select` if any name is not found in the module list.
    """

    def __init__(self, layer_names: list) -> None:
        self.layer_names = [item.name if hasattr(item, "name") else item for item in layer_names]

    def select(self, modules: list[ModuleInfo]) -> list[str]:
        """Validate and return the configured layer names.

        Parameters
        ----------
        modules : list[ModuleInfo]

        Returns
        -------
        list[str]

        Raises
        ------
        ValueError
            If any layer name is absent from *modules*.
        """
        available = {m.name for m in modules}
        missing = [n for n in self.layer_names if n not in available]
        if missing:
            raise ValueError(
                f"Layer(s) not found in model: {missing}. Inspect available names with model.named_modules()."
            )
        return list(self.layer_names)
