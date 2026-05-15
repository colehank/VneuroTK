"""vneurotk.vision — DNN vision representation module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vneurotk.vision.image_source import ImageSource
from vneurotk.vision.meta import ModelInfo, ModuleInfo
from vneurotk.vision.representation.visual_representations import (
    VisualRepresentation,
    VisualRepresentations,
)

# Model symbols require torch — imported lazily to allow torch-free usage
# of representation and cache utilities.
_MODEL_EXPORTS: dict[str, str] = {
    "VisionModel": "vneurotk.vision.model.base",
    "print_modules": "vneurotk.vision.model.base",
    "ModuleSelector": "vneurotk.vision.model.selector",
    "BlockLevelSelector": "vneurotk.vision.model.selector",
    "AllLeafSelector": "vneurotk.vision.model.selector",
    "CustomSelector": "vneurotk.vision.model.selector",
}

__all__ = [
    "VisionModel",
    "print_modules",
    "ModuleSelector",
    "BlockLevelSelector",
    "AllLeafSelector",
    "CustomSelector",
    "ModelInfo",
    "ModuleInfo",
    "VisualRepresentation",
    "VisualRepresentations",
    "ImageSource",
]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        import importlib

        module = importlib.import_module(_MODEL_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'vneurotk.vision' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.vision.model.base import VisionModel, print_modules  # noqa: F401
    from vneurotk.vision.model.selector import (  # noqa: F401
        AllLeafSelector,
        BlockLevelSelector,
        CustomSelector,
        ModuleSelector,
    )
