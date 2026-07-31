"""Vision data, metadata, representations, and optional model support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vneurotk.vision._cache import find_cached_models, print_cached_models
from vneurotk.vision.data import VisionData
from vneurotk.vision.image_source import ImageSource
from vneurotk.vision.meta import ExtractionProvenance, ModelInfo, ModuleInfo
from vneurotk.vision.representation import VisualRepresentation, VisualRepresentations

_MODEL_EXPORTS: dict[str, str] = {
    "VisionModel": "vneurotk.vision.model",
    "print_modules": "vneurotk.vision.model",
    "ModuleSelector": "vneurotk.vision.model",
    "BlockLevelSelector": "vneurotk.vision.model",
    "AllLeafSelector": "vneurotk.vision.model",
    "CustomSelector": "vneurotk.vision.model",
}

__all__ = [
    "AllLeafSelector",
    "BlockLevelSelector",
    "CustomSelector",
    "ExtractionProvenance",
    "ImageSource",
    "ModelInfo",
    "ModuleInfo",
    "ModuleSelector",
    "VisionData",
    "VisionModel",
    "VisualRepresentation",
    "VisualRepresentations",
    "find_cached_models",
    "print_cached_models",
    "print_modules",
]


def __getattr__(name: str) -> Any:
    """Load model APIs only when they are requested."""
    if name in _MODEL_EXPORTS:
        import importlib

        return getattr(importlib.import_module(_MODEL_EXPORTS[name]), name)
    raise AttributeError(f"module 'vneurotk.vision' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.vision.model import (  # noqa: F401
        AllLeafSelector,
        BlockLevelSelector,
        CustomSelector,
        ModuleSelector,
        VisionModel,
        print_modules,
    )
