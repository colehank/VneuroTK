"""Vision models and layer selection with lazy optional dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vneurotk.vision.meta import ModuleInfo
from vneurotk.vision.model.backend.base import BaseBackend

_EXPORTS: dict[str, str] = {
    "VisionModel": "vneurotk.vision.model.base",
    "print_modules": "vneurotk.vision.model.base",
    "ModuleSelector": "vneurotk.vision.model.selector",
    "BlockLevelSelector": "vneurotk.vision.model.selector",
    "AllLeafSelector": "vneurotk.vision.model.selector",
    "CustomSelector": "vneurotk.vision.model.selector",
}

__all__ = [
    "AllLeafSelector",
    "BaseBackend",
    "BlockLevelSelector",
    "CustomSelector",
    "ModuleInfo",
    "ModuleSelector",
    "VisionModel",
    "print_modules",
]


def __getattr__(name: str) -> Any:
    """Load torch-dependent model APIs only when requested."""
    if name in _EXPORTS:
        import importlib

        try:
            return getattr(importlib.import_module(_EXPORTS[name]), name)
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    "Vision models require PyTorch. Install it with: pip install 'vneurotk[vision]'"
                ) from exc
            raise
    raise AttributeError(f"module 'vneurotk.vision.model' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.vision.model.base import VisionModel, print_modules  # noqa: F401
    from vneurotk.vision.model.selector import (  # noqa: F401
        AllLeafSelector,
        BlockLevelSelector,
        CustomSelector,
        ModuleSelector,
    )
