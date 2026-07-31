"""Vision model backend interfaces and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vneurotk.vision.meta import ModuleInfo
from vneurotk.vision.model.backend.base import BaseBackend

_EXPORTS: dict[str, str] = {
    "ThingsVisionBackend": "vneurotk.vision.model.backend.thingsvision_backend",
    "TimmBackend": "vneurotk.vision.model.backend.timm_backend",
    "TransformersBackend": "vneurotk.vision.model.backend.transformers_backend",
}

__all__ = [
    "BaseBackend",
    "ModuleInfo",
    "ThingsVisionBackend",
    "TimmBackend",
    "TransformersBackend",
]


def __getattr__(name: str) -> Any:
    """Load a concrete backend only when requested."""
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'vneurotk.vision.model.backend' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.vision.model.backend.thingsvision_backend import ThingsVisionBackend  # noqa: F401
    from vneurotk.vision.model.backend.timm_backend import TimmBackend  # noqa: F401
    from vneurotk.vision.model.backend.transformers_backend import TransformersBackend  # noqa: F401
