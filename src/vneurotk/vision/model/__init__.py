"""vneurotk.vision.model — VisionModel and layer selection."""

try:
    import torch  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "vneurotk.vision.model requires torch + transformers.  Install with: uv add 'vneurotk[vision]'."
    ) from _exc

from vneurotk.vision.model.backend.base import BaseBackend, ModuleInfo
from vneurotk.vision.model.base import VisionModel, print_modules
from vneurotk.vision.model.selector import (
    AllLeafSelector,
    BlockLevelSelector,
    CustomSelector,
    ModuleSelector,
)

__all__ = [
    "VisionModel",
    "print_modules",
    "BaseBackend",
    "ModuleInfo",
    "ModuleSelector",
    "BlockLevelSelector",
    "AllLeafSelector",
    "CustomSelector",
]
