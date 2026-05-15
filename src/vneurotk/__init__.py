from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from vneurotk import utils
from vneurotk._log import set_log_level, setup_logging
from vneurotk.core.recording import BaseData
from vneurotk.io import read

# activate vneurotk logger at INFO; suppress MNE output below ERROR
setup_logging("INFO")

__version__ = version("vneurotk")
__author__ = "VneuroTK Contributors"

__all__ = [
    "__version__",
    "read",
    "utils",
    "setup_logging",
    "set_log_level",
    "BaseData",
    "find_cached_models",
    "print_cached_models",
    # vision
    "VisionModel",
    "print_modules",
    "ModuleSelector",
    "BlockLevelSelector",
    "AllLeafSelector",
    "CustomSelector",
    "VisualRepresentation",
    "VisualRepresentations",
    "ModelInfo",
    "ModuleInfo",
]

_VISION_EXPORTS = {
    "VisionModel": "vneurotk.vision.model.base",
    "print_modules": "vneurotk.vision.model.base",
    "ModuleSelector": "vneurotk.vision.model.selector",
    "BlockLevelSelector": "vneurotk.vision.model.selector",
    "AllLeafSelector": "vneurotk.vision.model.selector",
    "CustomSelector": "vneurotk.vision.model.selector",
    "VisualRepresentation": "vneurotk.vision.representation.visual_representations",
    "VisualRepresentations": "vneurotk.vision.representation.visual_representations",
    "ModelInfo": "vneurotk.vision.meta",
    "ModuleInfo": "vneurotk.vision.meta",
    "find_cached_models": "vneurotk.vision._cache",
    "print_cached_models": "vneurotk.vision._cache",
}


def __getattr__(name: str) -> Any:
    """Lazy re-export of vision submodule.

    All vision symbols depend on the ``[vision]`` optional extra (torch,
    transformers, timm).  Deferring their import keeps ``import vneurotk``
    cheap for users who only need the neural-data side of the toolkit.
    """
    if name in _VISION_EXPORTS:
        import importlib

        module = importlib.import_module(_VISION_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'vneurotk' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk._log import set_log_level, setup_logging  # noqa: F401
    from vneurotk.vision.meta import ModelInfo, ModuleInfo  # noqa: F401
    from vneurotk.vision.model.base import VisionModel, print_modules  # noqa: F401
    from vneurotk.vision.model.selector import (  # noqa: F401
        AllLeafSelector,
        BlockLevelSelector,
        CustomSelector,
        ModuleSelector,
    )
    from vneurotk.vision.representation.visual_representations import (
        VisualRepresentation,  # noqa: F401
        VisualRepresentations,  # noqa: F401
    )
