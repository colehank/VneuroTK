"""Public VneuroTK API."""

from __future__ import annotations

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from vneurotk import core, datasets, io, neuro, utils, vision, viz
from vneurotk._log import set_log_level, setup_logging
from vneurotk.core import BaseData, DataMode, Info, NeuroInfo, StimulusSet, TrialInfo, VisionInfo
from vneurotk.io import EPHYS_DTYPES, EPHYS_EXTENSIONS, BIDSPath, EphysPath, MNEPath, VTKPath, read
from vneurotk.neuro import (
    NeuroData,
    TrialStructure,
    build_trial_structure_continuous,
    build_trial_structure_epochs,
)

__version__ = version("vneurotk")
__author__ = "VneuroTK Contributors"

__all__ = [
    "__version__",
    "AllLeafSelector",
    "BIDSPath",
    "BaseData",
    "BlockLevelSelector",
    "CustomSelector",
    "DataMode",
    "EPHYS_DTYPES",
    "EPHYS_EXTENSIONS",
    "EphysPath",
    "ExtractionProvenance",
    "ImageSource",
    "Info",
    "MNEPath",
    "ModelInfo",
    "ModuleInfo",
    "ModuleSelector",
    "NeuroData",
    "NeuroInfo",
    "StimulusSet",
    "TrialInfo",
    "TrialStructure",
    "VTKPath",
    "VisionData",
    "VisionInfo",
    "VisionModel",
    "VisualRepresentation",
    "VisualRepresentations",
    "build_trial_structure_continuous",
    "build_trial_structure_epochs",
    "core",
    "datasets",
    "find_cached_models",
    "io",
    "neuro",
    "print_cached_models",
    "print_modules",
    "read",
    "set_log_level",
    "setup_logging",
    "utils",
    "vision",
    "viz",
]

_VISION_EXPORTS = {
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
}


def __getattr__(name: str) -> Any:
    """Resolve top-level vision re-exports without importing model extras eagerly."""
    if name in _VISION_EXPORTS:
        return getattr(vision, name)
    raise AttributeError(f"module 'vneurotk' has no attribute {name!r}")


if TYPE_CHECKING:
    from vneurotk.vision import (  # noqa: F401
        AllLeafSelector,
        BlockLevelSelector,
        CustomSelector,
        ExtractionProvenance,
        ImageSource,
        ModelInfo,
        ModuleInfo,
        ModuleSelector,
        VisionData,
        VisionModel,
        VisualRepresentation,
        VisualRepresentations,
        find_cached_models,
        print_cached_models,
        print_modules,
    )
