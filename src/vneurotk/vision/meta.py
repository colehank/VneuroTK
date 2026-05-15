"""Vision metadata dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ModelInfo", "ModuleInfo"]


@dataclass
class ModelInfo:
    """Provenance record for a feature extraction run.

    Parameters
    ----------
    model_id : str
        Model identifier passed to the backend, e.g. ``"facebook/dinov2-base"``
        or ``"resnet50"``.
    backend : str
        Backend used: ``"timm"``, ``"transformers"``, or ``"thingsvision"``.
    """

    model_id: str
    backend: str


@dataclass
class ModuleInfo:
    """Metadata for an enumerated module.

    Parameters
    ----------
    name : str
        Module name as from ``named_modules()``.
    module_type : str
        Class name of the module.
    depth : int
        Nesting depth in the module tree.
    n_params : int
        Total number of parameters in this module (including children).
    is_leaf : bool
        True if the module has no child modules (suitable for direct hooking).
    param_shapes : dict[str, tuple]
        Shape of each directly-owned parameter (empty for container modules).
        E.g. ``{"weight": (768, 768), "bias": (768,)}``.
    """

    name: str
    module_type: str
    depth: int
    n_params: int = 0
    is_leaf: bool = False
    param_shapes: dict[str, tuple] = field(default_factory=dict)
