"""Vision metadata and extraction provenance dataclasses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar

__all__ = ["ExtractionProvenance", "ModelInfo", "ModuleInfo", "UNKNOWN"]

UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExtractionProvenance:
    """Reproducibility metadata for one feature-extraction result.

    The record deliberately uses the literal string ``"unknown"`` for metadata
    that cannot be discovered locally.  This makes missing information explicit
    without requiring a model registry lookup or other network access.

    Parameters
    ----------
    backend : str
        Backend that executed the model.
    model_id : str
        Backend-native model identifier.
    model_revision : str
        Locally available model revision/commit, or ``"unknown"``.
    pretrained : bool or str
        Whether pretrained weights were requested, or ``"unknown"``.
    preprocessing : str
        Stable description of the processor or preprocessing transform.
    selector : str
        Stable description of the module selector.
    dependency_versions : mapping
        Locally installed dependency versions. Missing versions are explicit as
        ``"unknown"``.
    dtype : str
        Model parameter dtype, or ``"unknown"``.
    device : str
        Device used for inference, or ``"unknown"``.
    writer_version : str
        VneuroTK version that created this provenance record.
    stimulus_content_hash : str or None
        Optional ``sha256:...`` digest of the ordered stimulus mapping.
    """

    SERIALIZATION_VERSION: ClassVar[int] = 1

    backend: str = UNKNOWN
    model_id: str = UNKNOWN
    model_revision: str = UNKNOWN
    pretrained: bool | str = UNKNOWN
    preprocessing: str = UNKNOWN
    selector: str = UNKNOWN
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    dtype: str = UNKNOWN
    device: str = UNKNOWN
    writer_version: str = UNKNOWN
    stimulus_content_hash: str | None = None

    def __post_init__(self) -> None:
        versions = {
            str(name): str(value) if value is not None else UNKNOWN
            for name, value in sorted(self.dependency_versions.items())
        }
        object.__setattr__(self, "dependency_versions", MappingProxyType(versions))

    @classmethod
    def unknown(cls, *, model_id: str = UNKNOWN) -> ExtractionProvenance:
        """Return an explicit unknown record, retaining a known model ID."""
        return cls(model_id=model_id, dependency_versions={"vneurotk": UNKNOWN})

    def to_dict(self) -> dict[str, Any]:
        """Return the complete, serialization-stable mapping representation."""
        return {
            "serialization_version": self.SERIALIZATION_VERSION,
            "backend": self.backend,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "pretrained": self.pretrained,
            "preprocessing": self.preprocessing,
            "selector": self.selector,
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "dtype": self.dtype,
            "device": self.device,
            "writer_version": self.writer_version,
            "stimulus_content_hash": self.stimulus_content_hash,
        }

    def to_json(self) -> str:
        """Serialize deterministically as compact UTF-8-safe JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExtractionProvenance:
        """Construct from a serialized mapping.

        Absent fields are interpreted as explicit unknowns, which also makes the
        reader tolerant of early or manually-authored schema-1 records.
        """
        raw_version = value.get("serialization_version", cls.SERIALIZATION_VERSION)
        if (
            isinstance(raw_version, bool)
            or not isinstance(raw_version, int)
            or raw_version != cls.SERIALIZATION_VERSION
        ):
            raise ValueError(f"Unsupported extraction provenance serialization version {raw_version!r}.")
        raw_versions = value.get("dependency_versions", {})
        if not isinstance(raw_versions, Mapping):
            raise TypeError("dependency_versions must be a mapping.")
        pretrained = value.get("pretrained", UNKNOWN)
        if not isinstance(pretrained, (bool, str)):
            raise TypeError("pretrained must be bool or 'unknown'.")
        content_hash = value.get("stimulus_content_hash")
        return cls(
            backend=str(value.get("backend", UNKNOWN)),
            model_id=str(value.get("model_id", UNKNOWN)),
            model_revision=str(value.get("model_revision", UNKNOWN)),
            pretrained=pretrained,
            preprocessing=str(value.get("preprocessing", UNKNOWN)),
            selector=str(value.get("selector", UNKNOWN)),
            dependency_versions={str(k): str(v) for k, v in raw_versions.items()},
            dtype=str(value.get("dtype", UNKNOWN)),
            device=str(value.get("device", UNKNOWN)),
            writer_version=str(value.get("writer_version", UNKNOWN)),
            stimulus_content_hash=None if content_hash is None else str(content_hash),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> ExtractionProvenance:
        """Deserialize a record produced by :meth:`to_json`."""
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        decoded = json.loads(value)
        if not isinstance(decoded, dict):
            raise TypeError("Extraction provenance JSON must contain an object.")
        return cls.from_dict(decoded)


@dataclass
class ModelInfo:
    """Basic metadata for a loaded model.

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
