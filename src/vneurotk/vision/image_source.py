"""ImageSource protocol — the common interface for stimulus image stores."""

from __future__ import annotations

from typing import Any, runtime_checkable

from typing_extensions import Protocol

__all__ = ["ImageSource"]


@runtime_checkable
class ImageSource(Protocol):
    """Protocol for any mapping from stimulus ID to image data.

    Both :class:`~vneurotk.neuro.base.StimulusSet` and
    :class:`~vneurotk.io.loader.LazyH5Dict` satisfy this protocol, as does a
    plain :class:`dict`.  Callers that accept stimulus images should annotate
    their parameter as ``ImageSource`` rather than enumerating concrete types.
    """

    def __getitem__(self, stim_id: Any) -> Any: ...

    def __contains__(self, stim_id: Any) -> bool: ...

    def __len__(self) -> int: ...
