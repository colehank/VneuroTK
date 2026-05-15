"""vneurotk.core — Joint Data Object and shared primitives (internal module).

Public API is re-exported from the top-level ``vneurotk`` package.
Direct imports (``from vneurotk.core import BaseData``) are valid but not canonical.
"""

from __future__ import annotations

from vneurotk.core.info import Info
from vneurotk.core.recording import BaseData
from vneurotk.core.stimulus import StimulusSet

__all__ = ["BaseData", "StimulusSet", "Info"]
