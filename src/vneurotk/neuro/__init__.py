"""vneurotk.neuro — Neural-domain primitives for VneuroTK."""

from __future__ import annotations

from vneurotk.neuro.base import NeuroData
from vneurotk.neuro.trial import (
    TrialStructure,
    build_trial_structure_continuous,
    build_trial_structure_epochs,
)

__all__ = [
    "NeuroData",
    "TrialStructure",
    "build_trial_structure_continuous",
    "build_trial_structure_epochs",
]
