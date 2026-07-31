"""Offline smoke tests for canonical imports and documentation examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import vneurotk as vtk

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_public_exports() -> None:
    """The root and subpackages expose the supported API consistently."""
    from vneurotk import (
        BaseData,
        BIDSPath,
        EphysPath,
        ExtractionProvenance,
        MNEPath,
        NeuroData,
        NeuroInfo,
        StimulusSet,
        TrialInfo,
        VisionData,
        VisionInfo,
        VisualRepresentation,
        VisualRepresentations,
        VTKPath,
    )
    from vneurotk.core import BaseData as CoreBaseData
    from vneurotk.core import NeuroInfo as CoreNeuroInfo
    from vneurotk.io import BIDSPath as IoBIDSPath
    from vneurotk.neuro import NeuroData as NeuroNeuroData
    from vneurotk.vision import VisionData as VisionVisionData

    assert BaseData is CoreBaseData
    assert BIDSPath is IoBIDSPath
    assert NeuroData is NeuroNeuroData
    assert NeuroInfo is CoreNeuroInfo
    assert VisionData is VisionVisionData
    assert all(
        symbol is not None
        for symbol in (
            EphysPath,
            ExtractionProvenance,
            MNEPath,
            StimulusSet,
            TrialInfo,
            VisionInfo,
            VTKPath,
            VisualRepresentation,
            VisualRepresentations,
        )
    )


def test_core_import_does_not_eagerly_import_optional_dependencies() -> None:
    """Importing canonical data APIs works without loading optional stacks."""
    code = """
import json
import sys
import vneurotk
from vneurotk import BaseData, NeuroData, VisionData, VisualRepresentations
from vneurotk.vision.model import BaseBackend
from vneurotk.vision.model.backend import ModuleInfo
import vneurotk.viz
optional = ["torch", "transformers", "timm", "thingsvision", "matplotlib", "mne", "mne_bids"]
print(json.dumps({name: name in sys.modules for name in optional}, sort_keys=True))
"""
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "matplotlib": False,
        "mne": False,
        "mne_bids": False,
        "thingsvision": False,
        "timm": False,
        "torch": False,
        "transformers": False,
    }


def test_readme_quickstart_offline(tmp_path: Path) -> None:
    """Exercise the README's core construction, path, save, and read flow."""
    recording = vtk.BaseData.for_patterns(
        neuro=np.zeros((100, 32)),
        neuro_info={"ch_names": [f"ch{i}" for i in range(32)]},
    )
    assert recording.neuro.data.shape == (100, 32)

    path = vtk.VTKPath(tmp_path, subject="01", task="demo")
    assert path.fpath == tmp_path / "sub-01_task-demo.h5"
    recording.save(path)

    loaded = vtk.read(path)
    assert isinstance(loaded, vtk.BaseData)
    assert loaded.neuro.shape == (100, 32)


def test_documented_representation_mask_semantics() -> None:
    """Standalone and recording-attached masks intentionally return different types."""
    vr = vtk.VisualRepresentation(
        model="offline/model",
        module_name="layer",
        module_type="Linear",
        stim_ids=["a", "b"],
        array=np.arange(6).reshape(2, 3),
    )
    representations = vtk.VisualRepresentations([vr])
    mask = representations.meta["module_name"] == "layer"
    subset = representations[mask]
    assert isinstance(subset, vtk.VisualRepresentations)
    assert len(subset) == 1

    vision_data = vtk.VisionData(np.array(["b", "a"]))
    vision_data.add(representations)
    aligned = vision_data[vision_data.meta["module_name"] == "layer"]
    assert isinstance(aligned, np.ndarray)
    np.testing.assert_array_equal(aligned, np.array([[3, 4, 5], [0, 1, 2]]))
