<img src="https://raw.githubusercontent.com/colehank/VneuroTK/main/docs/assets/logo.svg" alt="VneuroTK logo" width="300" align="right" />

# VneuroTK

![PyPI version](https://img.shields.io/pypi/v/vneurotk.svg)
![CI](https://img.shields.io/github/actions/workflow/status/colehank/VneuroTK/ci.yml?branch=main&label=CI)
[![Documentation](https://img.shields.io/github/actions/workflow/status/colehank/VneuroTK/docs.yml?branch=main&label=docs)](https://colehank.github.io/VneuroTK/)

A Python toolkit for visual neuroscience.

> **Pre-alpha:** VneuroTK is under active development. Public APIs and the HDF5 schema may change before the first stable release. Pin versions and keep source data when using it in research workflows.

- GitHub: https://github.com/colehank/VneuroTK/
- Documentation: https://colehank.github.io/VneuroTK/
- PyPI package: https://pypi.org/project/vneurotk/

## Installation

Install core neural-data containers and HDF5 I/O:

```sh
pip install vneurotk
```

Install only the features you need:

```sh
pip install "vneurotk[vision]"       # PyTorch + Transformers
pip install "vneurotk[timm]"         # timm backend + shared vision stack
pip install "vneurotk[thingsvision]" # thingsvision backend (Python 3.11–3.12)
pip install "vneurotk[viz]"          # Matplotlib plots
pip install "vneurotk[mne]"          # M/EEG and BIDS support
```

See the [installation guide](https://colehank.github.io/VneuroTK/installation/) for all extras and source setup.

## Quickstart

```python
from pathlib import Path

import numpy as np
import vneurotk as vtk

recording = vtk.BaseData.for_patterns(
    neuro=np.zeros((100, 32)),
    neuro_info={"ch_names": [f"ch{i}" for i in range(32)]},
)

# NeuroData wraps the array; use .data for a plain ndarray.
assert recording.neuro.data.shape == (100, 32)

path = vtk.VTKPath(Path("outputs"), subject="01", task="demo")
recording.save(path)
loaded = vtk.read(path)
```

`VTKPath.fpath` is `outputs/sub-01_task-demo.h5`. Path classes construct locations without performing I/O; pass them to `vtk.read(...)`, `BaseData.save(...)`, or call `.load()` explicitly.

Vision support uses a named backend and returns one `VisualRepresentation` per selected module:

```python
model = vtk.VisionModel("facebook/dinov2-base", backend="transformers")
features = model.extract({"stimulus-1": np.zeros((224, 224, 3), dtype=np.uint8)})
layer = features[0]
print(layer.provenance.backend)
```

The `transformers`, `timm`, and `thingsvision` backends use their native model identifiers and may download weights unless those artifacts are already cached. The ThingsVision extra is currently limited to Python 3.11–3.12 because ThingsVision 1.4.4's eager TensorFlow import is not runtime-compatible with Python 3.13 in the supported dependency set. See the [vision guide](https://colehank.github.io/VneuroTK/usage/vision-models/) and [HDF5 format notes](https://colehank.github.io/VneuroTK/file-formats/hdf5-recordings/).

## Project policies

- [Citation](https://colehank.github.io/VneuroTK/citation/): cite the exact release or Git revision; datasets and models require separate citations.
- [Dataset provenance, licensing, citation, and ethics](https://colehank.github.io/VneuroTK/data-policy/): verify rights and authoritative metadata before use or redistribution.
- [Support](SUPPORT.md), [Governance](GOVERNANCE.md), and [Security](https://colehank.github.io/VneuroTK/security/).
