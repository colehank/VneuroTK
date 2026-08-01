# VneuroTK

A Python toolkit for modern visual neuroscience.

```{warning}
:class: dropdown
**Pre-alpha software**

VneuroTK is under active development. Public APIs and the HDF5 schema may
change before the first stable release. Pin versions and retain source data
when using it in research workflows.
```

## Quickstart

Install the core package, then create and persist a recording:

```sh
pip install vneurotk
```

```python
from pathlib import Path

import numpy as np
import vneurotk as vtk

recording = vtk.BaseData.for_patterns(
    neuro=np.zeros((100, 32)),
    neuro_info={"ch_names": [f"ch{i}" for i in range(32)]},
)
recording.save(vtk.VTKPath(Path("outputs"), subject="01", task="demo"))
loaded = vtk.read(Path("outputs/sub-01_task-demo.h5"))
```

`loaded.neuro` is a `NeuroData` container. Use `loaded.neuro.data` for its
plain NumPy array. See [Installation](installation.md) for vision, plotting,
and M/EEG extras.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Install VneuroTK and its optional dependencies.
:::

:::{grid-item-card} Usage
:link: usage
:link-type: doc
Learn how to use paths, data objects, and vision feature extraction.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc
Auto-generated documentation for all public modules and classes.
:::

:::{grid-item-card} Changelog
:link: changelog
:link-type: doc
Release notes and version history.
:::

:::{grid-item-card} Research and project policies
:link: data-policy
:link-type: doc
Review dataset rights and ethics, citation, support, governance, and security.
:::
::::
