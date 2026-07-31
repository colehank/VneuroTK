# VneuroTK

A Python toolkit for modern visual neuroscience.

!!! warning "Pre-alpha software"
    VneuroTK is under active development. Public APIs and the HDF5 schema may
    change before the first stable release. Pin versions and retain source data
    when using it in research workflows.

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

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Install vneurotk and its optional dependencies.

    [:octicons-arrow-right-24: Get started](installation.md)

-   :material-book-open-variant:{ .lg .middle } **Usage**

    ---

    Learn how to use paths, data objects, and vision feature extraction.

    [:octicons-arrow-right-24: Usage guides](usage.md)

-   :material-code-tags:{ .lg .middle } **API Reference**

    ---

    Auto-generated documentation for all public modules and classes.

    [:octicons-arrow-right-24: API docs](api.md)


-   :material-history:{ .lg .middle } **Changelog**

    ---

    Release notes and version history.

    [:octicons-arrow-right-24: Changelog](changelog.md)

-   :material-scale-balance:{ .lg .middle } **Research and project policies**

    ---

    Review dataset rights and ethics, citation, support, governance, and security.

    [:octicons-arrow-right-24: Data policy](data-policy.md)

</div>
