# Path system of vneurotk

Four path classes handle file location for different data sources. Path objects only construct paths — no I/O is triggered until they are passed to `vtk.read(...)`, `BaseData.save(...)`, or their explicit `.load()` method.

| Class | Use |
|---|---|
| `VTKPath` | VneuroTK HDF5 recordings; base naming convention |
| `EphysPath` | Session-level ephys data (spike raster, mean firing rate, etc.) |
| `MNEPath` | MNE-readable MEG / EEG files |
| `BIDSPath` | BIDS paths backed by the optional `mne-bids` package |

All `root` arguments accept `str` or `pathlib.Path`, and every `.fpath` result is
a `pathlib.Path`. `BIDSPath` can construct a fallback path without the `mne`
extra, but full BIDS behavior and loading require `vneurotk[mne]`.

## EphysPath

```python
from vneurotk.io import EphysPath

# Basic
p = EphysPath(root=EPHYS_ROOT, session_id="251024_FanFan_nsd1w_MSB", dtype="TrialRaster", extension="h5")
p.fpath  # → {root}/sessions/{session_id}/TrialRaster_{session_id}.h5

# Multi-probe (appends _probe{N})
p = EphysPath(root=EPHYS_ROOT, session_id="251024_FanFan_nsd1w_MSB", dtype="TrialRaster", probe=0, extension="h5")

# from_components: decompose session_id into fields
p = EphysPath.from_components(
    root=EPHYS_ROOT,
    date="251024", subject="FanFan", paradigm="nsd1w", region="MSB",
    dtype="TrialRaster", extension="h5",
)
```

Supported `dtype` values:

| dtype | Level |
|---|---|
| `TrialRaster`, `TrialRecord`, `UnitProp`, `MeanFr` | unit |
| `ChTrialRaster`, `ChTrialRecord`, `ChMeanFr`, `ChProp`, `ChStimFr` | channel |

Helper attributes: `p.session_dir`, `p.raw_dir`, `p.nwb_path`.

## MNEPath

Builds an MNE-style filename directly under `root`:
`{root}/sub-{subject}_ses-{session}_task-{task}_run-{run}_{suffix}{extension}`.
Use `BIDSPath` when the directory layout and entities should be delegated to
`mne_bids.BIDSPath`.

```python
from vneurotk.io import MNEPath

mne_path = MNEPath(
    root=MNE_ROOT,
    subject="01", session="ImageNet01", task="ImageNet", run="01",
    suffix="meg_clean", extension=".fif",
)
mne_path.fpath
```

## VTKPath

vneurotk's own HDF5 save format: `{root}/sub-{subject}_ses-{session}_task-{task}_run-{run}.h5`.

```python
from vneurotk.io import VTKPath

vtk_path = VTKPath(SAVE_ROOT, subject="01", session="ImageNet01", task="ImageNet", run="01")
vtk_path.fpath

# Or construct from an existing .h5 path
vtk_path = VTKPath(existing_h5_path)
```

## BIDSPath

```python
from vneurotk.io import BIDSPath

bids_path = BIDSPath(
    root=BIDS_ROOT,
    subject="01",
    session="01",
    task="images",
    run="01",
    suffix="meg",
    extension=".fif",
)
bids_path.fpath
```

With `mne-bids` installed, `.bids_path` exposes the wrapped
`mne_bids.BIDSPath`. Without it, `.bids_path` is `None` and `.fpath` uses the
base VneuroTK naming fallback.

All path objects are accepted by `vtk.read()`. Plain `str` and `pathlib.Path`
inputs are also accepted; `.h5` selects the VneuroTK HDF5 reader and other
formats are dispatched by the loader.
