# Path system of vneurotk

Three path classes handle file location for different data sources. Path objects only construct paths — no IO is triggered.

| Class | Use |
|---|---|
| `EphysPath` | Ephys data (spike raster, mean firing rate, etc.) |
| `MNEPath` | MEG / EEG in MNE-BIDS format |
| `VTKPath` | vneurotk HDF5 save format |

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

Follows MNE-BIDS convention: `{root}/sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_run-{run}_{suffix}{extension}`.

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

All path objects are passed directly to `vtk.read()`.
