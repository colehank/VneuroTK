# Build vneurotk data from different recordings

`vtk.read()` accepts a filesystem path or a VneuroTK path object and returns a
`BaseData` with lazy-loaded neural data. `BaseData.neuro` is a `NeuroData`
container, not an `ndarray` subclass: use `data.neuro.data` for the underlying
NumPy array. Its `.shape`, `.dtype`, `.ndim`, and `.size` attributes and
`np.asarray(data.neuro)` are provided as conveniences.

`BaseData` has three explicit modes:

| Mode | Raw shape | Trial structure |
|---|---|---|
| `continuous` | `(n_samples, n_channels)` | Required before using `.epochs` |
| `epochs` | `(n_trials, n_timebins, n_channels)` | Already trial-structured |
| `patterns` | `(n_rows, n_channels)` | Aggregated rows; no trial arrays required |

Use `BaseData.for_continuous(...)`, `BaseData.for_epochs(...)`, or
`BaseData.for_patterns(...)` when a 2-D array's meaning should be explicit.
Pattern rows need `trial_meta["stim_index"]` only when they must align with
vision features.

## MEG (MNE-BIDS)

```python
import mne
import numpy as np
import pandas as pd
import vneurotk as vtk
from vneurotk.io import MNEPath, VTKPath

mne_path = MNEPath(root=..., subject="01", session="ImageNet01",
                   task="ImageNet", run="01", suffix="meg_clean", extension=".fif")

data: vtk.BaseData = vtk.read(mne_path)
# BaseData(ntime=80000, nchan=273, n_trials=0, configured=False, neuro=<lazy>)
```

**Configure trial structure** — binds stimulus IDs, onset samples, time window, and image database:

```python
raw = mne.io.read_raw(mne_path.fpath, preload=False, verbose=False)
vision_onsets = vtk.utils.get_event_samples(raw, event_name="stim_on")

data.configure(
    vision_onsets=vision_onsets,   # (n_trials,) sample indices
    stim_ids=stim_ids,             # (n_trials,) stimulus IDs
    vision_db={sid: path, ...},    # {stim_id: image path / ndarray / PIL.Image}
    trial_window=[-0.2, 0.8],      # seconds (float) or samples (int)
)
```

**Access neural views** — `data.neuro` triggers lazy load:

```python
neuro = data.neuro                 # NeuroData wrapper
neuro.data                         # raw ndarray, (n_samples, n_channels)
neuro.epochs                       # (n_trials, n_timebins, n_channels)
neuro.continuous                   # (total_trial_samples, n_channels)
```

**Save and reload**:

```python
save_path = VTKPath(SAVE_ROOT, subject="01", session="ImageNet01", task="ImageNet", run="01")
data.save(save_path)

loaded = vtk.read(save_path)
# vision.db becomes LazyH5Dict — images decoded on demand
```

## Ephys

All ephys types are pre-configured on load; no `configure()` call needed.

```python
from vneurotk.io import EphysPath

# Trial-level spike raster — data_mode='epochs', shape (n_trials, n_timebins, n_units)
bd = vtk.read(EphysPath(root=EPHYS_ROOT, session_id=SES, dtype="TrialRaster"))
bd.neuro.shape  # (50932, 350, 333)

# Trial-level mean firing rate — shape (n_trials, n_units)
bd = vtk.read(EphysPath(root=EPHYS_ROOT, session_id=SES, dtype="MeanFr"))

# Stimulus-level channel firing rate — shape (n_stimuli, n_channels)
bd = vtk.read(EphysPath(root=EPHYS_ROOT, session_id=SES, dtype="ChStimFr"))
```

`trial_meta` (from `TrialRecord.csv`) is automatically attached and accessible as `data.trial_meta`.
