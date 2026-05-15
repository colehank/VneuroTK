# Build vneurotk data from different recordings

`vtk.read()` accepts any path object and returns a `BaseData` with lazy-loaded neural data.

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
neuro = data.neuro                 # NeuroData (np.ndarray subclass)
neuro.epochs                       # (n_trials, n_timebins, nchan)
neuro.continuous                   # (total_samples, nchan)
```

**Save and reload**:

```python
data.save(VTKPath(SAVE_ROOT, subject="01", session="ImageNet01", task="ImageNet", run="01"))

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
