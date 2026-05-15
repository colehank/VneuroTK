# Build with NOD-MEG

End-to-end workflow using [NOD-MEG](https://openneuro.org/datasets/ds005810): read raw MEG, configure trial structure, extract DNN features, and persist everything to HDF5.

```python
import os
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

import vneurotk as vtk
from vneurotk.datasets import sample
from vneurotk.io import MNEPath, VTKPath

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # optional: mirror for faster download

root = sample.data_path("nod-meg")
nod  = root / "nod-meg"
SAVE_ROOT = Path(tempfile.mkdtemp()) / "mne" / "NOD-MEG"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

## 1. Read MEG data

`vtk.read()` is lazy — neural data is not loaded until first access.

```python
sub, session, run = sample.NOD_SUBJECT, sample.NOD_SESSION, sample.NOD_RUN

mne_path = MNEPath(
    root=nod / "meg",
    subject=sub, session=session, task=sample.NOD_TASK, run=run,
    suffix="meg_clean", extension=".fif",
)

data: vtk.BaseData = vtk.read(mne_path)
# BaseData(ntime=80000, nchan=273, n_trials=0, configured=False, neuro=<lazy>)
```

## 2. Configure trial structure

`configure()` binds stimulus IDs, onset samples, time window, and image database.

```python
submeta = pd.read_csv(nod / "events" / f"sub-{sub}_events.csv")
runmeta = submeta.query(f"run == {int(run)} and session == '{session}'")
stim_ids = runmeta["image_id"].values
stims = {sid: nod / "stimuli" / f"{sid}.JPEG" for sid in np.unique(stim_ids)}

raw = mne.io.read_raw(mne_path.fpath, preload=False, verbose=False)
vision_onsets = vtk.utils.get_event_samples(raw, event_name="stim_on")

data.configure(
    vision_onsets=vision_onsets,
    stim_ids=stim_ids,
    vision_db=stims,
    trial_window=[-0.2, 0.8],
)
# BaseData(ntime=80000, nchan=273, n_trials=200, configured=True, neuro=<lazy>)
```

## 3. Access neural views

`data.neuro` triggers lazy load and returns a `NeuroData` (`np.ndarray` subclass).

```python
neuro = data.neuro

neuro.shape            # (80000, 273)  — raw continuous signal
neuro.epochs.shape     # (200, 250, 273)  — (n_trials, n_timebins, nchan)
neuro.continuous.shape # (80000, 273)
```

## 4. Load a vision model

`VisionModel` supports three backends: `"transformers"`, `"timm"`, `"thingsvision"`.

```python
vtk.print_cached_models()  # list locally cached models

model = vtk.VisionModel(
    "facebook/dinov2-base",
    backend="transformers",
    device=DEVICE,
)
model.print_modules(max_depth=3)  # inspect layer tree
```

Select target layers:

```python
all_modules = model.module_list
dinov2_layers = [m for m in all_modules if m.module_type == "Dinov2Layer"]
last_norm     = [m for m in all_modules if m.module_type == "LayerNorm"][-1]

model.set_selector(dinov2_layers + [last_norm])
# equivalent shorthand:
# model.set_selector(module_type="Dinov2Layer", module_name="layernorm")
```

## 5. Standalone extraction — `model.extract()`

Operates on any `{stim_id: image}` mapping, independent of `BaseData`.

```python
demo_imgs = {sid: nod / "stimuli" / f"{sid}.JPEG" for sid in list(np.unique(stim_ids))[:20]}
vrs = model.extract(demo_imgs, batch_size=16)
# VisualRepresentations(20 stimuli x 13 modules)

vrs.meta           # DataFrame: model, module_type, module_name, shape
vr  = vrs["layernorm"]           # VisualRepresentation
arr = vrs.numpy("layernorm")     # ndarray (20, 257, 768)

# Bool mask — multiple matches → VisualRepresentations
subset = vrs[vrs.meta["module_type"] == "Dinov2Layer"]  # 12 layers

# Stimulus subset (all layers aligned)
vrs_5 = vrs.select(vrs.stim_ids[:5])
```

## 6. Integrated extraction — `data.vision.extract_from()`

Uses the image database bound by `configure()`. Features stored at unique-stimulus granularity; trial-order arrays are produced at index time.

```python
data.vision.extract_from(model, batch_size=16)
# VisionData(200 stimuli x 13 modules)
```

## 7. Index `data.vision`

| Index | Returns |
|---|---|
| `data.vision["layer_name"]` | ndarray `(n_trials, ...)`, onset-aligned |
| `data.vision[int]` | ndarray for the *i*-th layer |
| `data.vision[bool_mask]` (1 match) | ndarray |
| `data.vision[bool_mask]` (multi) | `VisualRepresentations` |

```python
feat = data.vision["layernorm"]           # (200, 257, 768)
feat = data.vision[0]                     # first layer

meta = data.vision.meta
dinov2_mask = meta["module_type"] == "Dinov2Layer"
feat_multi  = data.vision[dinov2_mask]    # VisualRepresentations, 12 layers

# Drill into a single layer from the multi-layer result
filtered = feat_multi.meta.query("module_name == 'encoder.layer.11'").index
data.vision[feat_multi[filtered]]         # (200, 257, 768)
```

## 8. Add a second model

```python
model2 = vtk.VisionModel("timm/resnet50.a1_in1k", backend="timm", device=DEVICE)
model2.set_selector(module_type="Bottleneck", module_name="global_pool")

data.vision.extract_from(model2, batch_size=16)
data.vision.meta  # now includes ResNet50 layers alongside DINOv2
```

## 9. Save and reload

`save()` writes neural data, all visual representations, trial configuration, and image pixels into a single HDF5 file. On reload, `vision.db` becomes a `LazyH5Dict` — images decoded only when accessed.

```python
save_path = VTKPath(SAVE_ROOT, subject=sub, session=session, task="ImageNet", run=run)
data.save(save_path)

loaded: vtk.BaseData = vtk.read(save_path)
# BaseData(ntime=80000, nchan=273, n_trials=200, configured=True, has_vision=True, neuro=<lazy>)
```
