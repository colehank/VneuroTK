# Extract vision features with vneurotk data

`data.vision.extract_from()` uses the image database bound by `configure()` and stores features at unique-stimulus granularity. Onset-aligned (trial-order) arrays are produced at read time.

## Extract features

```python
# data must be configured first (vision_db bound via data.configure())
data.vision.extract_from(model, batch_size=16)
# VisionData(200 stimuli x 13 modules)
```

Calling `extract_from()` again with the same model is a no-op.

## Index `data.vision`

| Index | Returns |
|---|---|
| `data.vision["layer_name"]` | ndarray, shape `(n_trials, ...)` onset-aligned |
| `data.vision[int]` | ndarray for the *i*-th layer, onset-aligned |
| `data.vision[bool_mask]` (1 match) | ndarray, onset-aligned |
| `data.vision[bool_mask]` (multi) | `VisualRepresentations` |

```python
feat = data.vision["layernorm"]          # (200, 257, 768)

meta = data.vision.meta
dinov2_mask = meta["module_type"] == "Dinov2Layer"
feat_multi = data.vision[dinov2_mask]    # VisualRepresentations, 12 layers
```

## Add a second model

```python
model2 = vtk.VisionModel("timm/resnet50.a1_in1k", backend="timm", device=device)
model2.set_selector(module_type="Bottleneck", module_name="global_pool")

data.vision.extract_from(model2, batch_size=16)
data.vision.meta  # now contains layers from both models
```

## Save and reload

Features are persisted together with neural data in a single HDF5 file:

```python
data.save(vtk_path)

loaded = vtk.read(vtk_path)
# loaded.vision retains all extracted features
```
