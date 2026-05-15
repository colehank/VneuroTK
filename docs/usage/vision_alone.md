# Extract vision features from images

`model.extract()` operates on any `{stim_id: image}` mapping independently of `BaseData`.

## Prepare images

```python
import numpy as np

# Values can be: str/Path (local file), np.ndarray (HWC uint8), or PIL.Image
imgs: dict[str, np.ndarray] = {
    f"img_{i:03d}": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    for i in range(20)
}
```

## Extract features

```python
vrs = model.extract(imgs, batch_size=16)
# VisualRepresentations(20 stimuli x 13 modules)

vrs.meta  # DataFrame: model, module_type, module_name, shape
```

## Index results

| Index | Returns |
|---|---|
| `vrs["layer_name"]` | `VisualRepresentation` (single layer) |
| `vrs[int]` | `VisualRepresentation` (by position) |
| `vrs[bool_mask]` | `VisualRepresentations` (subset) or `VisualRepresentation` (1 match) |

```python
# By layer name
vr = vrs["layernorm"]
arr = vrs.numpy("layernorm")    # → ndarray, shape (20, 257, 768)
t   = vrs.to_tensor("layernorm")  # → torch.Tensor

# Bool mask — multiple matches → VisualRepresentations
subset = vrs[vrs.meta["module_type"] == "Dinov2Layer"]

# Select a stimulus subset (all layers aligned)
vrs_5 = vrs.select(list(vrs.stim_ids[:5]))
```
