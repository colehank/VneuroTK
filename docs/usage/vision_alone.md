# Extract vision features from images

`model.extract()` operates on any `{stim_id: image}` mapping independently of `BaseData`.
It always returns `VisualRepresentations`; indexing that collection by module
name, integer position, or boolean mask always returns representation objects,
never bare arrays.

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

# Structured extraction metadata is attached to every representation.
provenance = vrs[0].provenance
provenance.backend              # "transformers"
provenance.model_id             # backend-native model identifier
provenance.preprocessing        # locally discovered processor description
provenance.dependency_versions  # installed versions; unavailable values are "unknown"
```

`ExtractionProvenance` also records the locally available model revision,
pretrained/random-weight choice, selector configuration, model dtype, inference
device, and VneuroTK writer version. It never performs a registry or other
network lookup. Unknown metadata stays explicitly `"unknown"` rather than being
inferred. The object has stable `to_dict()` / `to_json()` serialization and
matching `from_dict()` / `from_json()` constructors.

To include an optional caller-computed digest of the ordered stimulus content,
pass it during extraction:

```python
vrs = model.extract(imgs, batch_size=16, stimulus_content_hash="sha256:...")
vrs[0].provenance.stimulus_content_hash  # "sha256:..."
```

VneuroTK does not hash or fetch stimulus content implicitly.

## Index results

| Index | Returns |
|---|---|
| `vrs["layer_name"]` | `VisualRepresentation` (single layer) |
| `vrs[int]` | `VisualRepresentation` (by position) |
| `vrs[bool_mask]` | `VisualRepresentations` (including zero or one match) |

```python
# By layer name
vr = vrs["layernorm"]
arr = vrs.numpy("layernorm")    # → ndarray, shape (20, 257, 768)
t   = vrs.to_tensor("layernorm")  # → torch.Tensor

# Bool mask — always a VisualRepresentations collection
subset = vrs[vrs.meta["module_type"] == "Dinov2Layer"]
one_layer = vrs[vrs.meta["module_name"] == "layernorm"]
vr = one_layer[0]

# Select a stimulus subset (all layers aligned)
vrs_5 = vrs.select(list(vrs.stim_ids[:5]))
```
