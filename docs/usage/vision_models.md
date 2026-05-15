# Explore vision models

`VisionModel` wraps `transformers`, `timm`, and `thingsvision` under a unified interface for layer-level activation extraction.

## List cached models

```python
import vneurotk as vtk

vtk.print_cached_models()
```

## Load a model

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = vtk.VisionModel(
    "facebook/dinov2-base",
    backend="transformers",  # "timm" | "thingsvision"
    device=device,
)
```

## Inspect layer structure

```python
model.print_modules(max_depth=3)
```

## Select target layers

Three equivalent approaches — results can be combined:

```python
# By type (simplest)
model.set_selector(module_type="Dinov2Layer")

# By name
model.set_selector(module_name="layernorm")

# Type + name union
model.set_selector(module_type="Dinov2Layer", module_name="layernorm")

# Custom list from module_list
all_modules = model.module_list
last_norm = [m for m in all_modules if m.module_type == "LayerNorm"][-1]
model.set_selector([m for m in all_modules if m.module_type == "Dinov2Layer"] + [last_norm])
```

Check selected layers:

```python
model.module_names   # list of selected layer names
```
