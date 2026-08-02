# Vision

DNN vision representation module.

## VisionData

```{eval-rst}
.. autoclass:: vneurotk.vision.data.VisionData
   :members:
```
## VisionModel

```{eval-rst}
.. autoclass:: vneurotk.vision.model.base.VisionModel
   :members:
```
## Backend interface

Concrete backends are normally selected through `VisionModel`; `BaseBackend` documents the interface implemented by each backend.

```{eval-rst}
.. autoclass:: vneurotk.vision.model.backend.base.BaseBackend
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.backend.transformers_backend.TransformersBackend
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.backend.timm_backend.TimmBackend
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.backend.thingsvision_backend.ThingsVisionBackend
```

## Model and module utilities

```{eval-rst}
.. autofunction:: vneurotk.vision.model.base.print_modules
```
```{eval-rst}
.. autoclass:: vneurotk.vision._cache.CachedModel
   :members:
```
```{eval-rst}
.. autofunction:: vneurotk.vision._cache.find_cached_models
```
```{eval-rst}
.. autofunction:: vneurotk.vision._cache.print_cached_models
```

## Module Selectors

```{eval-rst}
.. autoclass:: vneurotk.vision.model.selector.ModuleSelector
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.selector.BlockLevelSelector
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.selector.AllLeafSelector
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.model.selector.CustomSelector
   :members:
```
## Representations

```{eval-rst}
.. autoclass:: vneurotk.vision.representation.visual_representations.VisualRepresentations
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.representation.visual_representations.VisualRepresentation
   :members:
```
## Image Source

```{eval-rst}
.. autoclass:: vneurotk.vision.image_source.ImageSource
   :members:
```
## Metadata

```{eval-rst}
.. autoclass:: vneurotk.vision.meta.ExtractionProvenance
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.meta.ModelInfo
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.vision.meta.ModuleInfo
   :members:
```