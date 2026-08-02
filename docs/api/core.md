# Core

Joint data object and shared primitives.

## BaseData

```{eval-rst}
.. autoclass:: vneurotk.core.recording.BaseData
   :members:
```
## Data modes

```{eval-rst}
.. autodata:: vneurotk.core.recording.DataMode
```

## Metadata dictionaries

`NeuroInfo`, `VisionInfo`, and `TrialInfo` are `TypedDict` boundary types for the known metadata keys. Runtime values remain ordinary mutable dictionaries, so existing dict inputs, extra keys, persistence, and equality behavior remain compatible.

```{eval-rst}
.. autoclass:: vneurotk.core.metadata.NeuroInfo
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.core.metadata.VisionInfo
   :members:
```
```{eval-rst}
.. autoclass:: vneurotk.core.metadata.TrialInfo
   :members:
```
## StimulusSet

```{eval-rst}
.. autoclass:: vneurotk.core.stimulus.StimulusSet
   :members:
```
## Info

```{eval-rst}
.. autoclass:: vneurotk.core.info.Info
   :members:
```