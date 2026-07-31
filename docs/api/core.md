# Core

Joint data object and shared primitives.

## BaseData

::: vneurotk.core.recording.BaseData

## Metadata dictionaries

`NeuroInfo`, `VisionInfo`, and `TrialInfo` are `TypedDict` boundary types for the known metadata keys. Runtime values remain ordinary mutable dictionaries, so existing dict inputs, extra keys, persistence, and equality behavior remain compatible.

::: vneurotk.core.metadata.NeuroInfo

::: vneurotk.core.metadata.VisionInfo

::: vneurotk.core.metadata.TrialInfo

## StimulusSet

::: vneurotk.core.stimulus.StimulusSet

## Info

::: vneurotk.core.info.Info
