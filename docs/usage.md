# Usage

Each guide is a rendered notebook: narrative, runnable Python cells, and any committed outputs stay together as one source. Documentation builds never execute the notebooks; use the download link on a guide when you want to run it locally.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Path system
:link: usage/path
:link-type: doc
Manage file paths for Ephys, MEG/EEG, and VTK data with a unified path API.
:::

:::{grid-item-card} Neural data
:link: usage/data
:link-type: doc
Build VneuroTK datasets from different recording sources.
:::

:::{grid-item-card} Visualization
:link: usage/viz
:link-type: doc
Plot stimulus timing and neural activity for continuous and epoched recordings with Matplotlib.
+++
Requires the `viz` extra. This plots recordings; it does not extract DNN image representations.
:::

:::{grid-item-card} DNN vision models and backends
:link: usage/vision_models
:link-type: doc
Select and configure supported DNN vision-model backends.
+++
Requires a matching vision extra. Start with [Installation](installation.md) to distinguish backend dependencies from `viz` plotting.
:::

:::{grid-item-card} Standalone DNN vision extraction
:link: usage/vision_alone
:link-type: doc
Extract visual model features from images without VneuroTK data objects.
:::

:::{grid-item-card} Integrated DNN vision extraction
:link: usage/vision_union
:link-type: doc
Extract visual features and bind them directly to VneuroTK datasets.
:::
::::
