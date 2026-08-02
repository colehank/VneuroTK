# Examples

Work through complete VneuroTK workflows in notebooks that combine explanation and code. Documentation builds render the committed notebooks without executing them; only notebooks that include committed output display saved results.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Neural data
:link: example_ipynb/data
:link-type: doc
Build and inspect VneuroTK data objects across recording modes and sources.
+++
Core package; sample-backed sections identify additional requirements.
:::

:::{grid-item-card} Visualization
:link: example_ipynb/viz
:link-type: doc
Build a synthetic evoked-response recording, inspect focused windows, and customize returned Matplotlib figures.
+++
Requires the `viz` extra. Unlike DNN vision extraction, this plots neural recordings and stimulus timing.
:::

:::{grid-item-card} Path objects
:link: example_ipynb/path
:link-type: doc
Construct paths for electrophysiology, M/EEG, BIDS, and VneuroTK HDF5 data.
+++
Core path construction; BIDS integration requires the `mne` extra.
:::

:::{grid-item-card} DNN vision representations
:link: example_ipynb/vision
:link-type: doc
Configure a vision backend, select layers, and extract representations.
+++
Requires the matching vision backend and model assets.
:::

:::{grid-item-card} Integrated Neurovision workflow
:link: example_ipynb/neurovision
:link-type: doc
Align NOD-MEG trials with vision-model features and persist the combined recording.
+++
Requires M/EEG and vision extras, model assets, and the separately licensed sample data.
:::
::::
