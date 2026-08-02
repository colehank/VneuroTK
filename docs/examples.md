# Examples

Work through complete VneuroTK workflows in notebooks that combine explanation, code, and saved output. Documentation builds render the committed notebooks without executing them.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Neural data
:link: example_ipynb/data
:link-type: doc
Build and inspect VneuroTK data objects across recording modes and sources.
+++
Core package; sample-backed sections identify additional requirements.
:::

:::{grid-item-card} Path objects
:link: example_ipynb/path
:link-type: doc
Construct paths for electrophysiology, M/EEG, BIDS, and VneuroTK HDF5 data.
+++
Core path construction; BIDS integration requires the `mne` extra.
:::

:::{grid-item-card} Vision representations
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

## Source notebooks

Download the exact notebook sources rendered by this site:

- <a href="../example_ipynb/data.ipynb" download>Neural data notebook</a>
- <a href="../example_ipynb/path.ipynb" download>Path notebook</a>
- <a href="../example_ipynb/vision.ipynb" download>Vision notebook</a>
- <a href="../example_ipynb/neurovision.ipynb" download>Neurovision notebook</a>
