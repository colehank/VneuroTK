# VneuroTK

VneuroTK is a Python toolkit for visual neuroscience. It brings neural recordings, trial and stimulus metadata, and layer-level vision-model representations into a common workflow for research analysis.

The toolkit provides path abstractions and lazy loading for heterogeneous recording sources, explicit representations for continuous, epoched, and pattern data, standalone or recording-integrated vision feature extraction, and structured HDF5 persistence with extraction provenance.

```{warning}
:class: dropdown
**Pre-alpha software**

VneuroTK is under active development. Public APIs and the HDF5 schema may
change before the first stable release. Pin versions and retain source data
when using it in research workflows.
```

## Explore the documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
Install the core toolkit and the optional integrations needed by your workflow.
:::

:::{grid-item-card} File formats
:link: format/hdf5
:link-type: doc
Understand the VneuroTK HDF5 layout, persistence guarantees, and compatibility policy.
:::

:::{grid-item-card} Usage
:link: usage
:link-type: doc
Learn individual tasks through notebook-native guides for paths, neural data, and vision extraction.
:::

:::{grid-item-card} Examples
:link: examples
:link-type: doc
Follow complete notebooks with committed code and outputs.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
Look up the documented classes, functions, and modules.
:::

:::{grid-item-card} Project
:link: project
:link-type: doc
Contribute, get support, cite the toolkit, and review project policies.
:::
::::

See the [changelog](changelog.md) for release history.
