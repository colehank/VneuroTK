# Installation

```{warning}
**Pre-alpha software**

VneuroTK is under active development. Public APIs and storage details may
change before the first stable release; pin versions in reproducible work.
```

## Install from PyPI

Install the core package when you only need neural-data I/O and analysis:

```sh
uv add vneurotk
```

Or with `pip`:

```sh
pip install vneurotk
```

The core installation does not install PyTorch, Transformers, or Matplotlib. Visualization and DNN vision are separate optional features: `viz` installs Matplotlib for plotting recordings, while the vision-related extras install model backends for extracting representations from images.

## Optional dependencies

Install the extra that matches the features or model backend you need:

| Extra | Installs | Use for |
| --- | --- | --- |
| `vision` | `torch`, `transformers` | DNN vision-model and feature-extraction support |
| `viz` | `matplotlib` | Neural-recording and stimulus-timing visualization |
| `timm` | `torch`, `transformers`, `timm` | timm model backend, including the shared vision stack |
| `thingsvision` | `torch`, `transformers`, `numba`, `thingsvision` (Python 3.11–3.12) | thingsvision backend, including the shared vision stack |
| `mne` | `mne`, `mne-bids` | M/EEG analysis and BIDS support |
| `notebook` | `ipykernel`, `ipywidgets` | Jupyter notebooks |
| `cebra` | `cebra`, `trialcebra` | CEBRA integration |

For example:

```sh
uv add "vneurotk[vision]"       # DNN vision extraction: PyTorch + Transformers
uv add "vneurotk[viz]"          # recording visualization: Matplotlib
uv add "vneurotk[timm]"         # timm + shared vision dependencies
uv add "vneurotk[thingsvision]" # thingsvision + shared vision dependencies
uv add "vneurotk[mne]"          # M/EEG analysis
uv add "vneurotk[notebook]"     # Jupyter support
uv add "vneurotk[cebra]"        # CEBRA support
```

Multiple real extras can be installed together. For example, this installs M/EEG support, Jupyter, and recording visualization; it does not install DNN vision extraction:

```sh
uv add "vneurotk[mne,notebook,viz]"
```

The `thingsvision` extra is currently constrained to Python 3.11–3.12. ThingsVision 1.4.4 imports TensorFlow eagerly, and its current dependency stack is not runtime-compatible with Python 3.13 in this project; on newer Python versions the extra intentionally installs no backend dependencies rather than presenting an untested installation as supported.

The `torch` requirement in the vision-related extras is installable on CPU-only systems. If you need a CUDA- or accelerator-specific PyTorch build, follow the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for your platform before or while resolving the extra.

## From source

```sh
git clone https://github.com/colehank/vneurotk.git
cd vneurotk
uv sync
```

`uv sync` installs the core package only. Add the extras needed for local work:

```sh
uv sync --extra vision       # DNN vision extraction stack
uv sync --extra viz          # recording visualization
uv sync --extra timm         # timm backend + shared vision stack
uv sync --extra thingsvision # thingsvision backend + shared vision stack
```

## For contributors

Clone the repository and install the development tools:

```sh
git clone https://github.com/colehank/vneurotk.git
cd vneurotk
uv sync --group dev
```

The development group contains Ruff, pytest, coverage, ty, Sphinx, MyST-NB, numpydoc, and the PyData Sphinx Theme. Dependency groups and package extras can be combined:

```sh
uv sync --group dev --extra vision --extra viz
```

## Next steps

- [Visualize neural recordings](usage/viz) with `BaseData.plot` or `vneurotk.viz.plot_data` after installing `viz`.
- [Configure DNN vision models](usage/vision_models) after installing the backend extra you need.
- Compare the separate [Visualization API](api/viz.md) and [DNN Vision API](api/vision.md) before choosing extras.
