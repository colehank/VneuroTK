# Installation

## PyTorch

vneurotk requires [PyTorch](https://pytorch.org/get-started/locally/), but does not install it automatically to allow you to choose the version that matches your hardware. Install PyTorch **before** installing this package:

```sh
# uv（auto CUDA version detect）
uv pip install torch --torch-backend=auto
```

## Stable release

To install vneurotk:

```sh
uv add vneurotk
```

Or if you prefer `pip`, just replace `uv` with `pip install`:

```sh
pip install vneurotk
```

## Optional dependencies

```sh
uv add "vneurotk[mne]"          # M/EEG analysis: mne, mne-bids
uv add "vneurotk[notebook]"     # Jupyter: ipykernel, ipywidgets
uv add "vneurotk[timm]"         # timm models
uv add "vneurotk[thingsvision]" # thingsvision
uv add "vneurotk[cebra]"        # CEBRA
```

Multiple extras can be installed at once:

```sh
uv add "vneurotk[meg,notebook]"
```

## From source
```sh
git clone https://github.com/colehank/vneurotk
```

Once you have a copy of the source, you can install it with:

```sh
cd vneurotk
uv pip install torch --torch-backend=auto
uv sync
```

## For contributors

Clone and set up the development environment:

```bash
git clone https://github.com/colehank/vneurotk.git
cd vneurotk
uv pip install torch --torch-backend=auto
uv sync  # core only
```

Available additionals:

| Group/Extra | Contents | from |
|-------|----------|----------|
|default| core dependency|  numpy, h5py, transformers, ...
| `dev` | dev tools: ruff, pytest, coverage, ty, zensical, mkdocstrings-python | group |
| `mne` | M/EEG analysis: mne, mne-bids | extra |
| `notebook` | Jupyter: ipykernel, ipywidgets | extra |
| `timm` | timm models | extra |
| `thingsvision` | thingsvision | extra |
| `cebra` | CEBRA: cebra, trialcebra | extra |

To sync

```bash
uv sync --group dev      # dev tools
uv sync --extra mne      # MEG analysis
uv sync --extra notebook # Jupyter support
```

To sync groups/extras at once:

```bash
uv sync --group dev --extra mne --extra notebook
```