"""VneuroTK sample datasets.

Usage
-----
>>> from vneurotk.datasets import sample
>>> root = sample.data_path("nod-meg")          # download only NOD-MEG
>>> root = sample.data_path("monkey-vision")    # download only MonkeyVision
>>> root = sample.data_path()                   # download all datasets

Directory layout after download
--------------------------------
``data_path()`` returns the extracted root that contains both sub-trees::

    <root>/
    ├── nod-meg/
    │   ├── meg/
    │   │   └── sub-01_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif
    │   ├── events/
    │   │   └── sub-01_events.csv
    │   └── stimuli/
    │       └── <image_id>.JPEG  (200 images from run 01)
    └── monkey-vision/
        └── sessions/
            └── 251024_FanFan_nsd1w_MSB/
                ├── TrialRaster_251024_FanFan_nsd1w_MSB.h5
                ├── TrialRecord_251024_FanFan_nsd1w_MSB.csv
                ├── MeanFr_251024_FanFan_nsd1w_MSB.h5
                ├── ChMeanFr_251024_FanFan_nsd1w_MSB.h5
                ├── ChStimFr_251024_FanFan_nsd1w_MSB.h5
                ├── UnitProp_251024_FanFan_nsd1w_MSB.csv
                └── ChProp_251024_FanFan_nsd1w_MSB.csv
"""

from __future__ import annotations

from pathlib import Path

import pooch

# ---------------------------------------------------------------------------
# Zenodo configuration
# ---------------------------------------------------------------------------
_ZENODO_RECORD = "20094167"
_BASE_URL = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/"

_NOD_FNAME = "vneurotk-nod-meg-sample.zip"
_NOD_HASH = "md5:66118e0e5432376ea8418bf8b35f470f"

_MV_FNAME = "vneurotk-monkey-vision-sample.zip"
_MV_HASH = "md5:d17ecae87ffd93e95838159f74b2d82f"

# Both zips are extracted into this sub-directory inside the cache root.
_EXTRACT_DIR = "vneurotk-samples"

# Mapping from user-facing dataset name to (filename, hash).
_DATASETS: dict[str, tuple[str, str]] = {
    "nod-meg": (_NOD_FNAME, _NOD_HASH),
    "monkey-vision": (_MV_FNAME, _MV_HASH),
}

# ---------------------------------------------------------------------------
# NOD-MEG metadata constants
# ---------------------------------------------------------------------------
NOD_SUBJECT = "01"
NOD_SESSION = "ImageNet01"
NOD_TASK = "ImageNet"
NOD_RUN = "01"

# ---------------------------------------------------------------------------
# MonkeyVision metadata constants
# ---------------------------------------------------------------------------
EPHYS_SESSION_ID = "251024_FanFan_nsd1w_MSB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def data_path(
    dataset: str | list[str] | None = None,
    path: str | Path | None = None,
    progressbar: bool = True,
) -> Path:
    """Return the root directory of VneuroTK sample datasets.

    Downloads and caches on first call; subsequent calls return immediately.
    The default cache location is ``~/.cache/vneurotk/`` (platform-specific).

    Zip files are fetched from Zenodo (DOI 10.5281/zenodo.20094167) and
    extracted into a shared ``vneurotk-samples/`` sub-directory so you can
    navigate all datasets with a single root path.

    Parameters
    ----------
    dataset : str or list of str or None
        Which dataset(s) to download.  Accepted values:

        * ``"nod-meg"``        – NOD MEG recording + stimuli (~91 MB)
        * ``"monkey-vision"``  – MonkeyVision ephys recording (~50 MB)
        * a list combining the above names
        * ``None`` (default)   – download **all** datasets

    path : str or Path or None
        Override the default cache directory.
    progressbar : bool
        Show a tqdm progress bar while downloading.  Requires ``tqdm``.
        Defaults to ``True``.

    Returns
    -------
    Path
        Root directory containing ``nod-meg/`` and/or ``monkey-vision/``
        sub-trees.  Navigate with standard path operations::

            root = sample.data_path("nod-meg")
            nod  = root / "nod-meg"

    Examples
    --------
    >>> from vneurotk.datasets import sample
    >>> from vneurotk.io import MNEPath, EphysPath
    >>>
    >>> # download only the MEG dataset
    >>> root = sample.data_path("nod-meg")
    >>> mne_path = MNEPath(
    ...     root=root / "nod-meg" / "meg",
    ...     subject=sample.NOD_SUBJECT,
    ...     session=sample.NOD_SESSION,
    ...     task=sample.NOD_TASK,
    ...     run=sample.NOD_RUN,
    ...     suffix="meg_clean",
    ...     extension=".fif",
    ... )
    >>>
    >>> # download only the Ephys dataset
    >>> root = sample.data_path("monkey-vision")
    >>> raster_path = EphysPath(
    ...     root=root / "monkey-vision",
    ...     session_id=sample.EPHYS_SESSION_ID,
    ...     dtype="TrialRaster",
    ... )
    """
    valid = set(_DATASETS)
    if dataset is None:
        names: list[str] = list(_DATASETS)
    elif isinstance(dataset, str):
        if dataset not in valid:
            raise ValueError(f"Unknown dataset {dataset!r}. Choose from {sorted(valid)}.")
        names = [dataset]
    else:
        unknown = set(dataset) - valid
        if unknown:
            raise ValueError(f"Unknown dataset(s) {sorted(unknown)}. Choose from {sorted(valid)}.")
        names = list(dataset)

    cache = Path(path) if path else pooch.os_cache("vneurotk")
    downloader = pooch.HTTPDownloader(progressbar=progressbar)
    fetcher = pooch.create(
        path=cache,
        base_url=_BASE_URL,
        registry={fname: hash_ for fname, hash_ in _DATASETS.values()},
    )

    for name in names:
        fname, _ = _DATASETS[name]
        fetcher.fetch(
            fname,
            downloader=downloader,
            processor=pooch.Unzip(extract_dir=_EXTRACT_DIR),
        )

    return cache / _EXTRACT_DIR
