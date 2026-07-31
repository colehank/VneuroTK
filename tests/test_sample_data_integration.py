"""Bounded integration tests against the published sample archives."""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import vneurotk as vnt
from vneurotk.datasets import sample
from vneurotk.io import EphysPath, MNEPath
from vneurotk.utils import get_event_samples

pytestmark = [
    pytest.mark.sample_data,
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.slow,
]

_SESSION_ID = sample.EPHYS_SESSION_ID
_MONKEY_DTYPES = {
    "TrialRaster": ".h5",
    "TrialRecord": ".csv",
    "MeanFr": ".h5",
    "ChTrialRaster": ".h5",
    "ChTrialRecord": ".csv",
    "ChMeanFr": ".h5",
    "ChStimFr": ".h5",
    "UnitProp": ".csv",
    "ChProp": ".csv",
}


@pytest.fixture(scope="session")
def sample_cache() -> Path:
    """Use an explicit, job-local cache rather than the user's platform cache."""
    configured = os.environ.get("VNEUROTK_SAMPLE_CACHE")
    if configured:
        cache = Path(configured).expanduser()
    elif runner_temp := os.environ.get("RUNNER_TEMP"):
        cache = Path(runner_temp) / "vneurotk-sample-cache"
    else:
        cache = Path(tempfile.mkdtemp(prefix="vneurotk-sample-cache-"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture(scope="session")
def nod_root(sample_cache: Path) -> Path:
    root = sample.data_path("nod-meg", path=sample_cache, progressbar=False)
    return root / "nod-meg"


@pytest.fixture(scope="session")
def monkey_root(sample_cache: Path) -> Path:
    root = sample.data_path("monkey-vision", path=sample_cache, progressbar=False)
    return root / "monkey-vision"


@pytest.fixture(scope="session")
def monkey_session(monkey_root: Path) -> Path:
    return monkey_root / "sessions" / _SESSION_ID


def _ephys_path(monkey_root: Path, dtype: str) -> EphysPath:
    extension = "csv" if dtype in {"TrialRecord", "ChTrialRecord", "UnitProp", "ChProp"} else "h5"
    return EphysPath(root=monkey_root, session_id=_SESSION_ID, dtype=dtype, extension=extension)


def test_nod_archive_integrity(nod_root: Path):
    meg = nod_root / "meg" / "sub-01_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif"
    events = nod_root / "events" / "sub-01_events.csv"
    images = sorted((nod_root / "stimuli").glob("*.JPEG"))

    assert meg.is_file()
    assert events.is_file()
    assert len(images) == 200
    assert all(image.stat().st_size > 0 for image in images)


def test_monkey_archive_integrity(monkey_session: Path):
    expected = {f"{dtype}_{_SESSION_ID}{extension}" for dtype, extension in _MONKEY_DTYPES.items()}
    assert {path.name for path in monkey_session.iterdir() if path.is_file()} == expected

    for path in sorted(monkey_session.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            assert "data" in handle
    for path in sorted(monkey_session.glob("*.csv")):
        assert not pd.read_csv(path, nrows=2).empty


def test_nod_real_sample_smoke_stays_lazy(nod_root: Path):
    mne = pytest.importorskip("mne")
    mne_path = MNEPath(
        root=nod_root / "meg",
        subject=sample.NOD_SUBJECT,
        session=sample.NOD_SESSION,
        task=sample.NOD_TASK,
        run=sample.NOD_RUN,
        suffix="meg_clean",
        extension=".fif",
    )
    expected_path = nod_root / "meg" / "sub-01_ses-ImageNet01_task-ImageNet_run-01_meg_clean.fif"
    assert mne_path.fpath == expected_path

    data = vnt.read(mne_path, pre_load=False)
    assert data.data_mode == "continuous"
    assert data.neuro_info["shape"] == (80_000, 273)
    assert data.neuro_info["sfreq"] == 250.0
    assert data.ntime == 80_000
    assert data.nchan == 273
    assert "neuro=<lazy>" in repr(data)

    events = pd.read_csv(nod_root / "events" / "sub-01_events.csv")
    run_events = events.loc[
        (events["session"] == sample.NOD_SESSION) & (events["run"] == int(sample.NOD_RUN))
    ].reset_index(drop=True)
    stim_ids = run_events["image_id"].to_numpy()
    assert len(events) == 4_000
    assert len(run_events) == 200
    assert len(np.unique(stim_ids)) == 200

    image_paths = {stim_id: nod_root / "stimuli" / f"{stim_id}.JPEG" for stim_id in stim_ids}
    assert all(path.is_file() for path in image_paths.values())
    assert {path.stem for path in (nod_root / "stimuli").glob("*.JPEG")} == set(stim_ids)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw(mne_path.fpath, preload=False, verbose=False)
    assert raw.preload is False
    onsets = get_event_samples(raw, event_name="stim_on")
    assert onsets.shape == (200,)
    np.testing.assert_array_equal(onsets[[0, -1]], [1_846, 77_106])

    data.configure(
        stim_ids=stim_ids,
        trial_window=[-0.2, 0.8],
        vision_onsets=onsets,
        vision_db=image_paths,
    )
    assert data.is_configured
    assert data.n_trials == 200
    assert data.trial_info == {"baseline": [-50, 0], "trial_window": [-50, 200]}
    np.testing.assert_array_equal(data.trial_stim_ids, stim_ids)
    assert "neuro=<lazy>" in repr(data)

    decoded = {}
    for stim_id in sorted(stim_ids)[::99]:
        with data.vision.db[stim_id] as image:
            decoded[stim_id] = np.asarray(image.convert("RGB"))
    assert {key: (value.shape, value.dtype, int(value.sum())) for key, value in decoded.items()} == {
        "n01440764_26515": ((375, 375, 3), np.dtype("uint8"), 41_887_099),
        "n03127925_6450": ((375, 375, 3), np.dtype("uint8"), 47_184_981),
        "n12985857_535": ((375, 375, 3), np.dtype("uint8"), 34_108_700),
    }
    assert "neuro=<lazy>" in repr(data)


@pytest.mark.parametrize(
    ("dtype", "shape", "record_dtype", "property_dtype"),
    [
        ("TrialRaster", (50_932, 350, 333), "TrialRecord", "UnitProp"),
        ("ChTrialRaster", (50_932, 350, 384), "ChTrialRecord", "ChProp"),
    ],
)
def test_monkey_rasters_expose_metadata_without_materializing(
    monkey_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    shape: tuple[int, int, int],
    record_dtype: str,
    property_dtype: str,
):
    def reject_materialization(*_args, **_kwargs):
        pytest.fail("real sample raster must remain lazy")

    monkeypatch.setattr("vneurotk.io.loader._coo_to_dense", reject_materialization)
    data = vnt.read(_ephys_path(monkey_root, dtype), pre_load=False)

    assert data.data_mode == "epochs"
    assert data.neuro_info["shape"] == shape
    assert data.ntime == shape[1]
    assert data.nchan == shape[2]
    assert data.n_trials == shape[0]
    assert data.is_configured
    assert data.trial_meta.shape[0] == shape[0]
    assert data.trial_meta["id"].is_unique
    assert data.trial_meta["stim_index"].nunique() == 10_061
    assert len(data.neuro_info["ch_names"]) == shape[2]
    assert "neuro=<lazy>" in repr(data)

    record = pd.read_csv(_ephys_path(monkey_root, record_dtype).fpath)
    properties = pd.read_csv(_ephys_path(monkey_root, property_dtype).fpath)
    pd.testing.assert_series_equal(data.trial_meta["stim_index"], record["stim_index"])
    assert properties["id"].tolist() == list(range(shape[2]))


@pytest.mark.parametrize(
    ("dtype", "shape", "record_dtype", "property_dtype"),
    [
        ("MeanFr", (50_932, 333), "TrialRecord", "UnitProp"),
        ("ChMeanFr", (50_932, 384), "ChTrialRecord", "ChProp"),
    ],
)
def test_monkey_mean_fr_eager_shape_metadata_and_ids(
    monkey_root: Path,
    dtype: str,
    shape: tuple[int, int],
    record_dtype: str,
    property_dtype: str,
):
    data = vnt.read(_ephys_path(monkey_root, dtype), pre_load=False)
    record = pd.read_csv(_ephys_path(monkey_root, record_dtype).fpath)
    properties = pd.read_csv(_ephys_path(monkey_root, property_dtype).fpath)

    assert data.data_mode == "patterns"
    assert data.neuro.shape == shape
    assert np.isfinite(data.neuro.data[:8, :8]).all()
    assert data.is_configured
    assert data.trial_meta.shape == record.shape
    pd.testing.assert_series_equal(data.trial_meta["stim_index"], record["stim_index"])
    assert data.trial_stim_ids.tolist() == record["stim_index"].tolist()
    assert data.vision_info is not None
    assert data.vision_info["n_stim"] == 10_061
    assert data.neuro_info["ch_names"] == properties["id"].astype(str).tolist()


def test_monkey_ch_stim_fr_bounded_roundtrip(monkey_root: Path, tmp_path: Path):
    data = vnt.read(_ephys_path(monkey_root, "ChStimFr"), pre_load=False)
    properties = pd.read_csv(_ephys_path(monkey_root, "ChProp").fpath)

    assert data.data_mode == "patterns"
    assert data.neuro.shape == (10_060, 384)
    assert np.isfinite(data.neuro.data[:8, :8]).all()
    assert data.trial_meta["stim_index"].is_unique
    assert data.vision_info is not None
    assert data.trial_stim_ids.tolist() == data.vision_info["stim_ids"]
    assert len(data.vision_info["teststim"]) == 1_000
    assert set(data.vision_info["teststim"]) < set(data.vision_info["stim_ids"])
    assert data.neuro_info["ch_names"] == properties["id"].astype(str).tolist()

    row_count = 8
    channel_count = 16
    subset_ids = data.trial_stim_ids[:row_count]
    subset = vnt.BaseData.for_patterns(
        neuro=data.neuro.data[:row_count, :channel_count].copy(),
        neuro_info={
            "sfreq": None,
            "ch_names": data.neuro_info["ch_names"][:channel_count],
        },
        vision_info={"n_stim": row_count, "stim_ids": subset_ids.tolist()},
        trial_meta=pd.DataFrame({"stim_index": subset_ids}),
    )
    output = tmp_path / "ch-stim-fr-slice.h5"
    subset.save(output)
    roundtripped = vnt.read(output, pre_load=False)

    assert roundtripped.data_mode == "patterns"
    assert roundtripped._neuro is None
    assert tuple(roundtripped.neuro_info["shape"]) == (row_count, channel_count)
    np.testing.assert_array_equal(roundtripped.neuro, subset.neuro)
    np.testing.assert_array_equal(roundtripped.trial_stim_ids, subset_ids)
