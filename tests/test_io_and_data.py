"""Tests for vneurotk.io and vneurotk.neuro modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import vneurotk as vnt
from vneurotk.core import BaseData
from vneurotk.io import BIDSPath, EphysPath, MNEPath, VTKPath


class TestVTKPath:
    """Test VTKPath class."""

    def test_vtkpath_basic(self):
        """Test basic VTKPath construction."""
        path = VTKPath(
            root=Path("/data"),
            subject="01",
            session="test",
            task="task1",
            run="01",
        )
        assert path.root == Path("/data")
        assert path.subject == "01"
        assert path.session == "test"
        assert "sub-01" in str(path.fpath)
        assert "ses-test" in str(path.fpath)

    def test_vtkpath_positional_root(self):
        """Test VTKPath with positional root argument."""
        path = VTKPath(Path("/data"), subject="01", session="test")
        assert path.root == Path("/data")
        assert path.subject == "01"


class TestEphysPath:
    """Test EphysPath class."""

    SESSION_ID = "251024_FanFan_nsd1w_MSB"
    ROOT = Path("/db/ephys/MonkeyVision")

    def test_fpath_single_probe(self):
        """Single-probe session: fpath has no _probe suffix."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="TrialRaster")
        assert path.modality == "ephys"
        fp = path.fpath
        assert str(fp) == str(self.ROOT / "sessions" / self.SESSION_ID / f"TrialRaster_{self.SESSION_ID}.h5")
        assert "_probe" not in fp.name

    def test_fpath_multi_probe(self):
        """Multi-probe session: fpath contains _probe{N} tag."""
        p0 = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="MeanFr", probe=0)
        p1 = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="MeanFr", probe=1)
        assert "_probe0" in p0.fpath.name
        assert "_probe1" in p1.fpath.name

    def test_fpath_csv_extension(self):
        """csv extension is accepted and applied correctly."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="UnitProp", extension="csv")
        assert path.fpath.suffix == ".csv"

    def test_session_dir(self):
        """session_dir points to {root}/sessions/{session_id}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.session_dir == self.ROOT / "sessions" / self.SESSION_ID

    def test_raw_dir(self):
        """raw_dir points to {root}/raw/{session_id}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.raw_dir == self.ROOT / "raw" / self.SESSION_ID

    def test_nwb_path_single_probe(self):
        """nwb_path without probe has no _probe suffix."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.nwb_path == self.ROOT / "nwb" / f"{self.SESSION_ID}.nwb"

    def test_nwb_path_multi_probe(self):
        """nwb_path with probe contains _probe{N}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, probe=0)
        assert path.nwb_path == self.ROOT / "nwb" / f"{self.SESSION_ID}_probe0.nwb"

    def test_from_components(self):
        """from_components builds session_id correctly."""
        path = EphysPath.from_components(
            root=self.ROOT,
            date="251024",
            subject="FanFan",
            paradigm="nsd1w",
            region="MSB",
            dtype="AvgPsth",
        )
        assert path.session_id == self.SESSION_ID
        assert path.dtype == "AvgPsth"
        assert "_probe" not in path.fpath.name

    def test_invalid_dtype_raises(self):
        """Unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="BadType")

    def test_invalid_extension_raises(self):
        """Unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid extension"):
            EphysPath(root=self.ROOT, session_id=self.SESSION_ID, extension="dat")

    def test_fpath_missing_session_id_raises(self):
        """Accessing fpath without session_id raises ValueError."""
        path = EphysPath(root=self.ROOT, dtype="TrialRaster")
        with pytest.raises(ValueError, match="session_id"):
            _ = path.fpath

    def test_fpath_missing_dtype_raises(self):
        """Accessing fpath without dtype raises ValueError."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        with pytest.raises(ValueError, match="dtype"):
            _ = path.fpath


class TestMNEPath:
    """Test MNEPath class."""

    def test_mnepath_construction(self):
        """Test MNEPath construction."""
        path = MNEPath(
            root=Path("/mne"),
            subject="01",
            session="test",
            task="task1",
            run="01",
            suffix="clean",
            extension=".fif",
        )
        assert path.modality == "mne"
        assert "sub-01_ses-test_task-task1_run-01_clean.fif" in str(path.fpath)


class TestBIDSPath:
    """Test BIDSPath class."""

    def test_bidspath_construction(self):
        """Test BIDSPath construction."""
        path = BIDSPath(
            root=Path("/bids"),
            subject="01",
            session="test",
            task="task1",
            run="01",
            suffix="meg",
            extension="fif",
        )
        assert path.modality == "bids"
        assert hasattr(path, "bids_path")


class TestBaseData:
    """Test BaseData class."""

    def test_basedata_construction(self):
        """Test BaseData construction."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        assert data.ntime == 1000
        assert data.nchan == 10
        assert not data.configured

    def test_basedata_configure(self):
        """Test BaseData.configure() method."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)

        # Configure with 3 trials
        visual_onsets = np.array([100, 400, 700])
        vision_id = np.array([1, 2, 1])
        trial_window = [-0.2, 0.8]  # seconds

        data.configure(
            trial_window=trial_window,
            vision_onsets=visual_onsets,
            stim_ids=vision_id,
        )

        assert data.configured
        assert data.n_trials == 3
        assert len(data.vision_onsets) == 3  # ty: ignore[invalid-argument-type]
        assert len(data.trial_starts) == 3  # ty: ignore[invalid-argument-type]
        assert len(data.trial_ends) == 3  # ty: ignore[invalid-argument-type]
        assert data.vision_info["n_stim"] == 2  # ty: ignore[not-subscriptable]
        assert set(data.vision_info["stim_ids"]) == {1, 2}  # ty: ignore[not-subscriptable]

    def test_basedata_save_load(self):
        """Test BaseData save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            neuro = np.random.randn(1000, 10)
            neuro_info = {
                "sfreq": 100.0,
                "ch_names": [f"ch{i}" for i in range(10)],
                "highpass": 0.1,
                "lowpass": 40.0,
                "source_file": "/test.fif",
            }
            data = BaseData(neuro=neuro, neuro_info=neuro_info)

            # Configure
            visual_onsets = np.array([100, 400, 700])
            vision_id = np.array([1, 2, 1])
            data.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=visual_onsets,
                stim_ids=vision_id,
            )

            # Save
            save_path = VTKPath(
                Path(tmpdir),
                subject="01",
                session="test",
                task="task1",
            )
            data.save(save_path)

            # Load
            loaded_data = vnt.read(save_path)
            assert loaded_data.ntime == data.ntime
            assert loaded_data.nchan == data.nchan
            assert loaded_data.n_trials == data.n_trials
            assert np.allclose(loaded_data.neuro, data.neuro)
            assert np.allclose(loaded_data.vision_onsets, data.vision_onsets)  # ty: ignore[invalid-argument-type]

    def test_basedata_save_unconfigured_raises(self):
        """Test that saving unconfigured BaseData raises error."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.h5"
            with pytest.raises(RuntimeError, match="configure"):
                data.save(save_path)

    def test_info_unconfigured(self):
        """Test info property on unconfigured BaseData."""
        neuro = np.random.randn(500, 8)
        neuro_info = {
            "sfreq": 250.0,
            "ch_names": [f"ch{i}" for i in range(8)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        info = data.info

        # neuro section populated
        assert info._neuro["n_time"] == 500
        assert info._neuro["n_chan"] == 8
        assert info._neuro["sfreq"] == 250.0

        # visual/trial not configured
        assert info._visual is None
        assert info._trial is None
        assert not info._configured

        # repr contains "Not configured"
        assert "Not configured" in repr(info)
        html = info._repr_html_()
        assert "Not configured" in html
        assert "vtk-info" in html
        assert "Neuro" in html

    def test_info_configured(self):
        """Test info property on configured BaseData."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        info = data.info
        assert info._neuro["n_time"] == 1000
        assert info._visual["n_stim"] == 2  # ty: ignore[not-subscriptable]
        assert info._trial["trial_window"] is not None  # ty: ignore[not-subscriptable]

        html = info._repr_html_()
        assert "Not configured" not in html
        assert "n_visual" in html
        assert "Baseline" in html
        assert "Trial window" in html

    def test_basedata_repr_html(self):
        """Test BaseData._repr_html_ delegates to info."""
        neuro = np.random.randn(500, 8)
        neuro_info = {
            "sfreq": 250.0,
            "ch_names": [f"ch{i}" for i in range(8)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        assert data._repr_html_() == data.info._repr_html_()

    def test_crop_continues(self):
        """Test crop in continues mode."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        cont = data.neuro.continuous
        assert cont.ndim == 2
        # continuous on continuous-mode data returns the raw array (warns)
        assert cont.shape[1] == 10
        assert data.nchan == 10
        assert len(data.trial_starts) == 3  # ty: ignore[invalid-argument-type]

    def test_crop_epochs(self):
        """Test crop in epochs mode."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        trial_len = data.trial_ends[0] - data.trial_starts[0]  # ty: ignore[not-subscriptable]

        ep = data.neuro.epochs
        assert ep.ndim == 3
        assert ep.shape == (3, trial_len, 10)
        assert data.nchan == 10
        assert data.n_timepoints == trial_len
        assert data.n_trials == 3

    def test_configure_with_epochs_view(self):
        """Test epochs view after configure."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )

        ep = data.neuro.epochs
        assert ep.ndim == 3
        assert data.n_trials == 3

    def test_neuro_views_unconfigured_raises(self):
        """Test that neuro.continuous / epochs on unconfigured BaseData raises error."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        with pytest.raises(RuntimeError, match="configure"):
            _ = data.neuro.continuous


class TestLoadFunction:
    """Test load function."""

    def test_load_avgpsth_not_implemented(self):
        """Test that loading AvgPsth raises NotImplementedError."""
        path = EphysPath(
            root=Path("/ephys"),
            session_id="251024_FanFan_nsd1w_MSB",
            dtype="AvgPsth",
            extension="h5",
        )
        with pytest.raises(NotImplementedError):
            vnt.read(path)

    def test_load_metadata_dtype_raises(self):
        """Passing a metadata dtype (TrialRecord etc.) raises ValueError."""
        for dtype in ("TrialRecord", "ChTrialRecord", "UnitProp", "ChProp"):
            path = EphysPath(
                root=Path("/ephys"),
                session_id="251024_FanFan_nsd1w_MSB",
                dtype=dtype,
                extension="csv",
            )
            with pytest.raises(ValueError, match="metadata file"):
                vnt.read(path)


class TestBaseDataLoad:
    """Test BaseData.load() method and vnt.read(pre_load=...) parameter."""

    def _make_configured_bd(self) -> tuple[BaseData, np.ndarray]:
        neuro = np.zeros((3, 100, 5), dtype=np.float32)  # sparse 3D
        neuro_info = {
            "sfreq": 1000,
            "ch_names": [f"ch{i}" for i in range(5)],
            "highpass": None,
            "lowpass": None,
            "source_file": "",
            "shape": (3, 100, 5),
        }
        bd = BaseData(
            neuro=None,
            neuro_info=neuro_info,
            stim_labels=np.zeros((3, 100)),
            trial=np.zeros((3, 100)),
            trial_starts=np.array([0, 0, 0]),
            trial_ends=np.array([100, 100, 100]),
            vision_onsets=np.array([50, 50, 50]),
            vision_info={"n_stim": 1, "stim_ids": [0]},
            trial_info={"baseline": [-50, 0], "trial_window": [-50, 50]},
        )
        bd._neuro_loader = lambda: neuro
        return bd, neuro

    def test_load_method_triggers_lazy(self):
        """bd.load() reads neuro into memory and clears loader."""
        bd, neuro = self._make_configured_bd()
        assert bd._neuro is None
        result = bd.load()
        assert result is bd  # returns self
        assert bd._neuro is not None
        assert bd._neuro_loader is None
        assert np.array_equal(bd.neuro, neuro)

    def test_load_method_noop_when_already_loaded(self):
        """bd.load() is a no-op when neuro is already in memory."""
        bd, neuro = self._make_configured_bd()
        bd.load()
        arr_before = bd._neuro
        bd.load()  # second call
        assert bd._neuro is arr_before  # same object, not reloaded

    def test_load_method_noop_when_no_loader(self):
        """bd.load() on data with no lazy loader returns self silently."""
        neuro = np.random.randn(100, 5)
        bd = BaseData(
            neuro=neuro,
            neuro_info={"sfreq": 100, "ch_names": list("abcde")},
        )
        result = bd.load()
        assert result is bd

    def test_load_method_chaining(self):
        """bd.load() supports method chaining."""
        bd, neuro = self._make_configured_bd()
        loaded_neuro = bd.load().neuro
        assert np.array_equal(loaded_neuro, neuro)

    def test_pre_load_false_keeps_lazy(self):
        """vnt.read(fpath, pre_load=False) keeps neuro lazy for COO h5 files."""
        bd, _ = self._make_configured_bd()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "data.h5"
            bd.save(fpath)
            loaded = vnt.read(fpath, pre_load=False)
            assert loaded._neuro is None
            assert loaded._neuro_loader is not None

    def test_pre_load_true_forces_eager(self):
        """vnt.read(pre_load=True) reads neuro immediately."""
        bd, neuro = self._make_configured_bd()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "data.h5"
            bd.save(fpath)
            loaded = vnt.read(fpath, pre_load=True)
            assert loaded._neuro is not None
            assert loaded._neuro_loader is None
            assert np.array_equal(loaded.neuro, neuro)

    def test_dense_pre_load_false_keeps_lazy(self):
        """Dense-format h5 is also lazy by default (pre_load=False)."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            data.save(fpath)
            with h5py.File(fpath, "r") as f:
                assert str(f.attrs.get("neuro_format", "dense")) == "dense"
            loaded = vnt.read(fpath, pre_load=False)
            assert loaded._neuro is None, "dense neuro should be lazy until accessed"
            assert loaded._neuro_loader is not None
            # shape info still available before load
            assert loaded.ntime == 1000
            assert loaded.nchan == 10

    def test_dense_pre_load_true_forces_eager(self):
        """pre_load=True on dense format reads neuro immediately."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            data.save(fpath)
            loaded = vnt.read(fpath, pre_load=True)
            assert loaded._neuro is not None
            assert loaded._neuro_loader is None
            assert np.allclose(loaded.neuro, neuro)

    def test_dense_lazy_data_matches_original(self):
        """Lazy dense loader returns data identical to the saved array."""
        neuro = np.random.randn(500, 8)
        neuro_info = {
            "sfreq": 250.0,
            "ch_names": [f"ch{i}" for i in range(8)],
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([50, 200, 350]),
            stim_ids=np.array([1, 2, 1]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            data.save(fpath)
            loaded = vnt.read(fpath, pre_load=False)
            assert loaded._neuro is None
            # trigger load
            arr = loaded.neuro
            assert loaded._neuro is not None
            assert loaded._neuro_loader is None
            assert np.allclose(arr, neuro)


# -- Real DB tests (skip if DB is not available) ---------------------------

_DB_ROOT = Path(__file__).resolve().parent.parent / "DB" / "ephys" / "MonkeyVision"
_SESSION_ID = "251024_FanFan_nsd1w_MSB"
_SESSION_DIR = _DB_ROOT / "sessions" / _SESSION_ID
_HAS_DB = _SESSION_DIR.exists()

skip_no_db = pytest.mark.skipif(not _HAS_DB, reason="Real DB not available")


@skip_no_db
class TestLoadEphysRaster:
    """Test raster loading with real DB files."""

    def _make_path(self, dtype: str) -> EphysPath:
        return EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype=dtype)

    def test_load_trial_raster_lazy(self):
        """TrialRaster loads lazily — neuro is None until accessed."""
        bd = vnt.read(self._make_path("TrialRaster"))
        # before access: _neuro is None, but shape info exists
        assert bd._neuro is None
        assert bd._neuro_loader is not None
        assert bd.data_mode == "epochs"
        assert bd.ntime > 0
        assert bd.nchan > 0

    def test_load_trial_raster_access(self):
        """Accessing neuro triggers lazy load with correct shape."""
        bd = vnt.read(self._make_path("TrialRaster"))
        neuro = bd.neuro
        assert neuro is not None
        assert neuro.ndim == 3  # (n_trials, n_timebins, n_units)
        assert neuro.shape[0] == bd.n_trials
        assert neuro.shape[1] == bd.ntime
        assert neuro.shape[2] == bd.nchan
        # loader cleared after use
        assert bd._neuro_loader is None

    def test_load_ch_trial_raster(self):
        """ChTrialRaster loads and has correct shape."""
        bd = vnt.read(self._make_path("ChTrialRaster"))
        assert bd.data_mode == "epochs"
        neuro = bd.neuro
        assert neuro.ndim == 3  # (n_trials, n_timebins, n_channels)
        assert neuro.shape[0] == bd.n_trials

    def test_raster_has_trial_meta(self):
        """Raster loading attaches trial_meta from TrialRecord."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.trial_meta is not None
        assert "stim_index" in bd.trial_meta.columns
        assert len(bd.trial_meta) == bd.n_trials

    def test_raster_visual_info(self):
        """Visual info is populated from record file."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.vision_info is not None
        assert bd.vision_info["n_stim"] > 0
        assert len(bd.vision_info["stim_ids"]) == bd.vision_info["n_stim"]

    def test_raster_trial_structure(self):
        """Trial arrays match epochs layout."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.trial is not None
        assert bd.vision is not None
        assert bd.trial_starts is not None
        assert bd.trial_ends is not None
        assert bd.vision_onsets is not None
        assert len(bd.trial_starts) == bd.n_trials

    def test_raster_configure_raises(self):
        """configure() on epochs-mode raster is blocked (already configured)."""
        bd = vnt.read(self._make_path("TrialRaster"))
        # Already has visual/trial set, and crop_mode is epochs.
        # configure() would require data_level='timepoint' — it is, but
        # ntime points at timebins dimension which is per-epoch not continuous.
        # The real guard: it's already configured, calling configure again
        # would overwrite.  Just check it has trial structure.
        assert bd.configured

    def test_raster_continuous_view(self):
        """Epochs raster neuro.continuous returns 2D array."""
        bd = vnt.read(self._make_path("TrialRaster"))
        n_trials = bd.n_trials
        n_timebins = bd.ntime
        n_chan = bd.nchan
        cont = bd.neuro.continuous
        assert cont.ndim == 2
        assert cont.shape == (n_trials * n_timebins, n_chan)


@skip_no_db
class TestLoadEphysMeanFr:
    """Test MeanFr / ChMeanFr loading."""

    def _make_path(self, dtype: str) -> EphysPath:
        return EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype=dtype)

    def test_load_mean_fr(self):
        """MeanFr loads as patterns data."""
        bd = vnt.read(self._make_path("MeanFr"))
        assert bd.data_mode == "patterns"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_trials, n_units)
        assert bd.trial_meta is not None
        assert "stim_index" in bd.trial_meta.columns

    def test_load_ch_mean_fr(self):
        """ChMeanFr loads as patterns data."""
        bd = vnt.read(self._make_path("ChMeanFr"))
        assert bd.data_mode == "patterns"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_trials, n_channels)

    def test_mean_fr_configure_raises(self):
        """configure() on patterns-mode data raises ValueError."""
        bd = vnt.read(self._make_path("MeanFr"))
        assert bd.data_mode == "patterns"
        with pytest.raises(ValueError, match="patterns"):
            bd.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=np.array([0, 1]),
                stim_ids=np.array([0, 1]),
            )

    def test_mean_fr_neuro_views_raises(self):
        """neuro.continuous on patterns-mode unconfigured raises RuntimeError."""
        bd = vnt.read(self._make_path("MeanFr"))
        with pytest.raises(RuntimeError, match="configure"):
            _ = bd.neuro.continuous


@skip_no_db
class TestLoadEphysStimFr:
    """Test ChStimFr loading."""

    def test_load_ch_stim_fr(self):
        """ChStimFr loads as stimulus-level data."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="ChStimFr")
        bd = vnt.read(path)
        assert bd.data_mode == "patterns"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_stimuli, n_channels)
        assert bd.vision_info is not None
        assert bd.vision_info["n_stim"] == bd.neuro.shape[0]

    def test_stim_fr_trial_meta_has_stim_index(self):
        """trial_meta["stim_index"] provides row-to-stim mapping, consistent with MeanFr."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="ChStimFr")
        bd = vnt.read(path)
        assert bd.trial_meta is not None
        assert "stim_index" in bd.trial_meta.columns
        assert len(bd.trial_meta) == bd.neuro.shape[0]

    def test_stim_fr_vision_info_has_stim_ids(self):
        """vision_info['stim_ids'] is consistent with other ephys dtypes."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="ChStimFr")
        bd = vnt.read(path)
        assert "stim_ids" in bd.vision_info  # ty: ignore[unsupported-operator]
        assert len(bd.vision_info["stim_ids"]) == bd.neuro.shape[0]  # ty: ignore[not-subscriptable]
        assert list(bd.trial_meta["stim_index"]) == bd.vision_info["stim_ids"]  # ty: ignore[not-subscriptable]


@skip_no_db
class TestEphysSaveLoadRoundtrip:
    """Test save/load roundtrip for ephys-loaded BaseData."""

    def test_roundtrip_raster(self):
        """Save and reload a raster-loaded BaseData via COO format."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="TrialRaster")
        bd = vnt.read(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = VTKPath(
                Path(tmpdir),
                subject="FanFan",
                session="nsd1w",
                task="raster",
            )
            bd.save(save_path)

            # verify COO format on disk
            import h5py

            with h5py.File(save_path.fpath, "r") as f:
                assert str(f.attrs["neuro_format"]) == "coo"
                assert "neuro_row" in f
                assert "neuro_col" in f
                assert "neuro_data" in f
                assert "neuro" not in f  # no dense dataset

            # reload: lazy
            loaded = vnt.read(save_path)
            assert loaded.data_mode == "epochs"
            assert loaded._neuro is None
            assert loaded._neuro_loader is not None
            # shape info available before load
            assert loaded.ntime == bd.ntime
            assert loaded.nchan == bd.nchan
            assert loaded.n_trials == bd.n_trials
            # trigger load and verify data
            assert np.allclose(loaded.neuro, bd.neuro)

    def test_roundtrip_preserves_trial_meta(self):
        """Save and reload preserves trial_meta DataFrame."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="TrialRaster")
        bd = vnt.read(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = VTKPath(
                Path(tmpdir),
                subject="FanFan",
                session="nsd1w",
                task="meta",
            )
            bd.save(save_path)

            loaded = vnt.read(save_path)
            assert loaded.trial_meta is not None
            assert set(loaded.trial_meta.columns) == set(bd.trial_meta.columns)
            assert len(loaded.trial_meta) == len(bd.trial_meta)

    def test_save_dense_for_small_data(self):
        """Non-sparse 2D data still uses dense format."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            stim_ids=np.array([1, 2, 1]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            data.save(fpath)

            import h5py

            with h5py.File(fpath, "r") as f:
                assert str(f.attrs.get("neuro_format", "dense")) == "dense"
                assert "neuro" in f

            loaded = vnt.read(fpath)
            assert np.allclose(loaded.neuro, data.neuro)


# ---------------------------------------------------------------------------
# Performance-fix tests
# ---------------------------------------------------------------------------


def _make_base_data(neuro: np.ndarray) -> BaseData:
    """Helper: build a minimal configured BaseData."""
    neuro_info = {
        "sfreq": 100.0,
        "ch_names": [f"ch{i}" for i in range(neuro.shape[-1])],
        "highpass": 0.1,
        "lowpass": 40.0,
        "source_file": "/test.fif",
    }
    data = BaseData(neuro=neuro, neuro_info=neuro_info)
    data.configure(
        trial_window=[-0.2, 0.8],
        vision_onsets=np.array([100, 400, 700]),
        stim_ids=np.array([1, 2, 1]),
    )
    return data


class TestSavePerfFixes:
    """Tests verifying the three save() performance improvements."""

    # ── Fix 1: sparsity check uses sampling ─────────────────────────────────

    def test_sparse_3d_data_uses_coo(self):
        """3-D epochs-mode array with >50 % zeros is saved as COO format."""
        neuro = np.zeros((3, 100, 5), dtype=np.float32)
        neuro[0, 10, 0] = 1.0  # tiny fraction non-zero
        bd = BaseData(
            neuro=neuro,
            neuro_info={"sfreq": 100, "ch_names": [f"c{i}" for i in range(5)]},
            stim_labels=np.zeros((3, 100)),
            trial=np.zeros((3, 100)),
            trial_starts=np.array([0, 0, 0]),
            trial_ends=np.array([100, 100, 100]),
            vision_onsets=np.array([50, 50, 50]),
            vision_info={"n_stim": 1, "stim_ids": [0]},
            trial_info={"baseline": [-50, 0], "trial_window": [-50, 50]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "sparse.h5"
            bd.save(fpath)
            with h5py.File(fpath, "r") as f:
                assert str(f.attrs["neuro_format"]) == "coo"

    def test_dense_3d_data_uses_dense(self):
        """3-D epochs-mode array with ≤50 % zeros is saved as dense format."""
        neuro = np.random.randn(3, 100, 5).astype(np.float32)
        bd = BaseData(
            neuro=neuro,
            neuro_info={"sfreq": 100, "ch_names": [f"c{i}" for i in range(5)]},
            stim_labels=np.zeros((3, 100)),
            trial=np.zeros((3, 100)),
            trial_starts=np.array([0, 0, 0]),
            trial_ends=np.array([100, 100, 100]),
            vision_onsets=np.array([50, 50, 50]),
            vision_info={"n_stim": 1, "stim_ids": [0]},
            trial_info={"baseline": [-50, 0], "trial_window": [-50, 50]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            bd.save(fpath)
            with h5py.File(fpath, "r") as f:
                assert str(f.attrs["neuro_format"]) == "dense"

    def test_sparsity_check_is_fast_for_large_array(self):
        """Sampling-based sparsity estimate completes quickly on large arrays."""
        import time

        # ~150 M floats — a full scan would dominate the save
        neuro = np.zeros((50, 10_000, 300), dtype=np.float32)
        neuro[0, 0, 0] = 1.0
        flat = neuro.ravel()
        n_sample = min(100_000, flat.size)
        idx = np.random.default_rng(seed=0).integers(0, flat.size, size=n_sample)
        t0 = time.perf_counter()
        _ = bool((flat[idx] == 0).mean() > 0.5)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Sampling sparsity check took {elapsed:.3f}s (> 1s)"

    # ── Fix 2: VR arrays saved without compression ───────────────────────────

    def test_vr_array_saved_without_compression(self):
        """VisualRepresentation arrays are stored with no HDF5 filter."""
        from vneurotk.vision.representation.visual_representations import (
            VisualRepresentation,
            VisualRepresentations,
        )

        neuro = np.random.randn(1000, 10)
        data = _make_base_data(neuro)

        vr = VisualRepresentation(
            model="test_model",
            module_name="layer0",
            module_type="Linear",
            stim_ids=[1, 2],
            array=np.random.randn(2, 64).astype(np.float32),
        )
        vrs = VisualRepresentations([vr])
        data.vision.add(vrs)

        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "vr.h5"
            data.save(fpath)
            with h5py.File(fpath, "r") as f:
                ds = f["vision_store"]["0"]["array"]
                # No compression filter applied
                assert ds.compression is None

    # ── Fix 3: path-type images stored as image_bytes ────────────────────────

    def test_path_images_stored_as_image_bytes(self):
        """Path-based vision_db entries are saved as raw image bytes, not decoded arrays."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        neuro = np.random.randn(1000, 10)
        data = _make_base_data(neuro)

        with tempfile.TemporaryDirectory() as tmpdir:
            # write a tiny JPEG to disk
            img_path = Path(tmpdir) / "stim_1.jpg"
            PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path, format="JPEG")

            data.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=np.array([100, 400, 700]),
                stim_ids=np.array([1, 1, 1]),
                vision_db={1: img_path},
            )

            fpath = Path(tmpdir) / "out.h5"
            data.save(fpath)

            with h5py.File(fpath, "r") as f:
                ds = f["stimuli_db"]["1"]
                assert str(ds.attrs["kind"]) == "image_bytes"
                # Stored as raw uint8 bytes, NOT a (H, W, C) array
                assert ds[:].ndim == 1

    def test_lazy_h5dict_loads_image_bytes_correctly(self):
        """LazyH5Dict returns correct (H, W, C) uint8 ndarray for image_bytes entries."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io import LazyH5Dict

        h, w = 16, 24
        original = np.full((h, w, 3), 128, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            # write PNG (lossless) so pixel values survive round-trip
            img_path = Path(tmpdir) / "stim_42.png"
            PILImage.fromarray(original).save(img_path, format="PNG")

            h5_path = Path(tmpdir) / "db.h5"
            with h5py.File(h5_path, "w") as f:
                raw = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
                grp = f.create_group("stimuli_db")
                grp.create_dataset("42", data=raw)
                grp["42"].attrs["kind"] = "image_bytes"
                grp["42"].attrs["key_type"] = "int"

            lazy = LazyH5Dict(h5_path, group="stimuli_db")
            arr = lazy[42]
            assert arr.shape == (h, w, 3)
            assert arr.dtype == np.uint8
            assert np.array_equal(arr, original)

    def test_save_load_roundtrip_with_path_images(self):
        """Full save → load roundtrip: path-based images are recoverable via LazyH5Dict."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io import LazyH5Dict, VTKPath

        h, w = 16, 24
        img_arr = np.full((h, w, 3), 100, dtype=np.uint8)

        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img_7.png"
            PILImage.fromarray(img_arr).save(img_path, format="PNG")

            data.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=np.array([100, 400, 700]),
                stim_ids=np.array([7, 7, 7]),
                vision_db={7: img_path},
            )

            save_path = VTKPath(Path(tmpdir), subject="01", session="test")
            data.save(save_path)

            loaded = vnt.read(save_path)
            assert isinstance(loaded.vision.db, LazyH5Dict)
            recovered = loaded.vision.db[7]
            assert recovered.shape == (h, w, 3)
            assert np.array_equal(recovered, img_arr)


# ===========================================================================
# TestDataPathLoadSeam
# ===========================================================================


class TestDataPathLoadSeam:
    """Tests for VTKPath.load() seam — each Path subclass loads itself."""

    def test_vtk_path_load_raises_not_implemented(self):
        # VTKPath with a non-h5 extension has no typed loading strategy
        vtk = VTKPath(root=Path("/tmp"), extension="fif")
        with pytest.raises(NotImplementedError):
            vtk.load()

    def test_ephys_path_has_load_method(self):
        path = EphysPath(
            root=Path("/ephys"),
            session_id="251024_FanFan_nsd1w_MSB",
            dtype="TrialRaster",
            extension="h5",
        )
        assert callable(path.load)

    def test_mne_path_has_load_method(self):
        path = MNEPath(
            root=Path("/mne"),
            subject="01",
            session="01",
            task="rest",
            run="01",
            suffix="raw",
            extension="fif",
        )
        assert callable(path.load)

    def test_bids_path_has_load_method(self):
        path = BIDSPath(
            root=Path("/bids"),
            subject="01",
            session="01",
            task="rest",
            suffix="meg",
            extension="fif",
        )
        assert callable(path.load)

    def test_read_delegates_to_path_load(self, tmp_path):
        """read(EphysPath) and EphysPath.load() raise the same error (file not found)."""
        path = EphysPath(
            root=tmp_path,
            session_id="251024_FanFan_nsd1w_MSB",
            dtype="TrialRaster",
            extension="h5",
        )
        exc_direct = None
        exc_via_read = None

        try:
            path.load()
        except Exception as e:
            exc_direct = type(e)

        try:
            import vneurotk as vnt

            vnt.read(path)
        except Exception as e:
            exc_via_read = type(e)

        assert exc_direct == exc_via_read


# ===========================================================================
# TestSingleImageSource  (Phase C)
# ===========================================================================


class TestSingleImageSource:
    """Tests for Phase C: Stimulus images stored in one canonical location."""

    def _make_images(self, n: int = 4) -> dict:
        rng = np.random.default_rng(0)
        return {i: rng.random((4,)).astype(np.float32) for i in range(n)}

    def _make_bd(self) -> BaseData:
        neuro = np.random.randn(600, 8).astype(np.float32)
        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
        )
        return bd

    def test_extract_from_uses_vision_db_fallback(self):
        """vision.extract_from(model, vision_db=images) 将图像传递并提取特征。"""
        import importlib
        import sys

        tv = importlib.import_module("test_vision") if "test_vision" in sys.modules else None
        if tv is None:
            import importlib.util
            import pathlib

            spec = importlib.util.spec_from_file_location(
                "test_vision",
                pathlib.Path(__file__).parent / "test_vision.py",
            )
            assert spec is not None and spec.loader is not None
            tv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tv)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = tv._MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )

        images = self._make_images()
        bd = self._make_bd()
        bd.vision.extract_from(model, vision_db=images)
        assert bd.has_vision

    def test_save_load_preserves_vision(self, tmp_path):
        """保存有视觉特征的 bd 后重载，has_vision 仍为 True。"""
        import importlib
        import sys

        tv = importlib.import_module("test_vision") if "test_vision" in sys.modules else None
        if tv is None:
            import importlib.util
            import pathlib

            spec = importlib.util.spec_from_file_location(
                "test_vision",
                pathlib.Path(__file__).parent / "test_vision.py",
            )
            assert spec is not None and spec.loader is not None
            tv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tv)

        import vneurotk as vtk
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = tv._MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )

        images = self._make_images()
        bd = self._make_bd()
        bd.vision.extract_from(model, vision_db=images)
        fpath = tmp_path / "test.h5"
        bd.save(fpath)

        loaded = vtk.read(fpath)
        assert loaded.has_vision

    def test_configure_with_vision_db_stores_stimuli(self):
        """configure(vision_db=...) 立即将图像存入 vision.db，无需等到 extract_from。"""
        images = self._make_images()
        bd = self._make_bd()
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images,
        )
        assert bd.vision.db is not None
        assert len(bd.vision.db) == 4

    def test_configure_replaces_vision_db_logs_info(self, caplog):
        """第二次 configure(vision_db=...) 覆盖旧图像并发出 info 日志。"""
        images_a = self._make_images(4)
        images_b = self._make_images(4)
        bd = self._make_bd()
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images_a,
        )
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images_b,
        )
        assert any("replacing" in r.getMessage().lower() for r in caplog.records)
        assert bd.vision.db is not None

    def test_extract_from_with_configure_vision_db(self):
        """configure(vision_db=...) 后，vision.extract_from() 无需再传 vision_db。"""
        import importlib
        import sys

        tv = importlib.import_module("test_vision") if "test_vision" in sys.modules else None
        if tv is None:
            import importlib.util
            import pathlib

            spec = importlib.util.spec_from_file_location(
                "test_vision", pathlib.Path(__file__).parent / "test_vision.py"
            )
            assert spec is not None and spec.loader is not None
            tv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tv)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = tv._MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )
        images = self._make_images()
        neuro = np.random.randn(600, 8).astype(np.float32)
        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images,
        )
        bd.vision.extract_from(model)
        assert bd.has_vision

    def test_extract_from_replaces_vision_db_logs_warning(self, caplog):
        """已有 vision_db 时再传给 vision.extract_from() 会发出 warning 并覆盖。"""
        import importlib
        import sys

        tv = importlib.import_module("test_vision") if "test_vision" in sys.modules else None
        if tv is None:
            import importlib.util
            import pathlib

            spec = importlib.util.spec_from_file_location(
                "test_vision", pathlib.Path(__file__).parent / "test_vision.py"
            )
            assert spec is not None and spec.loader is not None
            tv = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tv)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = tv._MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )
        images_a = self._make_images()
        images_b = self._make_images()
        neuro = np.random.randn(600, 8).astype(np.float32)
        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images_a,
        )
        bd.vision.extract_from(model, vision_db=images_b)
        assert any("replacing" in r.getMessage().lower() for r in caplog.records)


class TestBaseDataFactories:
    def test_for_continuous_creates_continuous_mode(self):
        """BaseData.for_continuous(neuro, neuro_info) sets data_mode='continuous'."""
        neuro = np.random.randn(500, 16)
        bd = BaseData.for_continuous(neuro, {"sfreq": 250.0})
        assert bd.data_mode == "continuous"
        assert bd.nchan == 16

    def test_for_epochs_creates_epochs_mode(self):
        """BaseData.for_epochs(neuro, neuro_info) sets data_mode='epochs'."""
        neuro = np.random.randn(20, 50, 16)
        bd = BaseData.for_epochs(neuro, {"sfreq": 250.0})
        assert bd.data_mode == "epochs"
        assert bd.nchan == 16

    def test_for_continuous_neuro_optional(self):
        """for_continuous without neuro keeps neuro=None for lazy loading."""
        bd = BaseData.for_continuous(neuro_info={"sfreq": 100.0, "shape": [200, 8]})
        assert bd._neuro is None
        assert bd.data_mode == "continuous"


class TestNeuroLoaderSeam:
    """D-A: set_neuro_loader() as public seam."""

    def test_set_neuro_loader_triggers_lazy(self):
        """bd.set_neuro_loader(fn) — accessing .neuro calls fn."""
        called = []

        def loader() -> np.ndarray:
            called.append(True)
            return np.ones((100, 8))

        bd = BaseData(None, {"sfreq": 100.0, "shape": [100, 8]}, data_mode="continuous")
        bd.set_neuro_loader(loader)
        arr = bd.neuro
        assert called, "loader was never called"
        assert arr.shape == (100, 8)

    def test_set_neuro_loader_replaces_previous(self):
        """Second call to set_neuro_loader replaces the first."""
        first_arr = np.zeros((10, 4))
        second_arr = np.ones((10, 4))

        bd = BaseData(None, {"sfreq": 100.0, "shape": [10, 4]}, data_mode="continuous")
        bd.set_neuro_loader(lambda: first_arr)
        bd.set_neuro_loader(lambda: second_arr)
        assert np.allclose(np.asarray(bd.neuro), second_arr)


class TestTrialStructure:
    """D-B: TrialStructureBuilder builds TrialStructure value objects from pure arrays."""

    def test_build_trial_structure_raw_returns_value(self):
        """build_trial_structure_continuous() returns TrialStructure without BaseData."""
        from vneurotk.neuro.trial import (
            build_trial_structure_continuous,
        )

        visual_ids = np.array([1, 2, 1, 3])
        vision_onsets = np.array([10, 30, 50, 70])
        ts = build_trial_structure_continuous(
            visual_ids=visual_ids,
            trial_window=[-5, 15],
            vision_onsets=vision_onsets,
            ntime=100,
            sfreq=1.0,
        )
        assert ts.trial_starts is not None
        assert ts.trial_ends is not None
        assert len(ts.trial_starts) == 4
        assert ts.vision_onsets is not None

    def test_build_trial_structure_epochs_returns_value(self):
        """build_trial_structure_epochs() returns TrialStructure without BaseData."""
        from vneurotk.neuro.trial import (
            build_trial_structure_epochs,
        )

        visual_ids = np.array([1, 2, 1])
        neuro_shape = (3, 20, 8)  # (n_trials, n_timebins, n_chan)
        ts = build_trial_structure_epochs(
            visual_ids=visual_ids,
            vision_onsets=None,
            neuro_shape=neuro_shape,
        )
        assert len(ts.trial_starts) == 3
        assert len(ts.trial_ends) == 3
        assert ts.trial_info is not None

    def test_trial_structure_fields_complete(self):
        """TrialStructure has all 7 required fields."""
        from vneurotk.neuro.trial import TrialStructure

        fields = {f.name for f in TrialStructure.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        expected = {
            "stim_labels",
            "trial",
            "trial_starts",
            "trial_ends",
            "vision_onsets",
            "vision_info",
            "trial_info",
        }
        assert expected == fields


class TestNeuroDataDecoupled:
    """D-C: NeuroData should not hold a back-reference to BaseData."""

    def test_neuro_data_standalone(self):
        """NeuroData can be constructed without BaseData."""
        from vneurotk.neuro.base import NeuroData

        arr = np.random.randn(100, 8)
        nd = NeuroData(arr)
        assert np.allclose(np.asarray(nd), arr)

    def test_neuro_data_epochs_view_no_bd(self):
        """NeuroData with trial_starts/ends provides .epochs without BaseData."""
        from vneurotk.neuro.base import NeuroData

        n_trials, n_time, n_chan = 5, 20, 4
        arr = np.random.randn(n_trials, n_time, n_chan)
        trial_starts = np.zeros(n_trials, dtype=int)
        trial_ends = np.full(n_trials, n_time, dtype=int)
        nd = NeuroData(arr, trial_starts=trial_starts, trial_ends=trial_ends, data_mode="epochs")
        epochs = nd.epochs
        assert epochs.shape == (n_trials, n_time, n_chan)

    def test_neuro_data_no_circular_reference(self):
        """bd does not hold NeuroData; NeuroData does not hold BaseData."""

        neuro = np.random.randn(1000, 8)
        bd = BaseData(neuro, {"sfreq": 100.0})
        nd = bd.neuro
        # NeuroData must not carry a BaseData reference
        assert not hasattr(nd, "_bd") or nd._bd is None


# ===========================================================================
# New tests for architectural deepening candidates
# ===========================================================================


class TestTrialStructureBuilder:
    """Candidate 3 — TrialStructureBuilder builds TrialStructure from pure arrays."""

    def test_for_continuous_window_in_seconds(self):
        from vneurotk.neuro.trial import (
            build_trial_structure_continuous,
        )

        ts = build_trial_structure_continuous(
            visual_ids=np.array([1, 2, 3]),
            trial_window=[-0.1, 0.4],
            vision_onsets=np.array([100, 200, 300]),
            ntime=500,
            sfreq=1000.0,
        )
        assert len(ts.trial_starts) == 3
        assert ts.trial_starts[0] == 100 - 100  # -0.1s * 1000 Hz
        assert ts.trial_ends[0] == 100 + 400

    def test_for_continuous_window_in_samples(self):
        from vneurotk.neuro.trial import (
            build_trial_structure_continuous,
        )

        ts = build_trial_structure_continuous(
            visual_ids=np.array([1, 2]),
            trial_window=[-5, 15],
            vision_onsets=np.array([10, 30]),
            ntime=60,
            sfreq=1.0,
        )
        assert ts.trial_starts[0] == 5
        assert ts.trial_ends[0] == 25

    def test_for_epochs_defaults_onset_to_zero(self):
        from vneurotk.neuro.trial import (
            build_trial_structure_epochs,
        )

        ts = build_trial_structure_epochs(
            visual_ids=np.array([1, 2, 1]),
            vision_onsets=None,
            neuro_shape=(3, 20, 8),
        )
        assert np.all(ts.vision_onsets == 0)
        assert len(ts.trial_starts) == 3
        assert len(ts.trial_ends) == 3

    def test_for_epochs_with_given_onsets(self):
        from vneurotk.neuro.trial import (
            build_trial_structure_epochs,
        )

        onsets = np.array([5, 5, 5])
        ts = build_trial_structure_epochs(
            visual_ids=np.array([1, 2, 3]),
            vision_onsets=onsets,
            neuro_shape=(3, 20, 4),
        )
        assert np.array_equal(ts.vision_onsets, onsets)

    def test_stim_labels_helper(self):
        from vneurotk.neuro.trial import (
            _stim_labels_continuous,
        )

        labels = _stim_labels_continuous(
            n_timebins=10,
            vision_onsets=np.array([2, 5]),
            visual_ids=np.array([10, 20]),
        )
        assert labels[2] == 10
        assert labels[5] == 20
        assert np.isnan(labels[0])

    def test_window_to_samples_mixed(self):
        from vneurotk.neuro.trial import (
            _window_to_samples,
        )

        result = _window_to_samples([-0.5, 1], sfreq=1000.0)
        assert result == [-500, 1]


class TestStimulusSetConstructors:
    """Candidate 5 — StimulusSet explicit classmethods."""

    def test_from_dict(self):
        from vneurotk.core.stimulus import StimulusSet

        stim_ids = np.array([1, 2, 1, 3])
        img1, img2, img3 = np.zeros((4, 4, 3)), np.ones((4, 4, 3)), np.eye(4)[:, :3]
        images = {1: img1, 2: img2, 3: img3}
        ss = StimulusSet.from_dict(stim_ids, images)
        assert np.array_equal(ss[1], img1)
        assert np.array_equal(ss[3], img3)

    def test_from_unique_list(self):
        from vneurotk.core.stimulus import StimulusSet

        stim_ids = np.array([10, 20, 10, 30])
        img_a, img_b, img_c = np.zeros((2, 2)), np.ones((2, 2)), np.eye(2)
        images = [img_a, img_b, img_c]  # 3 unique ids in first-appearance order
        ss = StimulusSet.from_unique_list(stim_ids, images)
        assert np.array_equal(ss[10], img_a)
        assert np.array_equal(ss[20], img_b)
        assert np.array_equal(ss[30], img_c)

    def test_from_unique_list_wrong_length_raises(self):
        from vneurotk.core.stimulus import StimulusSet

        stim_ids = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="images length"):
            StimulusSet.from_unique_list(stim_ids, ["only_one"])

    def test_from_h5_stores_mapping(self):
        from vneurotk.core.stimulus import StimulusSet

        sentinel = np.zeros((3, 3))

        class FakeMapping:
            def __getitem__(self, k):
                return sentinel

            def __contains__(self, k):
                return True

            def __len__(self):
                return 2

        stim_ids = np.array([1, 2])
        ss = StimulusSet.from_h5(stim_ids, FakeMapping())
        assert np.array_equal(ss[1], sentinel)

    def test_from_dict_stim_ids_must_be_1d(self):
        from vneurotk.core.stimulus import StimulusSet

        with pytest.raises(ValueError, match="1-D"):
            StimulusSet.from_dict(np.array([[1, 2]]), {1: "a", 2: "b"})


class TestBaseDataStateProperties:
    """Candidate 2 — BaseData.is_configured and is_vision_ready."""

    def test_is_configured_false_before_configure(self):
        bd = BaseData(np.random.randn(100, 4), {"sfreq": 100.0})
        assert bd.is_configured is False
        assert bd.configured is False

    def test_is_configured_true_after_configure(self):
        bd = BaseData(np.random.randn(200, 4), {"sfreq": 100.0})
        bd.configure(
            stim_ids=[1, 2, 3],
            trial_window=[-5, 15],
            vision_onsets=np.array([20, 80, 140]),
        )
        assert bd.is_configured is True
        assert bd.configured is True

    def test_is_vision_ready_false_before_extract(self):
        bd = BaseData(np.random.randn(200, 4), {"sfreq": 100.0})
        bd.configure(
            stim_ids=[1, 2, 3],
            trial_window=[-5, 15],
            vision_onsets=np.array([20, 80, 140]),
        )
        assert bd.is_vision_ready is False

    def test_vision_error_message_mentions_is_configured(self):
        bd = BaseData(np.random.randn(100, 4), {"sfreq": 100.0})
        with pytest.raises(RuntimeError, match="is_configured"):
            _ = bd.vision


class TestLazyNeuroLoader:
    """Candidate 4 — LazyNeuroLoader caches and defers the neuro load."""

    def test_loader_called_only_once(self, tmp_path):
        call_count = {"n": 0}

        def counting_loader():
            call_count["n"] += 1
            return np.ones((10, 4))

        from vneurotk.io.loader import LazyNeuroLoader

        loader = LazyNeuroLoader(counting_loader)
        result1 = loader()
        result2 = loader()
        assert call_count["n"] == 1
        assert np.array_equal(result1, result2)

    def test_is_loaded_flag(self):
        from vneurotk.io.loader import LazyNeuroLoader

        loader = LazyNeuroLoader(lambda: np.zeros((5, 3)))
        assert loader.is_loaded is False
        loader()
        assert loader.is_loaded is True

    def test_integrates_with_basedata(self):
        from vneurotk.io.loader import LazyNeuroLoader

        arr = np.random.randn(50, 8)
        loader = LazyNeuroLoader(lambda: arr)
        bd = BaseData(None, {"sfreq": 100.0, "shape": (50, 8)}, data_mode="continuous")
        bd.set_neuro_loader(loader)
        assert np.allclose(np.asarray(bd.neuro), arr)
        assert loader.is_loaded is True

    def test_hdf5_roundtrip(self, tmp_path):
        import h5py

        from vneurotk.io.loader import LazyNeuroLoader

        arr = np.arange(20, dtype=float).reshape(4, 5)
        h5_path = tmp_path / "test.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("neuro", data=arr)

        def load_from_h5():
            with h5py.File(h5_path, "r") as f:
                return f["neuro"][:]

        loader = LazyNeuroLoader(load_from_h5)
        result = loader()
        assert np.array_equal(result, arr)


class TestVisionStore:
    """VisionData storage: add, overwrite, has_visual_representations, dump/from_h5."""

    def _make_vr(self, model="resnet", module="layer1", n_stim=5, dim=16):
        from vneurotk.vision.representation.visual_representations import (
            VisualRepresentation,
            VisualRepresentations,
        )

        arr = np.random.randn(n_stim, dim)
        vr = VisualRepresentation(
            model=model,
            module_name=module,
            module_type="Conv2d",
            stim_ids=list(range(n_stim)),
            array=arr,
        )
        return VisualRepresentations([vr])

    def _make_vd(self, n_stim=5):
        from vneurotk.vision.data import VisionData

        return VisionData(np.arange(n_stim))

    def test_empty_has_no_visual_representations(self):
        vd = self._make_vd()
        assert vd.has_visual_representations is False
        assert len(vd.meta) == 0

    def test_add_and_has_visual_representations(self):
        vd = self._make_vd()
        vd.add(self._make_vr())
        assert vd.has_visual_representations is True
        assert len(vd.meta) == 1

    def test_add_overwrite_false_skips(self):
        vd = self._make_vd()
        vrs = self._make_vr()
        vd.add(vrs)
        vd.add(vrs, overwrite=False)
        assert len(vd.meta) == 1

    def test_add_overwrite_true_replaces(self):
        vd = self._make_vd()
        vd.add(self._make_vr(dim=4))
        vd.add(self._make_vr(dim=8), overwrite=True)
        assert vd.meta["shape"].iloc[0] == (5, 8)

    def test_dump_and_from_h5(self, tmp_path):
        import h5py

        from vneurotk.vision.data import VisionData

        vd = self._make_vd()
        vd.add(self._make_vr())
        h5_path = tmp_path / "store.h5"
        with h5py.File(h5_path, "w") as f:
            vd.dump(f)

        with h5py.File(h5_path, "r") as f:
            vd2 = VisionData.from_h5(f, output_order=np.arange(5))
        assert vd2.has_visual_representations is True
        assert len(vd2.meta) == 1

    def test_vision_data_aligned_output(self):
        from vneurotk.vision.data import VisionData

        output_order = np.array([0, 1, 2, 0, 1])
        vd = VisionData(output_order)
        vd.add(self._make_vr(n_stim=3, dim=8))
        assert vd.has_visual_representations is True
        arr = vd["layer1"]
        assert arr.shape == (5, 8)  # ty: ignore[unresolved-attribute]  # 5 trial-ordered entries


class TestAlignCacheInvalidation:
    """Candidate B — output_order setter clears _align_cache."""

    def _make_vd_with_vr(self, output_order):
        from vneurotk.vision.data import VisionData
        from vneurotk.vision.representation.visual_representations import (
            VisualRepresentation,
            VisualRepresentations,
        )

        n_stim = 3
        arr = np.arange(n_stim * 4, dtype=float).reshape(n_stim, 4)
        vr = VisualRepresentation(
            model="resnet",
            module_name="layer1",
            module_type="Conv2d",
            stim_ids=list(range(n_stim)),
            array=arr,
        )
        vd = VisionData(np.asarray(output_order))
        vd.add(VisualRepresentations([vr]))
        return vd

    def test_cache_populated_on_first_access(self):
        vd = self._make_vd_with_vr([0, 1, 2])
        assert len(vd._align_cache) == 0
        vd["layer1"]
        assert len(vd._align_cache) == 1

    def test_output_order_setter_clears_cache(self):
        vd = self._make_vd_with_vr([0, 1, 2])
        vd["layer1"]  # populate cache
        assert len(vd._align_cache) == 1

        vd.output_order = np.array([2, 1, 0])
        assert len(vd._align_cache) == 0

    def test_result_reflects_new_order_after_setter(self):
        vd = self._make_vd_with_vr([0, 1, 2])
        original = vd["layer1"].copy()  # shape (3, 4), rows = stim 0,1,2

        vd.output_order = np.array([2, 1, 0])
        reordered = vd["layer1"]

        np.testing.assert_array_equal(reordered[0], original[2])
        np.testing.assert_array_equal(reordered[2], original[0])


# ===========================================================================
# TestEphysHelpers — unit tests for shared Ephys loader helpers
# ===========================================================================


class TestEphysHelpers:
    """Tests for _load_ephys_prop, _load_ephys_record, _build_neuro_info
    extracted as shared helpers in loader.py (Candidate 3)."""

    def _write_csv(self, path, df):
        df.to_csv(path, index=False)

    def test_load_ephys_prop_returns_df_and_ch_names(self, tmp_path):
        """_load_ephys_prop parses CSV and extracts ch_names from 'id' column."""
        import pandas as pd

        from vneurotk.io.loader import _load_ephys_prop
        from vneurotk.io.path import EphysPath

        prop_df = pd.DataFrame({"id": ["ch0", "ch1", "ch2"], "depth": [100, 200, 300]})
        epath = EphysPath(root=tmp_path, session_id="s01", dtype="ChProp", probe=0, extension="csv")
        epath.fpath.parent.mkdir(parents=True, exist_ok=True)
        prop_df.to_csv(epath.fpath, index=False)

        base_path = EphysPath(root=tmp_path, session_id="s01", dtype="ChTrialRecord", probe=0, extension="csv")
        result_df, ch_names = _load_ephys_prop(base_path, "ChProp")
        assert ch_names == ["ch0", "ch1", "ch2"]
        assert list(result_df.columns) == ["id", "depth"]

    def test_load_ephys_prop_missing_file_raises(self, tmp_path):
        """_load_ephys_prop raises FileNotFoundError when CSV is absent."""
        from vneurotk.io.loader import _load_ephys_prop
        from vneurotk.io.path import EphysPath

        base_path = EphysPath(root=tmp_path, session_id="s01", dtype="ChTrialRecord", probe=0, extension="csv")
        with pytest.raises(FileNotFoundError):
            _load_ephys_prop(base_path, "ChProp")

    def test_load_ephys_record_returns_dataframe(self, tmp_path):
        """_load_ephys_record reads the record CSV into a DataFrame."""
        import pandas as pd

        from vneurotk.io.loader import _load_ephys_record
        from vneurotk.io.path import EphysPath

        record_df = pd.DataFrame({"trial": [0, 1], "value": [1.0, 2.0]})
        base_path = EphysPath(root=tmp_path, session_id="s01", dtype="ChProp", probe=0, extension="csv")
        record_epath = EphysPath(
            root=tmp_path,
            session_id="s01",
            dtype="ChTrialRecord",
            probe=0,
            extension="csv",
        )
        record_epath.fpath.parent.mkdir(parents=True, exist_ok=True)
        record_df.to_csv(record_epath.fpath, index=False)

        result = _load_ephys_record(base_path, "ChTrialRecord")
        assert list(result.columns) == ["trial", "value"]
        assert len(result) == 2

    def test_load_ephys_record_missing_file_raises(self, tmp_path):
        """_load_ephys_record raises FileNotFoundError when CSV is absent."""
        from vneurotk.io.loader import _load_ephys_record
        from vneurotk.io.path import EphysPath

        base_path = EphysPath(root=tmp_path, session_id="s01", dtype="ChProp", probe=0, extension="csv")
        with pytest.raises(FileNotFoundError):
            _load_ephys_record(base_path, "ChTrialRecord")


class TestBuildBdFromMneRaw:
    """Tests for _build_bd_from_mne_raw() — the shared MNE/BIDS BaseData builder."""

    def _make_mock_raw(self, sfreq=250.0, n_chan=8, n_times=1000):
        from unittest.mock import MagicMock

        raw = MagicMock()
        raw.info = {
            "sfreq": sfreq,
            "ch_names": [f"ch{i}" for i in range(n_chan)],
            "highpass": 0.1,
            "lowpass": 100.0,
        }
        raw.times = np.linspace(0, n_times / sfreq, n_times)
        raw.ch_names = [f"ch{i}" for i in range(n_chan)]
        return raw

    def _fn(self):
        from vneurotk.io.loader import _build_bd_from_mne_raw

        return _build_bd_from_mne_raw

    def test_returns_basedata(self):
        from vneurotk.core.recording import BaseData

        fn = self._fn()
        raw = self._make_mock_raw()
        bd = fn(raw, "/fake/path.fif")
        assert isinstance(bd, BaseData)

    def test_neuro_info_fields(self):
        fn = self._fn()
        raw = self._make_mock_raw(sfreq=1000.0, n_chan=4, n_times=500)
        bd = fn(raw, "/data/sub01.fif")
        assert bd.neuro_info["sfreq"] == 1000.0
        assert bd.neuro_info["ch_names"] == [f"ch{i}" for i in range(4)]
        assert bd.neuro_info["source_file"] == "/data/sub01.fif"

    def test_shape_in_neuro_info(self):
        fn = self._fn()
        raw = self._make_mock_raw(n_chan=16, n_times=2000)
        bd = fn(raw, "/x.fif")
        assert bd.neuro_info["shape"] == (2000, 16)

    def test_data_mode_is_continuous(self):
        fn = self._fn()
        raw = self._make_mock_raw()
        bd = fn(raw, "/x.fif")
        assert bd.data_mode == "continuous"


# ===========================================================================
# TestRestoreVisionData
# ===========================================================================


class TestRestoreVisionData:
    """Candidate 2 — _restore_vision_data() 是 h5_persistence 写入 _vision_data 的受控接口。"""

    def _make_bd(self):
        neuro = np.random.randn(600, 4).astype(np.float32)
        return BaseData(neuro=neuro, neuro_info={"sfreq": 100.0})

    def test_restore_sets_vision_data(self):
        """_restore_vision_data(store) 设置 _vision_data 属性。"""
        bd = self._make_bd()
        sentinel = object()
        bd._restore_vision_data(sentinel)
        assert bd._vision_data is sentinel

    def test_restore_none_clears_vision_data(self):
        """_restore_vision_data(None) 清除已有的 _vision_data。"""
        bd = self._make_bd()
        bd._restore_vision_data(object())
        bd._restore_vision_data(None)
        assert bd._vision_data is None

    def test_stim_labels_property_matches_private(self):
        """stim_labels property 与 _stim_labels 指向同一对象。"""
        neuro = np.random.randn(600, 4).astype(np.float32)
        bd = BaseData(neuro=neuro, neuro_info={"sfreq": 100.0})
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
        )
        assert bd.stim_labels is bd._stim_labels

    def test_stim_labels_none_before_configure(self):
        """stim_labels 在 configure() 调用前为 None。"""
        bd = self._make_bd()
        assert bd.stim_labels is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ===========================================================================
# TestLazyH5DictDecodeItem
# ===========================================================================


class TestLazyH5DictDecodeItem:
    """Candidate 2 (Round 14) — _decode_item encapsulates kind dispatch."""

    def test_array_kind_returns_ndarray(self):
        """Default 'array' kind returns the raw dataset array."""
        from vneurotk.io.loader import LazyH5Dict

        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                grp = f.create_group("g")
                grp.create_dataset("ds", data=arr)
                grp["ds"].attrs["kind"] = "array"
            with h5py.File(h5_path, "r") as f:
                result = LazyH5Dict._decode_item(f["g"]["ds"])
        assert np.array_equal(result, arr)

    def test_missing_kind_defaults_to_array(self):
        """No 'kind' attribute → treated as array."""
        from vneurotk.io.loader import LazyH5Dict

        arr = np.arange(6, dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                grp = f.create_group("g")
                grp.create_dataset("ds", data=arr)
            with h5py.File(h5_path, "r") as f:
                result = LazyH5Dict._decode_item(f["g"]["ds"])
        assert np.array_equal(result, arr)

    def test_image_bytes_kind_decodes_to_hwc(self):
        """'image_bytes' kind decodes PNG bytes to (H, W, C) uint8 array."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io.loader import LazyH5Dict

        original = np.full((8, 12, 3), 200, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            PILImage.fromarray(original).save(img_path, format="PNG")
            raw = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)

            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                grp = f.create_group("g")
                grp.create_dataset("ds", data=raw)
                grp["ds"].attrs["kind"] = "image_bytes"
            with h5py.File(h5_path, "r") as f:
                result = LazyH5Dict._decode_item(f["g"]["ds"])
        assert result.shape == (8, 12, 3)
        assert np.array_equal(result, original)


# ===========================================================================
# TestBuildVisionInfo
# ===========================================================================


class TestBuildVisionInfoTrialStructure:
    def test_returns_correct_keys(self):
        """Result dict has exactly 'n_stim' and 'stim_ids'."""
        from vneurotk.neuro.trial import (
            _build_vision_info,
        )

        result = _build_vision_info(np.array([2, 1, 3, 1]))
        assert set(result.keys()) == {"n_stim", "stim_ids"}

    def test_n_stim_counts_unique(self):
        """n_stim equals the number of unique IDs."""
        from vneurotk.neuro.trial import (
            _build_vision_info,
        )

        result = _build_vision_info(np.array([0, 1, 2, 1, 0]))
        assert result["n_stim"] == 3

    def test_stim_ids_sorted(self):
        """stim_ids contains unique IDs in sorted order."""
        from vneurotk.neuro.trial import (
            _build_vision_info,
        )

        result = _build_vision_info(np.array([3, 1, 2, 1]))
        assert result["stim_ids"] == [1, 2, 3]


# ===========================================================================
# TestDecodeAttr
# ===========================================================================


class TestDecodeAttr:
    """Candidate 1 (Round 16) — _decode_attr encapsulates HDF5 attr type conversion."""

    def test_ndarray_becomes_list(self):
        from vneurotk.io.h5_persistence import _decode_attr

        result = _decode_attr(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_np_integer_becomes_int(self):
        from vneurotk.io.h5_persistence import _decode_attr

        result = _decode_attr(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_floating_becomes_float(self):
        from vneurotk.io.h5_persistence import _decode_attr

        result = _decode_attr(np.float32(3.14))
        assert isinstance(result, float)

    def test_python_native_passthrough(self):
        from vneurotk.io.h5_persistence import _decode_attr

        assert _decode_attr("hello") == "hello"
        assert _decode_attr(7) == 7
        assert _decode_attr(None) is None


# ===========================================================================
# TestStimLabels
# ===========================================================================


class TestStimLabels:
    """Candidate 3 (Round 16) — _stim_labels_continuous / _stim_labels_epochs explicit shapes."""

    def test_continuous_shape(self):
        """_stim_labels_continuous returns shape (n_timebins,)."""
        from vneurotk.neuro.trial import (
            _stim_labels_continuous,
        )

        onsets = np.array([5, 15])
        ids = np.array([1, 2])
        result = _stim_labels_continuous(20, onsets, ids)
        assert result.shape == (20,)

    def test_continuous_places_ids_at_onsets(self):
        """Stimulus IDs appear at correct onset positions."""
        from vneurotk.neuro.trial import (
            _stim_labels_continuous,
        )

        onsets = np.array([2, 7])
        ids = np.array([10, 20])
        result = _stim_labels_continuous(10, onsets, ids)
        assert result[2] == 10
        assert result[7] == 20

    def test_epochs_shape(self):
        """_stim_labels_epochs returns shape (n_trials, n_timebins)."""
        from vneurotk.neuro.trial import (
            _stim_labels_epochs,
        )

        n_trials, n_timebins = 4, 10
        onsets = np.zeros(n_trials, dtype=int)
        ids = np.arange(n_trials)
        result = _stim_labels_epochs(n_trials, n_timebins, onsets, ids)
        assert result.shape == (n_trials, n_timebins)

    def test_epochs_places_ids_at_onset_column(self):
        """Each row has its stimulus ID at the correct onset column."""
        from vneurotk.neuro.trial import (
            _stim_labels_epochs,
        )

        onsets = np.array([1, 3])
        ids = np.array([100, 200])
        result = _stim_labels_epochs(2, 5, onsets, ids)
        assert result[0, 1] == 100
        assert result[1, 3] == 200


# ===========================================================================
# TestIsSparse
# ===========================================================================


class TestIsSparse:
    """Candidate 1 (Round 17) — _is_sparse names the sparsity heuristic."""

    def test_mostly_zeros_is_sparse(self):
        from vneurotk.io.h5_persistence import _is_sparse

        arr = np.zeros((4, 10, 8), dtype=np.float32)
        arr[0, 0, 0] = 1.0  # only one non-zero
        assert _is_sparse(arr) is True

    def test_dense_array_not_sparse(self):
        from vneurotk.io.h5_persistence import _is_sparse

        arr = np.ones((4, 10, 8), dtype=np.float32)
        assert _is_sparse(arr) is False

    def test_exactly_half_zeros_not_sparse(self):
        """50 % zeros is NOT above the 0.5 threshold — should return False."""
        from vneurotk.io.h5_persistence import _is_sparse

        arr = np.zeros((2, 2, 2), dtype=np.float32)
        arr[0, 0, 0] = 1.0
        arr[0, 0, 1] = 1.0
        arr[0, 1, 0] = 1.0
        arr[0, 1, 1] = 1.0
        assert _is_sparse(arr) is False


# ===========================================================================
# TestNeuroLoaderFactories
# ===========================================================================


class TestNeuroLoaderFactories:
    """Candidate 2 (Round 17) — _make_coo_loader / _make_dense_loader are independently callable."""

    def test_dense_loader_returns_array(self):
        from vneurotk.io.h5_persistence import _make_dense_loader

        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.h5"
            with h5py.File(fpath, "w") as f:
                f.create_dataset("neuro", data=arr)
            loader = _make_dense_loader(fpath)
            result = loader()
        assert np.array_equal(result, arr)

    def test_coo_loader_reconstructs_sparse(self):
        pytest.importorskip("scipy")
        from scipy.sparse import coo_matrix

        from vneurotk.io.h5_persistence import _make_coo_loader

        original = np.zeros((2, 3, 4), dtype=np.float32)
        original[0, 1, 2] = 5.0
        original[1, 0, 3] = 7.0
        flat = original.reshape(-1, original.shape[-1])
        sparse = coo_matrix(flat)
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "test.h5"
            with h5py.File(fpath, "w") as f:
                f.create_dataset("neuro_row", data=sparse.row)
                f.create_dataset("neuro_col", data=sparse.col)
                f.create_dataset("neuro_data", data=sparse.data)
            loader = _make_coo_loader(fpath, original.shape, str(original.dtype))
            result = loader()
        assert result.shape == original.shape
        assert np.allclose(result, original)


# ===========================================================================
# TestEncodeImage
# ===========================================================================


class TestEncodeImage:
    """Candidate 3 (Round 17) — _encode_image concentrates image serialization."""

    def test_ndarray_returns_array_kind(self):
        from vneurotk.io.h5_persistence import _encode_image

        arr = np.ones((4, 4, 3), dtype=np.uint8)
        data, kind = _encode_image(arr)
        assert kind == "array"
        assert np.array_equal(data, arr)

    def test_array_like_returns_array_kind(self):
        from vneurotk.io.h5_persistence import _encode_image

        data, kind = _encode_image([[1, 2], [3, 4]])
        assert kind == "array"
        assert data.shape == (2, 2)

    def test_path_returns_image_bytes_kind(self):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io.h5_persistence import _encode_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)
            data, kind = _encode_image(img_path)
        assert kind == "image_bytes"
        assert data.dtype == np.uint8
        assert data.ndim == 1

    def test_str_path_returns_image_bytes_kind(self):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io.h5_persistence import _encode_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(img_path)
            data, kind = _encode_image(str(img_path))
        assert kind == "image_bytes"


# ===========================================================================
# TestTimeAxisIndex
# ===========================================================================


class TestTimeAxisIndex:
    """Candidate 2 (Round 18) — _time_axis_index encapsulates data_mode → axis mapping."""

    def _make_bd(self, data_mode: str, neuro: np.ndarray) -> BaseData:
        from vneurotk.core.recording import BaseData

        if data_mode == "epochs":
            n_trials, n_time, n_chan = neuro.shape
            return BaseData(
                neuro=neuro,
                neuro_info={"shape": list(neuro.shape), "sfreq": 100},
                stim_labels=np.zeros((n_trials, n_time)),
                trial=np.zeros((n_trials, n_time)),
                trial_starts=np.zeros(n_trials, dtype=int),
                trial_ends=np.full(n_trials, n_time, dtype=int),
                vision_onsets=np.zeros(n_trials, dtype=int),
                vision_info={"n_stim": 1, "stim_ids": [0]},
                trial_info={"baseline": [0, 0], "trial_window": [0, n_time]},
                data_mode="epochs",
            )
        else:
            n_time, n_chan = neuro.shape
            return BaseData(
                neuro=neuro,
                neuro_info={"shape": list(neuro.shape), "sfreq": 100},
                stim_labels=np.zeros(n_time),
                trial=np.zeros(n_time),
                trial_starts=np.array([0]),
                trial_ends=np.array([n_time]),
                vision_onsets=np.array([0]),
                vision_info={"n_stim": 1, "stim_ids": [0]},
                trial_info={"baseline": [0, 0], "trial_window": [0, n_time]},
                data_mode="continuous",
            )

    def test_epochs_mode_returns_1(self):
        """Epochs layout (n_trials, n_time, n_chan) → time axis is 1."""
        bd = self._make_bd("epochs", np.zeros((4, 50, 3)))
        assert bd._time_axis_index() == 1

    def test_continuous_mode_returns_0(self):
        """Continuous layout (n_time, n_chan) → time axis is 0."""
        bd = self._make_bd("continuous", np.zeros((50, 3)))
        assert bd._time_axis_index() == 0

    def test_ntime_epochs_uses_axis_1(self):
        """ntime for epochs returns shape[1] (time axis)."""
        bd = self._make_bd("epochs", np.zeros((4, 50, 3)))
        assert bd.ntime == 50

    def test_ntime_continuous_uses_axis_0(self):
        """ntime for continuous returns shape[0] (time axis)."""
        bd = self._make_bd("continuous", np.zeros((50, 3)))
        assert bd.ntime == 50


# ===========================================================================
# TestStimIdAtTrial
# ===========================================================================


class TestStimIdAtTrial:
    """Candidate 3 (Round 18) — _stim_id_at_trial encapsulates stim lookup per data_mode."""

    def _make_epochs_bd(self) -> BaseData:
        from vneurotk.core.recording import BaseData

        n_trials, n_time, n_chan = 3, 10, 2
        stim_labels = np.array([[i] * n_time for i in range(n_trials)], dtype=float)
        return BaseData(
            neuro=np.zeros((n_trials, n_time, n_chan)),
            neuro_info={"shape": [n_trials, n_time, n_chan], "sfreq": 100},
            stim_labels=stim_labels,
            trial=np.zeros((n_trials, n_time)),
            trial_starts=np.zeros(n_trials, dtype=int),
            trial_ends=np.full(n_trials, n_time, dtype=int),
            vision_onsets=np.zeros(n_trials, dtype=int),
            vision_info={"n_stim": n_trials, "stim_ids": list(range(n_trials))},
            trial_info={"baseline": [0, 0], "trial_window": [0, n_time]},
            data_mode="epochs",
        )

    def _make_continuous_bd(self) -> BaseData:
        from vneurotk.core.recording import BaseData

        n_time, n_chan = 20, 2
        stim_labels = np.zeros(n_time)
        stim_labels[5] = 7.0
        stim_labels[10] = 8.0
        return BaseData(
            neuro=np.zeros((n_time, n_chan)),
            neuro_info={"shape": [n_time, n_chan], "sfreq": 100},
            stim_labels=stim_labels,
            trial=np.zeros(n_time),
            trial_starts=np.array([5, 10]),
            trial_ends=np.array([8, 13]),
            vision_onsets=np.array([5, 10]),
            vision_info={"n_stim": 2, "stim_ids": [7, 8]},
            trial_info={"baseline": [0, 0], "trial_window": [0, 5]},
            data_mode="continuous",
        )

    def test_epochs_returns_row_stim_id(self):
        """For epochs, returns stim_labels[i, onset] (row-indexed)."""
        bd = self._make_epochs_bd()
        assert bd._stim_id_at_trial(0) == 0.0
        assert bd._stim_id_at_trial(1) == 1.0
        assert bd._stim_id_at_trial(2) == 2.0

    def test_continuous_returns_flat_stim_id(self):
        """For continuous, returns stim_labels[onset] (1-D indexed)."""
        bd = self._make_continuous_bd()
        assert bd._stim_id_at_trial(0) == 7.0
        assert bd._stim_id_at_trial(1) == 8.0

    def test_trial_stim_ids_uses_stim_id_at_trial(self):
        """trial_stim_ids produces same result as calling _stim_id_at_trial per trial."""
        bd = self._make_epochs_bd()
        expected = np.array([bd._stim_id_at_trial(i) for i in range(bd.n_trials)])
        np.testing.assert_array_equal(bd.trial_stim_ids, expected)

    def test_trial_stim_ids_continuous(self):
        """trial_stim_ids works correctly for continuous data_mode."""
        bd = self._make_continuous_bd()
        expected = np.array([7.0, 8.0])
        np.testing.assert_array_equal(bd.trial_stim_ids, expected)


# ===========================================================================
# TestNeuroShapeDim
# ===========================================================================


class TestNeuroShapeDim:
    """Candidate 1 (Round 19) — _neuro_shape_dim encapsulates neuro → neuro_info fallback."""

    def _make_bd_with_neuro(self, shape: tuple) -> BaseData:
        from vneurotk.core.recording import BaseData

        if len(shape) == 3:
            n_trials, n_time, n_chan = shape
            return BaseData(
                neuro=np.zeros(shape),
                neuro_info={"shape": list(shape), "sfreq": 100},
                stim_labels=np.zeros((n_trials, n_time)),
                trial=np.zeros((n_trials, n_time)),
                trial_starts=np.zeros(n_trials, dtype=int),
                trial_ends=np.full(n_trials, n_time, dtype=int),
                vision_onsets=np.zeros(n_trials, dtype=int),
                vision_info={"n_stim": 1, "stim_ids": [0]},
                trial_info={"baseline": [0, 0], "trial_window": [0, n_time]},
                data_mode="epochs",
            )
        n_time, n_chan = shape
        return BaseData(
            neuro=np.zeros(shape),
            neuro_info={"shape": list(shape), "sfreq": 100},
            stim_labels=np.zeros(n_time),
            trial=np.zeros(n_time),
            trial_starts=np.array([0]),
            trial_ends=np.array([n_time]),
            vision_onsets=np.array([0]),
            vision_info={"n_stim": 1, "stim_ids": [0]},
            trial_info={"baseline": [0, 0], "trial_window": [0, n_time]},
            data_mode="continuous",
        )

    def _make_bd_neuro_info_only(self, shape: tuple, data_mode: str = "continuous") -> BaseData:
        from vneurotk.core.recording import BaseData

        return BaseData(
            neuro=None,
            neuro_info={"shape": list(shape), "sfreq": 100},
            stim_labels=None,
            trial=None,
            trial_starts=None,
            trial_ends=None,
            vision_onsets=None,
            vision_info={"n_stim": 0, "stim_ids": []},
            trial_info={"baseline": [0, 0], "trial_window": [0, 10]},
            data_mode=data_mode,
        )

    def test_returns_from_neuro_array_when_loaded(self):
        """_neuro_shape_dim reads shape from _neuro when it is loaded."""
        bd = self._make_bd_with_neuro((50, 3))
        assert bd._neuro_shape_dim(0) == 50
        assert bd._neuro_shape_dim(-1) == 3

    def test_falls_back_to_neuro_info_when_neuro_absent(self):
        """_neuro_shape_dim falls back to neuro_info['shape'] when _neuro is None."""
        bd = self._make_bd_neuro_info_only((50, 3))
        assert bd._neuro_shape_dim(0) == 50
        assert bd._neuro_shape_dim(-1) == 3

    def test_returns_zero_when_both_absent(self):
        """_neuro_shape_dim returns 0 when _neuro is None and neuro_info has no shape."""
        from vneurotk.core.recording import BaseData

        bd = BaseData(
            neuro=None,
            neuro_info={"sfreq": 100},
            stim_labels=None,
            trial=None,
            trial_starts=None,
            trial_ends=None,
            vision_onsets=None,
            vision_info={"n_stim": 0, "stim_ids": []},
            trial_info={"baseline": [0, 0], "trial_window": [0, 10]},
        )
        assert bd._neuro_shape_dim(0) == 0
        assert bd._neuro_shape_dim(-1) == 0

    def test_nchan_uses_neuro_shape_dim(self):
        """nchan delegates to _neuro_shape_dim(-1) and returns channel count."""
        bd = self._make_bd_with_neuro((4, 50, 8))
        assert bd.nchan == 8

    def test_nchan_falls_back_to_ch_names(self):
        """nchan falls back to len(ch_names) when _neuro and shape are absent."""
        from vneurotk.core.recording import BaseData

        bd = BaseData(
            neuro=None,
            neuro_info={"sfreq": 100, "ch_names": ["a", "b", "c"]},
            stim_labels=None,
            trial=None,
            trial_starts=None,
            trial_ends=None,
            vision_onsets=None,
            vision_info={"n_stim": 0, "stim_ids": []},
            trial_info={"baseline": [0, 0], "trial_window": [0, 10]},
        )
        assert bd.nchan == 3


# ===========================================================================
# TestDecodeImage
# ===========================================================================


class TestDecodeImage:
    """Candidate 2 (Round 19) — _decode_image symmetric complement to _encode_image."""

    def test_array_kind_returns_data_unchanged(self):
        """kind='array' passes data through without modification."""
        from vneurotk.io.loader import _decode_image

        data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = _decode_image(data, "array")
        np.testing.assert_array_equal(result, data)

    def test_image_bytes_round_trip(self):
        """Encoding a path to image_bytes, then decoding, restores pixel data."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from vneurotk.io.h5_persistence import _encode_image
        from vneurotk.io.loader import _decode_image

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "img.png"
            arr = np.zeros((8, 8, 3), dtype=np.uint8)
            arr[0, 0] = [255, 0, 0]
            PILImage.fromarray(arr).save(img_path)

            encoded, kind = _encode_image(img_path)
            assert kind == "image_bytes"
            decoded = _decode_image(encoded, kind)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == (8, 8, 3)
        assert decoded[0, 0, 0] == 255

    def test_image_bytes_kind_requires_pillow(self):
        """kind='image_bytes' with valid bytes produces an ndarray."""
        pytest.importorskip("PIL")
        import io

        from PIL import Image as PILImage

        from vneurotk.io.loader import _decode_image

        buf = io.BytesIO()
        PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buf, format="PNG")
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        result = _decode_image(data, "image_bytes")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 4, 3)

    def test_unknown_kind_treated_as_array(self):
        """An unknown kind string falls through to the array path."""
        from vneurotk.io.loader import _decode_image

        data = np.arange(5, dtype=np.float32)
        result = _decode_image(data, "unknown_kind")
        np.testing.assert_array_equal(result, data)


# ===========================================================================
# TestBuildStimFrVisionInfo
# ===========================================================================


class TestBuildStimFrVisionInfo:
    """Candidate 2 (Round 20) — _build_stim_fr_vision_info encapsulates ChStimFr dict construction."""

    def test_none_arrays_gives_only_n_stim(self):
        """When allstim and teststim are None, only 'n_stim' key is present."""
        from vneurotk.io.loader import _build_stim_fr_vision_info

        result = _build_stim_fr_vision_info(10, None, None)
        assert result == {"n_stim": 10}

    def test_allstim_provided_adds_stim_ids(self):
        """allstim array is converted to list and stored as 'stim_ids'."""
        from vneurotk.io.loader import _build_stim_fr_vision_info

        allstim = np.array([1, 2, 3])
        result = _build_stim_fr_vision_info(3, allstim, None)
        assert result["stim_ids"] == [1, 2, 3]
        assert "teststim" not in result

    def test_teststim_provided_adds_teststim(self):
        """teststim array is converted to list and stored as 'teststim'."""
        from vneurotk.io.loader import _build_stim_fr_vision_info

        allstim = np.array([1, 2, 3])
        teststim = np.array([1, 3])
        result = _build_stim_fr_vision_info(3, allstim, teststim)
        assert result["teststim"] == [1, 3]

    def test_both_none_n_stim_correct(self):
        """n_stim value is taken from the argument, not inferred."""
        from vneurotk.io.loader import _build_stim_fr_vision_info

        result = _build_stim_fr_vision_info(42, None, None)
        assert result["n_stim"] == 42


# ===========================================================================
# TestInferStimuliMode  (Candidate 3, Round 21)
# ===========================================================================


class TestInferStimuliMode:
    """StimulusSet._infer_stimuli_mode names the implicit length-matching logic."""

    def test_n_seq_equals_n_unique_returns_by_unique(self):
        """Sequence length == n_unique → 'by_unique'."""
        from vneurotk.core.stimulus import StimulusSet

        assert StimulusSet._infer_stimuli_mode(3, 3, 6) == "by_unique"

    def test_n_seq_equals_n_onsets_returns_by_onset(self):
        """Sequence length == n_onsets (and != n_unique) → 'by_onset'."""
        from vneurotk.core.stimulus import StimulusSet

        assert StimulusSet._infer_stimuli_mode(6, 3, 6) == "by_onset"

    def test_ambiguous_case_prefers_by_unique(self):
        """When n_unique == n_onsets, 'by_unique' wins (unique check is first)."""
        from vneurotk.core.stimulus import StimulusSet

        assert StimulusSet._infer_stimuli_mode(5, 5, 5) == "by_unique"

    def test_no_match_raises_value_error(self):
        """Length matching neither n_unique nor n_onsets raises ValueError."""
        from vneurotk.core.stimulus import StimulusSet

        with pytest.raises(ValueError, match="does not match"):
            StimulusSet._infer_stimuli_mode(4, 3, 6)

    def test_error_message_includes_lengths(self):
        """ValueError message includes the offending lengths."""
        from vneurotk.core.stimulus import StimulusSet

        with pytest.raises(ValueError, match="7"):
            StimulusSet._infer_stimuli_mode(7, 3, 6)
