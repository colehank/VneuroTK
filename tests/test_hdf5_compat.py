"""HDF5 recording schema compatibility tests."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import vneurotk as vnt
from vneurotk.core import BaseData
from vneurotk.io.h5_persistence import (
    CURRENT_SCHEMA_VERSION,
    FORMAT_MAGIC,
    FORMAT_MAGIC_ATTR,
    MIN_SUPPORTED_SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    WRITER_VERSION_ATTR,
)
from vneurotk.vision.representation.visual_representations import VisualRepresentation, VisualRepresentations

pytestmark = pytest.mark.hdf5_compat

FIXTURES = Path(__file__).parent / "fixtures" / "hdf5"


@pytest.fixture(autouse=True, scope="session")
def verify_fixture_checksums():
    """Fail clearly if a checked-in historical binary changed."""
    for line in (FIXTURES / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = FIXTURES / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_v0_dense_numeric_lazy_and_eager():
    path = FIXTURES / "v0_dense_numeric.h5"
    lazy = vnt.read(path, pre_load=False)
    assert lazy.data_mode == "continuous"
    assert lazy._neuro is None
    assert lazy.neuro.shape == (20, 3)
    np.testing.assert_allclose(lazy.neuro, np.arange(60, dtype=np.float32).reshape(20, 3) / 10)
    assert lazy.trial_stim_ids.tolist() == [10.0, 20.0]
    assert lazy.trial_meta.to_dict("list") == {"condition": ["train", "test"], "stim_index": [10, 20]}
    np.testing.assert_array_equal(lazy.vision.db[10], np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

    eager = vnt.read(path, pre_load=True)
    assert eager._neuro is not None
    assert eager._neuro_loader is None


def test_v0_dense_string_and_historical_path_image():
    loaded = vnt.read(FIXTURES / "v0_dense_string_path.h5")
    assert loaded.trial_stim_ids.tolist() == ["face", "scene"]
    assert loaded.vision_info == {"n_stim": 2, "stim_ids": [b"face", b"scene"]}
    image = loaded.vision.db["face"]
    assert image.shape == (3, 4, 3)
    assert image[0, 0].tolist() == [120, 30, 0]


def test_v0_coo_vision_and_trial_metadata_remain_lazy():
    loaded = vnt.read(FIXTURES / "v0_coo_epochs_vision.h5", pre_load=False)
    assert loaded.data_mode == "epochs"
    assert loaded._neuro is None
    assert loaded.has_vision
    assert loaded.vision.meta.loc[0, "module_name"] == "features"
    record = next(iter(loaded.vision._records.values()))
    assert record.provenance.model_id == "historical-model"
    assert record.provenance.backend == "unknown"
    assert record.provenance.pretrained == "unknown"
    assert record._array is None
    np.testing.assert_array_equal(loaded.vision["features"], [[3.0, 30.0], [7.0, 70.0]])
    assert record._array is not None
    assert loaded.trial_meta.to_dict("list") == {"condition": ["train", "test"], "stim_index": [3, 7]}

    expected = np.zeros((2, 6, 4), dtype=np.float32)
    expected[0, 1, 2] = 1.5
    expected[1, 3, 0] = -2.0
    expected[1, 5, 3] = 4.0
    np.testing.assert_array_equal(loaded.neuro, expected)


def _v1_recording() -> BaseData:
    neuro = np.arange(160, dtype=np.float32).reshape(40, 4)
    recording = BaseData(neuro, {"sfreq": 100.0, "ch_names": ["a", "b", "c", "d"]})
    images = {1: np.full((2, 3, 3), 1, dtype=np.uint8), 2: np.full((2, 3, 3), 2, dtype=np.uint8)}
    recording.configure(
        stim_ids=np.array([1, 2]),
        trial_window=[-2, 4],
        vision_onsets=np.array([8, 25]),
        vision_db=images,
    )
    recording.trial_meta = pd.DataFrame({"stim_index": [1, 2], "condition": ["train", "test"]})
    return recording


def _recording_with_ids(ids, images=None) -> BaseData:
    recording = BaseData(np.arange(240, dtype=np.float32).reshape(80, 3), {"sfreq": 100.0})
    values = np.empty(len(ids), dtype=object)
    values[:] = ids
    recording.configure(
        stim_ids=values,
        trial_window=[-2, 3],
        vision_onsets=np.asarray([10 + 10 * i for i in range(len(ids))]),
        vision_db=images,
    )
    return recording


def test_schema1_requires_nonempty_data_mode_and_v0_missing_mode_defaults(tmp_path):
    v1_path = tmp_path / "schema1.h5"
    _v1_recording().save(v1_path)
    for replacement, message in ((None, "missing required root attribute 'data_mode'"), ("", "must be nonempty")):
        corrupt = tmp_path / f"corrupt-{replacement!r}.h5"
        shutil.copyfile(v1_path, corrupt)
        with h5py.File(corrupt, "r+") as f:
            if replacement is None:
                del f.attrs["data_mode"]
            else:
                f.attrs["data_mode"] = replacement
        with pytest.raises(ValueError, match=message):
            vnt.read(corrupt)

    legacy = tmp_path / "legacy-without-mode.h5"
    shutil.copyfile(FIXTURES / "v0_dense_numeric.h5", legacy)
    with h5py.File(legacy, "r+") as f:
        if "data_mode" in f.attrs:
            del f.attrs["data_mode"]
    assert vnt.read(legacy).data_mode == "continuous"


def test_schema1_missing_mode_never_reinterprets_epochs_as_continuous(tmp_path):
    path = tmp_path / "epochs.h5"
    shutil.copyfile(FIXTURES / "v0_coo_epochs_vision.h5", path)
    with h5py.File(path, "r+") as f:
        f.attrs[FORMAT_MAGIC_ATTR] = FORMAT_MAGIC
        f.attrs[SCHEMA_VERSION_ATTR] = CURRENT_SCHEMA_VERSION
        f.attrs[WRITER_VERSION_ATTR] = "test"
        del f.attrs["data_mode"]
    with pytest.raises(ValueError, match="missing required root attribute 'data_mode'"):
        vnt.read(path)


@pytest.mark.parametrize("stim_id", [True, 17, 1.25, "folder/name", "你好/scene"])
def test_schema1_scalar_stimulus_ids_roundtrip(tmp_path, stim_id):
    path = tmp_path / "id.h5"
    image = np.full((2, 2, 3), 7, dtype=np.uint8)
    _recording_with_ids([stim_id], {stim_id: image}).save(path)
    loaded = vnt.read(path)
    result = loaded.trial_stim_ids[0]
    native = result.item() if isinstance(result, np.generic) else result
    assert type(native) is type(stim_id)
    assert native == stim_id
    np.testing.assert_array_equal(loaded.vision.db[stim_id], image)
    with h5py.File(path, "r") as f:
        assert list(f["stimuli_db"]) == ["0"]


def test_integer_and_string_ids_remain_distinct(tmp_path):
    path = tmp_path / "distinct.h5"
    images = {
        1: np.full((2, 2, 3), 1, dtype=np.uint8),
        "1": np.full((2, 2, 3), 2, dtype=np.uint8),
    }
    _recording_with_ids([1, "1"], images).save(path)
    loaded = vnt.read(path)
    assert loaded.trial_stim_ids.tolist() == [1, "1"]
    np.testing.assert_array_equal(loaded.vision.db[1], images[1])
    np.testing.assert_array_equal(loaded.vision.db["1"], images["1"])


@pytest.mark.parametrize("stim_ids", [[True, 2], [1, "1"]])
def test_configured_heterogeneous_ids_roundtrip_with_matching_metadata_types(tmp_path, stim_ids):
    path = tmp_path / "heterogeneous.h5"
    recording = BaseData(np.arange(240, dtype=np.float32).reshape(80, 3), {"sfreq": 100.0})
    recording.configure(
        stim_ids=stim_ids,
        trial_window=[-2, 3],
        vision_onsets=np.array([10, 30]),
    )

    recording.save(path)
    loaded = vnt.read(path)

    assert loaded.vision_info is not None
    assert loaded.trial_stim_ids.tolist() == stim_ids
    assert [type(value) for value in loaded.trial_stim_ids.tolist()] == [type(value) for value in stim_ids]
    assert [type(value) for value in loaded.vision_info["stim_ids"]] == [type(value) for value in stim_ids]


@pytest.mark.parametrize("bad_id", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_float_stimulus_ids_fail_save_and_typed_load(tmp_path, bad_id):
    path = tmp_path / "bad-float.h5"
    with pytest.raises(ValueError, match="Non-finite float (stimulus ID|object stimulus label)"):
        _recording_with_ids([bad_id], {bad_id: np.zeros((2, 2, 3), dtype=np.uint8)}).save(path)
    assert not path.exists()

    finite_path = tmp_path / "finite.h5"
    _recording_with_ids([1.25], {1.25: np.zeros((2, 2, 3), dtype=np.uint8)}).save(finite_path)
    with h5py.File(finite_path, "r+") as f:
        f["vision_info/stim_ids/values"][0] = float("nan").hex()
    with pytest.raises(ValueError, match="Invalid non-finite float in typed scalar encoding"):
        vnt.read(finite_path)


@pytest.mark.parametrize("bad_id", [b"bytes", (1, 2), object()])
def test_unsupported_stimulus_ids_fail_without_replacing_target(tmp_path, bad_id):
    path = tmp_path / "recording.h5"
    old = b"existing destination"
    path.write_bytes(old)
    recording = _recording_with_ids([1])
    labels = recording._stim_labels
    onsets = recording.vision_onsets
    vision_info = recording.vision_info
    assert labels is not None and onsets is not None and vision_info is not None
    labels[onsets[0]] = bad_id
    vision_info["stim_ids"] = [bad_id]
    with pytest.raises(TypeError, match="Unsupported (object stimulus label|stimulus ID) type"):
        recording.save(path)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


def test_unsupported_object_stimulus_label_never_becomes_empty(tmp_path):
    recording = _recording_with_ids(["ok"])
    labels = recording._stim_labels
    onsets = recording.vision_onsets
    assert labels is not None and onsets is not None
    labels[onsets[0]] = object()
    with pytest.raises(TypeError, match="Unsupported object stimulus label type object"):
        recording.save(tmp_path / "bad-label.h5")
    assert not list(tmp_path.iterdir())


def test_embedded_nul_string_fails_clearly_without_replacing_target(tmp_path):
    path = tmp_path / "recording.h5"
    old = b"existing destination"
    path.write_bytes(old)
    recording = _recording_with_ids(["valid"])
    assert recording.vision_info is not None
    recording.vision_info["stim_ids"] = ["embedded\x00nul"]

    with pytest.raises(ValueError, match="Unsupported stimulus ID string.*embedded NUL"):
        recording.save(path)

    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


def test_trial_metadata_index_nullable_categorical_and_datetimes_roundtrip(tmp_path):
    path = tmp_path / "metadata.h5"
    recording = _recording_with_ids([1, 2, 3])
    index = pd.Index(["trial/α", "trial/β", "trial/γ"], name="trial_name")
    metadata = pd.DataFrame(
        {
            "nullable_int": pd.array([1, pd.NA, 3], dtype="Int64"),
            "nullable_bool": pd.array([True, pd.NA, False], dtype="boolean"),
            "category": pd.Categorical(["train", "test", "train"], categories=["test", "train"], ordered=True),
            "datetime": pd.to_datetime(["2024-01-01", None, "2024-01-03"]),
            "zoned": pd.to_datetime(["2024-02-01", None, "2024-02-03"], utc=True),
        },
        index=index,
    )
    recording.trial_meta = metadata
    recording.save(path)
    loaded = vnt.read(path)
    pd.testing.assert_frame_equal(loaded.trial_meta, metadata)


def test_datetime_categorical_trial_metadata_roundtrip(tmp_path):
    path = tmp_path / "datetime-categories.h5"
    recording = _recording_with_ids([1, 2, 3])
    metadata = pd.DataFrame(
        {
            "naive": pd.Categorical(
                pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
                ordered=True,
            ),
            "zoned": pd.Categorical(
                pd.to_datetime(
                    ["2024-02-01 08:00", "2024-02-02 09:00", "2024-02-01 08:00"],
                    utc=True,
                ).tz_convert("Asia/Shanghai"),
            ),
        }
    )
    recording.trial_meta = metadata

    recording.save(path)

    pd.testing.assert_frame_equal(vnt.read(path).trial_meta, metadata)


def test_trial_metadata_columns_index_roundtrips_name_and_categorical_type(tmp_path):
    path = tmp_path / "column-index.h5"
    recording = _recording_with_ids([1, 2])
    columns = pd.CategoricalIndex(
        ["condition", "score"],
        categories=["score", "condition", "unused"],
        ordered=True,
        name="field_kind",
    )
    metadata = pd.DataFrame([["train", 1], ["test", 2]], columns=columns)
    recording.trial_meta = metadata
    recording.save(path)
    pd.testing.assert_frame_equal(vnt.read(path).trial_meta, metadata)


def test_unsupported_trial_metadata_object_fails_clearly(tmp_path):
    recording = _recording_with_ids([1])
    recording.trial_meta = pd.DataFrame({"bad": [[1, 2]]})
    with pytest.raises(TypeError, match="Unsupported trial metadata column 'bad' type list"):
        recording.save(tmp_path / "bad-meta.h5")


def test_invalid_mutated_trial_state_preserves_target(tmp_path):
    path = tmp_path / "recording.h5"
    path.write_bytes(b"existing")
    recording = _v1_recording()
    recording.vision_onsets = np.array([8, 40])

    assert recording.configured is False
    with pytest.raises(ValueError, match="vision_onsets must be within the recording"):
        recording.save(path)

    assert path.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda f: f["vision_onsets"].__setitem__(1, 40), "vision_onsets must be within the recording"),
        (lambda f: f["trial_starts"].__setitem__(0, 12), "trial end must be greater than start"),
        (lambda f: f.attrs.__setitem__("stim_labels_shape", [2, 20]), "continuous stim_labels must have shape"),
    ],
)
def test_schema1_invalid_trial_state_is_rejected(tmp_path, mutation, message):
    path = tmp_path / "invalid-state.h5"
    _v1_recording().save(path)
    with h5py.File(path, "r+") as f:
        mutation(f)

    with pytest.raises(ValueError, match=message):
        vnt.read(path)


def test_lazy_readers_verify_the_opened_handle(tmp_path, monkeypatch):
    from vneurotk.io import _h5_codec

    path = tmp_path / "snapshot.h5"
    recording = _v1_recording()
    recording.vision.add(
        VisualRepresentations(
            [
                VisualRepresentation(
                    model="model",
                    module_name="features",
                    module_type="Linear",
                    stim_ids=[1, 2],
                    array=np.array([[1.0], [2.0]]),
                )
            ]
        )
    )
    recording.save(path)

    opened = []
    original = _h5_codec.verify_open_file_identity

    def record_open_handle(f, expected, checked_path):
        assert f.id.valid
        opened.append(f)
        original(f, expected, checked_path)

    monkeypatch.setattr(_h5_codec, "verify_open_file_identity", record_open_handle)
    monkeypatch.setattr("vneurotk.io.h5_persistence.verify_open_file_identity", record_open_handle)
    monkeypatch.setattr("vneurotk.io.loader.verify_open_file_identity", record_open_handle)

    loaded = vnt.read(path)
    np.testing.assert_array_equal(loaded.neuro, recording.neuro)
    np.testing.assert_array_equal(loaded.vision.db[1], recording.vision.db[1])
    np.testing.assert_array_equal(loaded.vision["features"], [[1.0], [2.0]])
    assert len(opened) >= 4


def test_atomic_overwrite_preserves_permissions_and_new_file_respects_umask(tmp_path):
    existing = tmp_path / "existing.h5"
    _v1_recording().save(existing)
    os.chmod(existing, 0o640)
    _v1_recording().save(existing)
    assert stat.S_IMODE(existing.stat().st_mode) == 0o640

    new_path = tmp_path / "new.h5"
    old_umask = os.umask(0o027)
    try:
        _v1_recording().save(new_path)
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(new_path.stat().st_mode) == 0o640


def test_atomic_replace_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "recording.h5"
    old = b"old recording"
    path.write_bytes(old)

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        _v1_recording().save(path)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


def test_fixed_byte_text_attributes_decode_consistently(tmp_path):
    path = tmp_path / "fixed-bytes.h5"
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    with h5py.File(path, "w") as f:
        group = f.create_group("stimuli_db")
        group.attrs.create("encoding", np.bytes_("legacy"))
        ds = group.create_dataset("3", data=image)
        ds.attrs.create("key_type", np.bytes_("int"))
        ds.attrs.create("kind", np.bytes_("array"))
    from vneurotk.io import LazyH5Dict

    lazy = LazyH5Dict(path)
    np.testing.assert_array_equal(lazy[3], image)


def test_default_and_disabled_compression_roundtrip(tmp_path):
    default_path = tmp_path / "compressed.h5"
    plain_path = tmp_path / "plain.h5"
    original = _v1_recording()
    original.save(default_path)
    original.save(plain_path, compression=None)
    with h5py.File(default_path, "r") as compressed, h5py.File(plain_path, "r") as plain:
        assert compressed["neuro"].compression == "gzip"
        assert compressed["neuro"].chunks is not None
        assert plain["neuro"].compression is None
        assert plain["neuro"].chunks is not None
    np.testing.assert_array_equal(vnt.read(default_path).neuro, original.neuro)
    np.testing.assert_array_equal(vnt.read(plain_path).neuro, original.neuro)


def test_neuro_info_shape_is_authoritative_and_contradictions_preserve_target(tmp_path):
    path = tmp_path / "shape.h5"
    recording = _v1_recording()
    recording.neuro_info["shape"] = (1, 999)
    path.write_bytes(b"existing")
    with pytest.raises(ValueError, match="contradicts actual neural data shape"):
        recording.save(path)
    assert path.read_bytes() == b"existing"

    good_path = tmp_path / "good-shape.h5"
    good = _v1_recording()
    good.save(good_path)
    with h5py.File(good_path, "r") as f:
        assert tuple(f["neuro_info"].attrs["shape"]) == good.neuro.shape
    with h5py.File(good_path, "r+") as f:
        f["neuro_info"].attrs["shape"] = (1, 999)
    with pytest.raises(ValueError, match="neural dataset shape"):
        vnt.read(good_path)


def test_stale_lazy_neuro_images_and_activations_fail_after_atomic_replace(tmp_path):
    path = tmp_path / "snapshot.h5"
    first = _v1_recording()
    first.vision.add(
        VisualRepresentations(
            [
                VisualRepresentation(
                    model="model",
                    module_name="features",
                    module_type="Linear",
                    stim_ids=[1, 2],
                    array=np.array([[1.0], [2.0]]),
                )
            ]
        )
    )
    first.save(path)

    stale = vnt.read(path)
    stale_record = next(iter(stale.vision._records.values()))
    assert stale._neuro is None and stale_record._array is None
    _v1_recording().save(path)

    changed = "has changed since this object was loaded"
    with pytest.raises(RuntimeError, match=changed):
        _ = stale.neuro
    with pytest.raises(RuntimeError, match=changed):
        _ = stale.vision.db[1]
    with pytest.raises(RuntimeError, match=changed):
        _ = stale.vision["features"]


@pytest.mark.parametrize("payload", ["neuro", "image", "activation"])
def test_lazy_reader_detects_replace_between_open_and_read(tmp_path, monkeypatch, payload):
    path = tmp_path / "snapshot.h5"
    replacement = tmp_path / "replacement.h5"
    recording = _v1_recording()
    recording.vision.add(
        VisualRepresentations(
            [
                VisualRepresentation(
                    model="model",
                    module_name="features",
                    module_type="Linear",
                    stim_ids=[1, 2],
                    array=np.array([[1.0], [2.0]]),
                )
            ]
        )
    )
    recording.save(path)
    loaded = vnt.read(path)
    _v1_recording().save(replacement)

    original_file = h5py.File
    replaced = False

    def replace_before_open(name, mode="r", *args, **kwargs):
        nonlocal replaced
        if not replaced and Path(name) == path and mode == "r":
            os.replace(replacement, path)
            replaced = True
        return original_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", replace_before_open)
    access = {
        "neuro": lambda: loaded.neuro,
        "image": lambda: loaded.vision.db[1],
        "activation": lambda: loaded.vision["features"],
    }[payload]

    with pytest.raises(RuntimeError, match="has changed since this object was loaded"):
        access()
    assert replaced


def test_v1_roundtrip_header_lazy_eager_images_and_metadata(tmp_path):
    path = tmp_path / "v1.h5"
    original = _v1_recording()
    original.save(path)

    with h5py.File(path, "r") as f:
        assert f.attrs[FORMAT_MAGIC_ATTR] == FORMAT_MAGIC
        assert f.attrs[SCHEMA_VERSION_ATTR] == CURRENT_SCHEMA_VERSION
        assert isinstance(f.attrs[WRITER_VERSION_ATTR], str)
        assert f.attrs[WRITER_VERSION_ATTR]

    lazy = vnt.read(path)
    assert lazy._neuro is None
    np.testing.assert_array_equal(lazy.neuro, original.neuro)
    assert lazy.trial_meta.to_dict("list") == original.trial_meta.to_dict("list")
    np.testing.assert_array_equal(lazy.vision.db[2], np.full((2, 3, 3), 2, dtype=np.uint8))

    eager = vnt.read(path, pre_load=True)
    assert eager._neuro is not None
    assert eager._neuro_loader is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda f: f.attrs.__delitem__(SCHEMA_VERSION_ATTR),
            "missing required root attribute 'vneurotk_schema_version'",
        ),
        (lambda f: f.attrs.__delitem__(WRITER_VERSION_ATTR), "missing required root attribute 'writer_version'"),
        (lambda f: f.attrs.__setitem__(FORMAT_MAGIC_ATTR, "not-a-recording"), "vneurotk_format must be 'recording'"),
        (
            lambda f: f.attrs.__setitem__(SCHEMA_VERSION_ATTR, CURRENT_SCHEMA_VERSION + 1),
            rf"supported range is {MIN_SUPPORTED_SCHEMA_VERSION}\.\.{CURRENT_SCHEMA_VERSION}",
        ),
        (
            lambda f: f.attrs.__setitem__(SCHEMA_VERSION_ATTR, "one"),
            "vneurotk_schema_version must be an integer",
        ),
    ],
)
def test_invalid_format_headers_are_rejected(tmp_path, mutation, message):
    path = tmp_path / "bad-header.h5"
    _v1_recording().save(path)
    with h5py.File(path, "r+") as f:
        mutation(f)
    with pytest.raises(ValueError, match=message):
        vnt.read(path)


def test_missing_magic_with_version_is_corrupt(tmp_path):
    path = tmp_path / "missing-magic.h5"
    _v1_recording().save(path)
    with h5py.File(path, "r+") as f:
        del f.attrs[FORMAT_MAGIC_ATTR]
    with pytest.raises(ValueError, match="missing required root attribute 'vneurotk_format'"):
        vnt.read(path)


def test_missing_required_dense_dataset_is_rejected(tmp_path):
    path = tmp_path / "missing-neuro.h5"
    _v1_recording().save(path)
    with h5py.File(path, "r+") as f:
        del f["neuro"]
    with pytest.raises(ValueError, match="missing required dataset 'neuro'"):
        vnt.read(path)


def test_corrupt_coo_components_are_rejected(tmp_path):
    path = tmp_path / "corrupt-coo.h5"
    shutil.copyfile(FIXTURES / "v0_coo_epochs_vision.h5", path)
    with h5py.File(path, "r+") as f:
        del f["neuro_col"]
    with pytest.raises(ValueError, match=r"missing required dataset\(s\): neuro_col"):
        vnt.read(path)


def test_schema1_does_not_accept_legacy_data_mode_typo(tmp_path):
    path = tmp_path / "schema1-legacy-typo.h5"
    _v1_recording().save(path)
    with h5py.File(path, "r+") as f:
        f.attrs["data_mode"] = "continues"
    with pytest.raises(ValueError, match="Invalid data_mode 'continues'"):
        vnt.read(path)
