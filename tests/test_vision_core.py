"""Torch-independent tests for the optional vision data model."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from vneurotk.vision.meta import UNKNOWN, ExtractionProvenance
from vneurotk.vision.representation.visual_representations import (
    VisualRepresentation,
    VisualRepresentations,
)


def _make_vr(
    n_stim: int = 5,
    d: int = 8,
    model: str = "test_model",
    module_name: str = "layer_a",
    module_type: str = "Linear",
) -> VisualRepresentation:
    return VisualRepresentation(
        model=model,
        module_name=module_name,
        module_type=module_type,
        stim_ids=list(range(n_stim)),
        array=np.random.rand(n_stim, d).astype(np.float32),
    )


def _make_vrs(n_stim: int = 5, d: int = 8) -> VisualRepresentations:
    vr_a = VisualRepresentation(
        model="test_model",
        module_name="layer_a",
        module_type="Linear",
        stim_ids=list(range(n_stim)),
        array=np.random.rand(n_stim, d).astype(np.float32),
    )
    vr_b = VisualRepresentation(
        model="test_model",
        module_name="layer_b",
        module_type="ReLU",
        stim_ids=list(range(n_stim)),
        array=np.random.rand(n_stim, 4, d).astype(np.float32),
    )
    return VisualRepresentations([vr_a, vr_b])


# ===========================================================================
# TestExtractionProvenance
# ===========================================================================


class TestExtractionProvenance:
    def test_stable_serialization_roundtrip(self):
        provenance = ExtractionProvenance(
            backend="transformers",
            model_id="org/model",
            model_revision="abc123",
            pretrained=True,
            preprocessing="Processor(size=224)",
            selector="CustomSelector(layer_names=['encoder.0'])",
            dependency_versions={"transformers": "4.50", "torch": "2.7"},
            dtype="float32",
            device="cpu",
            writer_version="0.1.0",
            stimulus_content_hash="sha256:1234",
        )
        encoded = provenance.to_json()
        assert encoded == provenance.to_json()
        assert '"dependency_versions":{"torch":"2.7","transformers":"4.50"}' in encoded
        assert ExtractionProvenance.from_json(encoded) == provenance

    def test_unknown_fields_are_explicit(self):
        provenance = ExtractionProvenance.unknown(model_id="legacy-model")
        assert provenance.model_id == "legacy-model"
        assert provenance.backend == UNKNOWN
        assert provenance.pretrained == UNKNOWN
        assert provenance.stimulus_content_hash is None

    def test_select_preserves_provenance(self):
        provenance = ExtractionProvenance(backend="mock", model_id="test_model")
        vr = VisualRepresentation(
            model="test_model",
            module_name="layer",
            module_type="Linear",
            stim_ids=[0, 1],
            array=np.ones((2, 3)),
            provenance=provenance,
        )
        assert vr.select([1]).provenance is provenance


# ===========================================================================
# TestVisualRepresentation (atomic)
# ===========================================================================


class TestVisualRepresentation:
    def test_basic_properties(self):
        vr = _make_vr(n_stim=10, d=16)
        assert vr.n_stim == 10
        assert vr.shape == (10, 16)
        assert vr.model == "test_model"
        assert vr.module_name == "layer_a"

    def test_select_by_id(self):
        vr = _make_vr(n_stim=5, d=8)
        sub = vr.select([1, 3])
        assert sub.n_stim == 2
        assert list(sub.stim_ids) == [1, 3]
        assert sub.array.shape == (2, 8)

    def test_select_allows_repeated_ids_in_explicit_view(self):
        vr = _make_vr(n_stim=3, d=2)

        sub = vr.select([0, 1, 0])

        assert sub.stim_ids == (0, 1, 0)
        np.testing.assert_array_equal(sub.array, vr.array[[0, 1, 0]])

    def test_select_missing_id_raises(self):
        vr = _make_vr(n_stim=3)
        with pytest.raises(KeyError):
            vr.select([99])

    def test_repr(self):
        vr = _make_vr()
        r = repr(vr)
        assert "VisualRepresentation" in r
        assert "test_model" in r


# ===========================================================================
# TestVisualRepresentations (container)
# ===========================================================================


class TestVisualRepresentations:
    def test_basic_properties(self):
        vrs = _make_vrs(n_stim=10, d=16)
        assert vrs.n_stim == 10
        assert set(vrs.module_names) == {"layer_a", "layer_b"}

    def test_meta_columns(self):
        vrs = _make_vrs()
        assert list(vrs.meta.columns) == ["model", "module_type", "module_name", "shape"]
        assert len(vrs.meta) == 2

    def test_bool_mask_filter(self):
        vrs = _make_vrs(n_stim=5, d=8)
        meta = vrs.meta
        subset = vrs[meta["module_name"] == "layer_a"]
        assert isinstance(subset, VisualRepresentations)
        assert len(subset) == 1
        assert subset[0].module_name == "layer_a"

    def test_bool_mask_multi(self):
        vrs = _make_vrs(n_stim=5, d=8)
        meta = vrs.meta
        subset = vrs[meta["model"] == "test_model"]
        assert len(subset) == 2

    def test_mismatched_stim_ids_raises(self):
        """VisualRepresentations rejects VRs with different stim_ids at construction."""
        vr_a = _make_vr(n_stim=3, module_name="layer_a")
        vr_b = VisualRepresentation(
            model="test_model",
            module_name="layer_b",
            module_type="Linear",
            stim_ids=[10, 20, 30],  # different IDs
            array=np.zeros((3, 4)),
        )
        with pytest.raises(ValueError, match="stim_ids"):
            VisualRepresentations([vr_a, vr_b])

    def test_single_vr_no_validation(self):
        """Single-VR construction never triggers stim_ids check."""
        vr = _make_vr(n_stim=5, module_name="layer_a")
        vrs = VisualRepresentations([vr])
        assert len(vrs) == 1

    def test_int_index(self):
        vrs = _make_vrs()
        vr = vrs[0]
        assert isinstance(vr, VisualRepresentation)

    def test_iter(self):
        vrs = _make_vrs()
        names = [vr.module_name for vr in vrs]
        assert set(names) == {"layer_a", "layer_b"}

    def test_getitem_layer_name(self):
        vrs = _make_vrs(n_stim=5, d=8)
        vr = vrs["layer_a"]
        assert isinstance(vr, VisualRepresentation)
        assert vr.shape == (5, 8)

    def test_numpy_layer(self):
        vrs = _make_vrs(n_stim=5, d=8)
        arr = vrs.numpy("layer_a")
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 8)

    @pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is required")
    def test_to_tensor_layer(self):
        vrs = _make_vrs(n_stim=4, d=8)
        t = vrs.to_tensor("layer_a")
        assert t.shape == (4, 8)

    def test_select_by_id(self):
        vrs = _make_vrs(n_stim=5, d=8)
        sub = vrs.select([1, 3])
        assert sub.n_stim == 2
        assert list(sub.stim_ids) == [1, 3]

    def test_select_by_index(self):
        vrs = _make_vrs(n_stim=5, d=8)
        sub = vrs.select_by_index([0, 4])
        assert list(sub.stim_ids) == [0, 4]

    def test_repr(self):
        vrs = _make_vrs()
        r = repr(vrs)
        assert "VisualRepresentations" in r
        assert "5 stimuli" in r

    def test_select_missing_id_raises(self):
        vrs = _make_vrs(n_stim=3)
        with pytest.raises(KeyError):
            vrs.select([99])

    def test_empty_container(self):
        vrs = VisualRepresentations([])
        assert len(vrs) == 0
        assert vrs.n_stim == 0
        assert len(vrs.meta) == 0

    def test_by_module_returns_single_vr(self):
        vrs = _make_vrs(n_stim=5, d=8)
        vr = vrs.by_module("layer_a")
        assert isinstance(vr, VisualRepresentation)
        assert vr.module_name == "layer_a"

    def test_by_module_raises_on_unknown_name(self):
        vrs = _make_vrs()
        with pytest.raises(KeyError, match="not found"):
            vrs.by_module("nonexistent_layer")

    def test_by_module_with_model_param(self):
        """model= 参数精确匹配。"""
        vr_a = VisualRepresentation(
            model="modelA",
            module_name="layer1",
            module_type="L",
            stim_ids=[0],
            array=np.zeros((1, 2)),
        )
        vr_b = VisualRepresentation(
            model="modelB",
            module_name="layer1",
            module_type="L",
            stim_ids=[0],
            array=np.ones((1, 2)),
        )
        vrs = VisualRepresentations([vr_a, vr_b])
        result = vrs.by_module("layer1", model="modelA")
        assert result.model == "modelA"
        np.testing.assert_array_equal(result.array, np.zeros((1, 2)))

    def test_by_module_ambiguous_raises(self):
        """多模型共享 module_name 且未指定 model= 时报 KeyError。"""
        vr_a = VisualRepresentation(
            model="modelA",
            module_name="layer1",
            module_type="L",
            stim_ids=[0],
            array=np.zeros((1, 2)),
        )
        vr_b = VisualRepresentation(
            model="modelB",
            module_name="layer1",
            module_type="L",
            stim_ids=[0],
            array=np.ones((1, 2)),
        )
        vrs = VisualRepresentations([vr_a, vr_b])
        with pytest.raises(KeyError, match="disambiguate"):
            vrs.by_module("layer1")

    def test_by_module_model_not_found_raises(self):
        """指定 model= 但该 model 下无此 module 时报 KeyError。"""
        vrs = _make_vrs()
        with pytest.raises(KeyError, match="not found"):
            vrs.by_module("layer_a", model="nonexistent_model")

    def test_filter_returns_subset(self):
        vrs = _make_vrs(n_stim=5, d=8)
        mask = vrs.meta["module_name"] == "layer_a"
        subset = vrs.filter(mask)
        assert isinstance(subset, VisualRepresentations)
        assert len(subset) == 1
        assert subset[0].module_name == "layer_a"

    def test_getitem_str_delegates_to_by_module(self):
        vrs = _make_vrs(n_stim=5, d=8)
        assert vrs["layer_a"].module_name == vrs.by_module("layer_a").module_name

    def test_getitem_bool_mask_delegates_to_filter(self):
        vrs = _make_vrs(n_stim=5, d=8)
        mask = vrs.meta["module_name"] == "layer_b"
        assert vrs[mask][0].module_name == vrs.filter(mask)[0].module_name


class TestAssertSharedStimIds:
    """Direct coverage for the shared-stimulus validator."""

    @staticmethod
    def _make_vr(module_name: str, stim_ids: list[int]) -> VisualRepresentation:
        return VisualRepresentation(
            model="m",
            module_name=module_name,
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.zeros((len(stim_ids), 4)),
        )

    def test_empty_list_passes(self):
        VisualRepresentations._assert_shared_stim_ids([])

    def test_single_vr_passes(self):
        VisualRepresentations._assert_shared_stim_ids([self._make_vr("a", [0, 1, 2])])

    def test_matching_stim_ids_passes(self):
        representations = [self._make_vr("a", [0, 1]), self._make_vr("b", [0, 1])]
        VisualRepresentations._assert_shared_stim_ids(representations)

    def test_validator_rejects_mismatched_stim_ids(self):
        representations = [self._make_vr("a", [0, 1]), self._make_vr("b", [0, 2])]
        with pytest.raises(ValueError, match="stim_ids"):
            VisualRepresentations._assert_shared_stim_ids(representations)


# ===========================================================================
# TestVisionDataNamed
# ===========================================================================


class TestVisionDataNamed:
    def _make_vision_data(self, n_stim: int = 4, d: int = 8) -> Any:
        from vneurotk.vision.data import VisionData

        output_order = np.array([2, 0, 1, 3])  # shuffled
        vd = VisionData(output_order=output_order)
        stim_ids = list(range(n_stim))
        vr = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.arange(n_stim * d, dtype=np.float32).reshape(n_stim, d),
        )
        vrs = VisualRepresentations([vr])
        vd.add(vrs)
        return vd

    def test_by_module_returns_trial_aligned_array(self):

        vd = self._make_vision_data()
        arr = vd.by_module("layer_x")
        assert isinstance(arr, np.ndarray)
        # output_order=[2,0,1,3] → row 2 first, row 0 second, etc.
        assert arr.shape[0] == 4
        np.testing.assert_array_equal(arr[0], vd["layer_x"][0])

    def test_multi_module_view_allows_repeated_output_ids(self):
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.array([0, 1, 0]))
        source = VisualRepresentations(
            [
                VisualRepresentation(
                    model="m",
                    module_name=module_name,
                    module_type="Linear",
                    stim_ids=[0, 1],
                    array=np.array([[offset], [offset + 1]]),
                )
                for module_name, offset in [("layer_a", 10), ("layer_b", 20)]
            ]
        )
        vd.add(source)

        aligned = vd[np.array([True, True])]

        assert isinstance(aligned, VisualRepresentations)
        assert aligned.stim_ids == (0, 1, 0)
        np.testing.assert_array_equal(aligned["layer_a"].array[:, 0], [10, 11, 10])
        np.testing.assert_array_equal(aligned["layer_b"].array[:, 0], [20, 21, 20])

    def test_vision_data_rejects_repeated_selected_view_as_source(self):
        from vneurotk.vision.data import VisionData

        selected = _make_vrs(n_stim=2, d=2).select([0, 1, 0])
        vd = VisionData(output_order=np.array([0, 1, 0]))

        with pytest.raises(ValueError, match="stores unique-stimulus representations"):
            vd.add(selected)

        assert not vd.has_visual_representations

    def test_by_module_raises_on_unknown_name(self):

        vd = self._make_vision_data()
        with pytest.raises(KeyError):
            vd.by_module("does_not_exist")

    def test_getitem_str_matches_by_module(self):
        vd = self._make_vision_data()
        np.testing.assert_array_equal(vd["layer_x"], vd.by_module("layer_x"))


class TestPR6Validation:
    """Focused, torch-independent coverage for PR 6 invariants."""

    def test_stimulus_set_from_dict_requires_all_unique_ids(self):
        from vneurotk.core.stimulus import StimulusSet

        with pytest.raises(ValueError, match=r"missing 2 unique stimulus ID\(s\): \[2, 3\]"):
            StimulusSet.from_dict(np.array([1, 2, 1, 3]), {1: "image-1"})

    def test_stimulus_set_from_dict_ignores_extra_keys(self):
        from vneurotk.core.stimulus import StimulusSet

        stimulus_set = StimulusSet.from_dict(
            np.array([2, 1, 2]),
            {1: "image-1", 2: "image-2", 99: "unused"},
        )

        assert list(stimulus_set.items()) == [(2, "image-2"), (1, "image-1")]
        assert 99 not in stimulus_set

    def test_visual_representation_rejects_array_length_mismatch(self):
        with pytest.raises(ValueError, match=r"stim_ids length 2.*first dimension 3.*module='layer_x'"):
            VisualRepresentation(
                model="m",
                module_name="layer_x",
                module_type="Linear",
                stim_ids=[10, 20],
                array=np.zeros((3, 4)),
            )

    def test_visual_representation_validates_lazy_shape_without_loading(self):
        loaded = False

        def loader() -> np.ndarray:
            nonlocal loaded
            loaded = True
            return np.zeros((3, 4))

        with pytest.raises(ValueError, match=r"stim_ids length 2.*first dimension 3.*shape=\(3, 4\)"):
            VisualRepresentation(
                model="m",
                module_name="lazy_layer",
                module_type="Linear",
                stim_ids=[10, 20],
                array_loader=loader,
                shape=(3, 4),
            )

        assert loaded is False

    def test_visual_representation_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match=r"stim_ids must be unique.*duplicate ID\(s\): \[10\]"):
            VisualRepresentation(
                model="m",
                module_name="layer_x",
                module_type="Linear",
                stim_ids=[10, 20, 10],
                array=np.zeros((3, 4)),
            )

    def test_relevant_images_lists_all_missing_ids(self):
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.array([3, 1, 3, 2]))
        with pytest.raises(ValueError, match=r"Missing 2 stimulus ID\(s\): \[1, 2\]"):
            vd._relevant_images({3: "image-3"})

    def test_extraction_rejects_incomplete_activations_without_storing(self):
        from vneurotk.vision.data import VisionData

        class IncompleteModel:
            model_id = "incomplete-model"
            module_names = ["layer_x"]

            def extract_for_modules(self, images, module_names, batch_size):
                assert list(images) == [0, 1, 2]
                return VisualRepresentations(
                    [
                        VisualRepresentation(
                            model=self.model_id,
                            module_name=module_names[0],
                            module_type="Linear",
                            stim_ids=[0, 1],
                            array=np.zeros((2, 4)),
                        )
                    ]
                )

        vd = VisionData(output_order=np.array([0, 1, 2]))
        with pytest.raises(ValueError, match=r"Missing 1 ID\(s\): \[2\]"):
            vd.extract_from(IncompleteModel(), vision_db={0: "a", 1: "b", 2: "c"})

        assert not vd.has_visual_representations

    def test_output_order_setter_is_atomic_when_record_lacks_ids(self):
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.array([0, 1]))
        vd.add(
            VisualRepresentations(
                [
                    VisualRepresentation(
                        model="m",
                        module_name="layer_x",
                        module_type="Linear",
                        stim_ids=[0, 1],
                        array=np.arange(8).reshape(2, 4),
                    )
                ]
            )
        )
        cached = vd.by_module("layer_x")

        with pytest.raises(ValueError, match=r"Missing 1 ID\(s\): \[2\]"):
            vd.output_order = np.array([1, 2])

        np.testing.assert_array_equal(vd.output_order, np.array([0, 1]))
        assert vd.by_module("layer_x") is cached
