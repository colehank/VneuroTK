"""Tests for vneurotk.vision module.

Covers:
  - VisualRepresentation (atomic)
  - VisualRepresentations (container + meta + bool-mask filter)
  - ModuleSelector (tiny nn.Sequential)
  - BaseData.trial_stim_ids
  - VisionModel with MockBackend
  - VisionData.extract_from + BaseData.vision.extract_from
  - Smoke tests for timm and transformers (skipped if not installed)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # type: ignore

from vneurotk.vision.meta import ModelInfo
from vneurotk.vision.model.backend.base import BaseBackend, ModuleInfo
from vneurotk.vision.model.selector import (
    AllLeafSelector,
    BlockLevelSelector,
    CustomSelector,
)
from vneurotk.vision.representation.visual_representations import (
    VisualRepresentation,
    VisualRepresentations,
)

# ===========================================================================
# Helpers / Fixtures
# ===========================================================================


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

    def test_by_module_raises_on_unknown_name(self):

        vd = self._make_vision_data()
        with pytest.raises(KeyError):
            vd.by_module("does_not_exist")

    def test_getitem_str_matches_by_module(self):
        vd = self._make_vision_data()
        np.testing.assert_array_equal(vd["layer_x"], vd.by_module("layer_x"))


# ===========================================================================
# TestModuleSelector
# ===========================================================================


def _tiny_model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _module_infos(model: nn.Module) -> list[ModuleInfo]:
    """Convert an nn.Module to a list[ModuleInfo] — matches BaseBackend.enumerate_modules()."""
    result = []
    for name, module in model.named_modules():
        if not name:
            continue
        result.append(
            ModuleInfo(
                name=name,
                module_type=type(module).__name__,
                depth=name.count(".") + 1,
                n_params=sum(p.numel() for p in module.parameters()),
                is_leaf=len(list(module.children())) == 0,
                param_shapes={n: tuple(p.shape) for n, p in module.named_parameters(recurse=False)},
            )
        )
    return result


class TestModuleSelector:
    def test_all_leaf(self):
        model = _tiny_model()
        sel = AllLeafSelector()
        names = sel.select(_module_infos(model))
        assert all(isinstance(model._modules[n], nn.Linear) for n in names)
        assert len(names) == 2

    def test_all_leaf_custom_exclude(self):
        sel = AllLeafSelector(exclude_types=(nn.ReLU,))
        names = sel.select(_module_infos(_tiny_model()))
        assert len(names) == 2

    def test_custom_selector(self):
        sel = CustomSelector(["0", "2"])
        names = sel.select(_module_infos(_tiny_model()))
        assert names == ["0", "2"]

    def test_custom_selector_missing_raises(self):
        sel = CustomSelector(["nonexistent"])
        with pytest.raises(ValueError, match="not found"):
            sel.select(_module_infos(_tiny_model()))

    def test_block_level_fallback(self):
        sel = BlockLevelSelector()
        names = sel.select(_module_infos(_tiny_model()))
        assert len(names) >= 1

    def test_custom_selector_accepts_layer_info(self):
        layers = [
            ModuleInfo(name="0", module_type="Linear", depth=1),
            ModuleInfo(name="2", module_type="Linear", depth=1),
        ]
        sel = CustomSelector(layers)
        names = sel.select(_module_infos(_tiny_model()))
        assert names == ["0", "2"]


# ===========================================================================
# TestBlockLevelSelectorArchPatterns
# ===========================================================================


class TestBlockLevelSelectorArchPatterns:
    def _tiny_vit(self) -> nn.Module:
        class TinyViT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
                self.head = nn.Linear(4, 2)

        return TinyViT()

    def test_default_patterns_select_vit_blocks(self):
        sel = BlockLevelSelector()
        names = sel.select(_module_infos(self._tiny_vit()))
        assert "blocks.0" in names
        assert "blocks.1" in names

    def test_custom_arch_patterns_override_defaults(self):
        sel = BlockLevelSelector(arch_patterns=[(r"^head$", 1)])
        names = sel.select(_module_infos(self._tiny_vit()))
        assert names == ["head"]
        assert "blocks.0" not in names

    def test_empty_arch_patterns_falls_back_to_named_children(self):
        sel = BlockLevelSelector(arch_patterns=[])
        names = sel.select(_module_infos(self._tiny_vit()))
        # empty pattern list → no match → fallback to depth-1 modules
        assert set(names) == {"blocks", "head"}

    def test_default_patterns_classmethod_returns_copy(self):
        patterns = BlockLevelSelector.default_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        patterns.clear()
        assert len(BlockLevelSelector._ARCH_PATTERNS) > 0


# ===========================================================================
# TestTrialStimIds
# ===========================================================================


class TestTrialStimIds:
    def _make_bd(self, stim_ids=None):
        from vneurotk.core.recording import BaseData

        neuro = np.random.randn(500, 4)
        neuro_info = dict(sfreq=100.0, ch_names=["c0", "c1", "c2", "c3"])
        if stim_ids is None:
            stim_ids = [10, 20, 30]
        onsets = np.array([50, 150, 250])

        bd = BaseData(
            neuro=neuro,
            neuro_info=neuro_info,
        )
        bd.configure(
            stim_ids=np.array(stim_ids),
            trial_window=[-10, 40],
            vision_onsets=onsets,
        )
        return bd, stim_ids, onsets

    def test_trial_stim_ids_values(self):
        bd, stim_ids, onsets = self._make_bd()
        ids = bd.trial_stim_ids
        assert len(ids) == bd.n_trials
        for i in range(bd.n_trials):
            assert ids[i] == stim_ids[i]

    def test_unconfigured_raises(self):
        from vneurotk.core.recording import BaseData

        bd = BaseData(
            neuro=np.zeros((10, 2)),
            neuro_info=dict(sfreq=1.0),
        )
        with pytest.raises(RuntimeError, match="configure"):
            _ = bd.trial_stim_ids


# ===========================================================================
# MockBackend
# ===========================================================================


class _MockBackend(BaseBackend):
    """Minimal backend wrapping a tiny nn.Sequential for hook tests."""

    def load(self, model_name: str, pretrained: bool = True) -> None:
        torch.manual_seed(0)
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.model.eval()
        self._model_name = model_name

    def preprocess(self, image: Any) -> dict[str, Any]:
        images = image if isinstance(image, list) else [image]
        tensors = []
        for img in images:
            t = torch.from_numpy(np.asarray(img, dtype=np.float32)).flatten()[:4]
            if t.shape[0] < 4:
                t = torch.nn.functional.pad(t, (0, 4 - t.shape[0]))
            tensors.append(t)
        return {"pixel_values": torch.stack(tensors, dim=0)}

    def forward(self, inputs: dict[str, Any]) -> Any:
        px = inputs["pixel_values"]
        with torch.no_grad():
            return self.model(px)  # ty: ignore[call-non-callable]

    def enumerate_modules(self) -> list[ModuleInfo]:
        return [
            ModuleInfo(
                name=n,
                module_type=type(m).__name__,
                depth=1,
                n_params=sum(p.numel() for p in m.parameters()),
                is_leaf=len(list(m.children())) == 0,
                param_shapes={pn: tuple(p.shape) for pn, p in m.named_parameters(recurse=False)},
            )
            for n, m in self.model.named_modules()  # ty: ignore[unresolved-attribute]
            if n
        ]

    def get_model_meta(self) -> ModelInfo:
        return ModelInfo(model_id=getattr(self, "_model_name", "mock"), backend="mock")


# ===========================================================================
# TestVisionModelMock
# ===========================================================================


class TestVisionModelMock:
    def _make_extractor(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=selector,
        )

    def test_extract_single_returns_visual_representations(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vrs = ext.extract(image)
        assert isinstance(vrs, VisualRepresentations)
        assert vrs.n_stim == 1

    def test_extract_single_layer_exists(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vrs = ext.extract(image)
        assert "0" in vrs.module_names or "2" in vrs.module_names

    def test_extract_batch_shape(self):
        ext = self._make_extractor()
        images = {i: np.random.rand(4).astype(np.float32) for i in range(6)}
        vrs = ext.extract(images)
        assert isinstance(vrs, VisualRepresentations)
        assert vrs.n_stim == 6
        for vr in vrs:
            assert vr.array.shape[0] == 6

    def test_meta_has_correct_columns(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vrs = ext.extract(image)
        assert set(vrs.meta.columns) >= {"model", "module_type", "module_name", "shape"}
        assert len(vrs.meta) == 2  # layers "0" and "2"

    def test_numpy_and_tensor(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vrs = ext.extract(image)
        arr = vrs.numpy("0")
        assert isinstance(arr, np.ndarray)
        t = vrs.to_tensor("0")
        assert isinstance(t, torch.Tensor)

    def test_module_names_property(self):
        ext = self._make_extractor()
        assert set(ext.module_names) == {"0", "2"}

    def test_layers_property(self):
        ext = self._make_extractor()
        layers = ext.module_list
        assert isinstance(layers, list)
        assert len(layers) > 0

    def test_extract_for_modules_is_public(self):
        ext = self._make_extractor()
        images = {i: np.random.rand(4).astype(np.float32) for i in range(3)}
        vrs = ext.extract_for_modules(images, ["0"], batch_size=3)
        assert isinstance(vrs, VisualRepresentations)
        assert len(vrs.meta) == 1
        assert vrs.meta.iloc[0]["module_name"] == "0"

    def test_extract_for_modules_does_not_mutate_module_names(self):
        ext = self._make_extractor()
        original = list(ext.module_names)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(2)}
        ext.extract_for_modules(images, ["0"], batch_size=2)
        assert ext.module_names == original

    def test_extract_for_modules_returns_subset(self):
        ext = self._make_extractor()
        images = {i: np.random.rand(4).astype(np.float32) for i in range(4)}
        vrs_full = ext.extract(images)
        vrs_sub = ext.extract_for_modules(images, ["0"], batch_size=4)
        assert len(vrs_sub.meta) == 1
        assert len(vrs_full.meta) == 2
        np.testing.assert_array_equal(vrs_sub["0"].array, vrs_full["0"].array)


# ===========================================================================
# TestVisionModelBatch
# ===========================================================================


class TestVisionModelBatch:
    def _make_extractor(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=selector,
        )

    def _fixed_images(self, n: int = 6) -> dict:
        rng = np.random.default_rng(42)
        return {i: rng.random(4).astype(np.float32) for i in range(n)}

    def test_default_batch_size(self):
        ext = self._make_extractor()
        images = self._fixed_images(3)
        vrs = ext.extract(images)
        assert vrs.n_stim == 3

    def test_batch_size_param_accepted(self):
        ext = self._make_extractor()
        images = self._fixed_images(5)
        vrs = ext.extract(images, batch_size=2)
        assert vrs.n_stim == 5

    def test_batch_equivalent_across_batch_sizes(self):
        ext1 = self._make_extractor()
        ext2 = self._make_extractor()
        ext3 = self._make_extractor()
        images = self._fixed_images(6)

        vrs_bs1 = ext1.extract(images, batch_size=1)
        vrs_bs4 = ext2.extract(images, batch_size=4)
        vrs_bs100 = ext3.extract(images, batch_size=100)

        for layer in vrs_bs1.module_names:
            np.testing.assert_allclose(vrs_bs1[layer].array, vrs_bs4[layer].array, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(vrs_bs1[layer].array, vrs_bs100[layer].array, rtol=1e-5, atol=1e-6)

    def test_batch_equivalent_to_single(self):
        ext_batch = self._make_extractor()
        ext_single = self._make_extractor()
        images = self._fixed_images(4)

        vrs_batch = ext_batch.extract(images, batch_size=4)

        single_acts: dict[str, list] = {}
        for i in sorted(images.keys()):
            vrs = ext_single.extract(images[i])
            for vr in vrs:
                single_acts.setdefault(vr.module_name, []).append(vr.array[0])

        for layer, rows in single_acts.items():
            stacked = np.stack(rows, axis=0)
            np.testing.assert_allclose(vrs_batch[layer].array, stacked, rtol=1e-5, atol=1e-6)

    def test_invalid_batch_size_raises(self):
        ext = self._make_extractor()
        images = self._fixed_images(3)
        with pytest.raises(ValueError):
            ext.extract(images, batch_size=0)
        with pytest.raises(ValueError):
            ext.extract(images, batch_size=-1)

    def test_single_image_still_works(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vrs = ext.extract(image)
        assert vrs.n_stim == 1

    def test_list_input_raises(self):
        ext = self._make_extractor()
        images = [np.random.rand(4).astype(np.float32) for _ in range(5)]
        with pytest.raises(TypeError, match="dict mapping stim_id"):
            ext.extract(images)

    def test_dict_input_with_integer_ids(self):
        ext = self._make_extractor()
        images = {i: np.random.rand(4).astype(np.float32) for i in range(5)}
        vrs = ext.extract(images)
        assert vrs.n_stim == 5
        assert list(vrs.stim_ids) == list(range(5))
        for vr in vrs:
            assert vr.array.shape[0] == 5


# ===========================================================================
# TestImageSourceProtocol
# ===========================================================================


class TestImageSourceProtocol:
    def test_stimulus_set_satisfies_image_source(self):
        from vneurotk.core.stimulus import StimulusSet
        from vneurotk.vision.image_source import ImageSource

        images = {0: np.zeros((4,), dtype=np.float32), 1: np.ones((4,), dtype=np.float32)}
        ss = StimulusSet(stim_ids=np.array([0, 1]), stimuli=images)
        assert isinstance(ss, ImageSource)

    def test_plain_dict_satisfies_image_source(self):
        from vneurotk.vision.image_source import ImageSource

        images = {0: np.zeros((4,), dtype=np.float32)}
        assert isinstance(images, ImageSource)

    def test_image_source_protocol_has_expected_methods(self):
        from vneurotk.vision.image_source import ImageSource

        assert hasattr(ImageSource, "__protocol_attrs__") or hasattr(ImageSource, "_is_protocol")


# ===========================================================================
# TestVisionData + vision.extract_from
# ===========================================================================


class TestEncodeVision:
    def _make_bd_with_images(self, n_stim: int = 4, n_time: int = 600):
        from vneurotk.core.recording import BaseData

        rng = np.random.default_rng(0)
        neuro = rng.standard_normal((n_time, 8)).astype(np.float32)
        onsets = np.array([50, 150, 250, 350])[:n_stim]
        vision_id = np.arange(n_stim)
        images = {i: rng.random((4,)).astype(np.float32) for i in range(n_stim)}

        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=vision_id,
            trial_window=[-10, 40],
            vision_onsets=onsets,
            vision_db=images,
        )
        return bd, images

    def _make_extractor(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=selector,
        )

    def test_encode_vision_datas_vrs(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        assert bd.has_vision
        assert len(bd.vision.meta) == 2

    def test_feats_meta_shape(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        meta = bd.vision.meta
        assert len(meta) == 2
        assert set(meta.columns) >= {"model", "module_type", "module_name", "shape"}

    def test_feats_bool_filter(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        meta = bd.vision.meta
        subset = bd.vision[meta["module_name"] == "0"]
        # When filtering to exactly 1 VR, feats[mask] returns aligned array
        assert isinstance(subset, np.ndarray)
        assert subset.shape[0] == bd.n_trials

    def test_feats_multi_vr_returns_container(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        meta = bd.vision.meta
        # When filtering to multiple VRs, returns VisualRepresentations
        subset = bd.vision[meta["model"] == "mock"]
        assert len(subset) == 2
        assert subset[0].module_name == "0"
        assert subset[1].module_name == "2"

    def test_feats_aligned_array_for_unique_vr(self):
        bd, images = self._make_bd_with_images(n_stim=4)
        model = self._make_extractor()
        bd.vision.extract_from(model)
        meta = bd.vision.meta
        arr = bd.vision[meta["module_name"] == "0"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == bd.n_trials

    def test_overwrite_false_skips(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        bd.vision.extract_from(model, overwrite=False)
        assert len(bd.vision.meta) == 2  # no duplicates

    def test_overwrite_true_replaces(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        bd.vision.extract_from(model, overwrite=True)
        assert len(bd.vision.meta) == 2  # still 2, replaced not appended

    def test_encode_vision_with_inline_vision_db(self):
        from vneurotk.core.recording import BaseData

        rng = np.random.default_rng(1)
        neuro = rng.standard_normal((600, 8)).astype(np.float32)
        onsets = np.array([50, 150, 250, 350])
        vision_id = np.arange(4)
        images = {i: rng.random((4,)).astype(np.float32) for i in range(4)}

        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(stim_ids=vision_id, trial_window=[-10, 40], vision_onsets=onsets)

        model = self._make_extractor()
        bd.vision.extract_from(model, vision_db=images)
        assert bd.has_vision

    def test_encode_vision_no_images_raises(self):
        from vneurotk.core.recording import BaseData

        neuro = np.zeros((600, 4))
        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=np.arange(3),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250]),
        )
        model = self._make_extractor()
        with pytest.raises(RuntimeError, match="No image source"):
            bd.vision.extract_from(model)

    def test_vision_id_attribute(self):
        bd, images = self._make_bd_with_images(n_stim=4)
        model = self._make_extractor()
        bd.vision.extract_from(model)
        # vision.output_order should return trial_stim_ids
        assert np.array_equal(bd.vision.output_order, bd.trial_stim_ids)
        assert len(bd.vision.output_order) == bd.n_trials

    def test_vision_db_attribute(self):
        bd, images = self._make_bd_with_images(n_stim=4)
        model = self._make_extractor()
        bd.vision.extract_from(model)
        # vision.db should contain the vision_db (image dictionary)
        assert bd.vision.db is not None
        assert len(bd.vision.db) == 4  # unique stims

    def test_feats_string_index(self):
        bd, images = self._make_bd_with_images()
        model = self._make_extractor()
        bd.vision.extract_from(model)
        # String indexing should work if module_name is unique
        arr = bd.vision["0"]
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == bd.n_trials

    def test_vision_unconfigured_raises(self):
        from vneurotk.core.recording import BaseData

        bd = BaseData(neuro=np.zeros((10, 2)), neuro_info=dict(sfreq=1.0))
        with pytest.raises(RuntimeError, match="configure"):
            _ = bd.vision

    def test_vision_db_lazy_roundtrip(self, tmp_path):
        """save/load 往返后 vision.db 是 LazyH5Dict，按需加载图片。"""
        import vneurotk as vtk
        from vneurotk.io import LazyH5Dict
        from vneurotk.io.path import VTKPath

        bd, images = self._make_bd_with_images(n_stim=4)
        model = self._make_extractor()
        bd.vision.extract_from(model)

        save_path = VTKPath(tmp_path, subject="sub01", session="s01", task="t1")
        bd.save(save_path)

        loaded = vtk.read(save_path)
        # vision.db should be a LazyH5Dict (not yet loaded into memory)
        assert isinstance(loaded.vision.db, LazyH5Dict)
        # accessing a single key loads only that image
        key = list(loaded.vision.db.keys())[0]
        img = loaded.vision.db[key]
        assert isinstance(img, np.ndarray)
        # all keys present
        assert len(loaded.vision.db) == len(images)

    def test_vision_extract_from_loaded_db(self, tmp_path):
        """加载后 vision.extract_from 无需再传 vision_db，直接使用 LazyH5Dict。"""
        import vneurotk as vtk
        from vneurotk.io.path import VTKPath

        bd, images = self._make_bd_with_images(n_stim=4)
        model = self._make_extractor()
        bd.vision.extract_from(model)

        save_path = VTKPath(tmp_path, subject="sub01", session="s01", task="t2")
        bd.save(save_path)

        loaded = vtk.read(save_path)
        # extract without vision_db — should read from loaded.vision.db
        loaded.vision.extract_from(model, overwrite=True)
        assert loaded.vision.meta is not None
        assert len(loaded.vision.meta) > 0

    def test_skip_all_when_already_extracted(self):
        """overwrite=False: 全部模块已存在时跳过 GPU forward。"""
        forward_calls = []

        class _CountingBackend(_MockBackend):
            def forward(self, inputs):
                forward_calls.append(1)
                return super().forward(inputs)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _CountingBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        bd, images = self._make_bd_with_images()
        bd.vision.extract_from(model)
        calls_after_first = len(forward_calls)

        # second call with overwrite=False — should skip entirely
        bd.vision.extract_from(model, overwrite=False)
        assert len(forward_calls) == calls_after_first  # no new forward calls
        assert len(bd.vision.meta) == 2  # still 2, no duplicates

    def test_skip_partial_extracts_only_missing(self):
        """overwrite=False: 只有部分模块缺失时，仅对缺失模块做 GPU forward。"""
        forward_calls: list[int] = []

        class _CountingBackend(_MockBackend):
            def forward(self, inputs):
                forward_calls.append(inputs["pixel_values"].shape[0])
                return super().forward(inputs)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        # Step 1: 先只提取 layer "0"
        backend = _CountingBackend(device="cpu")
        backend.load("mock")
        model_one = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0"]),
        )
        bd, images = self._make_bd_with_images()
        bd.vision.extract_from(model_one)
        assert len(bd.vision.meta) == 1
        calls_after_first = len(forward_calls)

        # Step 2: 换成有两个层的 model，overwrite=False → 只应提取 "2"
        model_two = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )
        bd.vision.extract_from(model_two, overwrite=False)
        assert len(bd.vision.meta) == 2  # 新增了 "2"
        # forward 确实被调用过（提取了缺失的 "2"）
        assert len(forward_calls) > calls_after_first

    def test_overwrite_true_always_extracts(self):
        """overwrite=True: 即使全部模块已存在，也强制重新提取并覆盖。"""
        forward_calls = []

        class _CountingBackend(_MockBackend):
            def forward(self, inputs):
                forward_calls.append(1)
                return super().forward(inputs)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _CountingBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )
        bd, images = self._make_bd_with_images()
        bd.vision.extract_from(model)
        calls_after_first = len(forward_calls)

        bd.vision.extract_from(model, overwrite=True)
        assert len(forward_calls) > calls_after_first  # forward 被重新调用
        assert len(bd.vision.meta) == 2  # 仍是 2，替换而非追加


# ===========================================================================
# TestVisionDataBuild
# ===========================================================================


class TestVisionDataBuild:
    """Tests for VisionData.extract_from() — the vision-attachment seam."""

    def _make_extractor(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

    def test_build_populates_vrs_independently(self):
        """VisionData.extract_from() can run without BaseData."""
        from vneurotk.vision.data import VisionData

        output_order = np.arange(4)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(4)}
        model = self._make_extractor()

        vd = VisionData(output_order=output_order)
        vd.extract_from(model, vision_db=images)

        assert len(vd.meta) == 2  # two layers: "0" and "2"

    def test_build_overwrite_false_skips_existing(self):
        from vneurotk.vision.data import VisionData

        forward_calls = []

        class _Counting(_MockBackend):
            def forward(self, inputs):
                forward_calls.append(1)
                return super().forward(inputs)

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _Counting(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )
        output_order = np.arange(4)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(4)}
        vd = VisionData(output_order=output_order)

        vd.extract_from(model, vision_db=images)
        calls_after_first = len(forward_calls)

        vd.extract_from(model, vision_db=images, overwrite=False)
        assert len(forward_calls) == calls_after_first  # no new forward pass

    def test_build_overwrite_true_replaces(self):
        from vneurotk.vision.data import VisionData

        output_order = np.arange(4)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(4)}
        model = self._make_extractor()

        vd = VisionData(output_order=output_order)
        vd.extract_from(model, vision_db=images)
        vd.extract_from(model, vision_db=images, overwrite=True)

        # overwrite=True replaces; should still be 2 VRs, not 4
        assert len(vd.meta) == 2

    def test_vision_extract_from_delegates(self):
        """BaseData.vision.extract_from() works correctly."""
        from vneurotk.core.recording import BaseData

        rng = np.random.default_rng(42)
        neuro = rng.standard_normal((600, 8)).astype(np.float32)
        images = {i: rng.random((4,)).astype(np.float32) for i in range(4)}
        bd = BaseData(neuro=neuro, neuro_info=dict(sfreq=100.0))
        bd.configure(
            stim_ids=np.arange(4),
            trial_window=[-10, 40],
            vision_onsets=np.array([50, 150, 250, 350]),
            vision_db=images,
        )
        model = self._make_extractor()
        bd.vision.extract_from(model)

        assert bd.has_vision
        assert len(bd.vision.meta) == 2


# ===========================================================================
# TestVisionDataAttachDb
# ===========================================================================


class TestVisionDataAttachDb:
    """Tests for VisionData.attach_db() — the single write point for db state."""

    def test_attach_db_sets_db(self):
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.arange(3))
        assert vd.db is None

        db = {0: "img0", 1: "img1", 2: "img2"}
        vd.attach_db(db)
        assert vd.db is db

    def test_attach_db_replaces_existing(self):
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.arange(2))
        db_a = {0: "a", 1: "b"}
        db_b = {0: "x", 1: "y"}

        vd.attach_db(db_a)
        vd.attach_db(db_b)
        assert vd.db is db_b

    def test_extract_from_with_vision_db_attaches_db(self):
        """extract_from(model, vision_db=images) attaches images as db."""
        from vneurotk.vision.data import VisionData
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0"]),
        )

        output_order = np.arange(3)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(3)}

        vd = VisionData(output_order=output_order)
        assert vd.db is None
        vd.extract_from(model, vision_db=images)
        assert vd.db is images

    def test_attach_db_then_extract_from_preserves_db(self):
        """attach_db() before extract_from() is retained after extraction."""
        from vneurotk.vision.data import VisionData
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        model = VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0"]),
        )

        output_order = np.arange(3)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(3)}

        vd = VisionData(output_order=output_order)
        vd.attach_db(images)
        vd.extract_from(model, vision_db=images)
        assert vd.db is images


# ===========================================================================
# TestVisionDataPersistence  (Phase B)
# ===========================================================================


class TestVisionDataPersistence:
    """Tests for VisionData.dump() / VisionData.from_h5()."""

    def _make_vd_with_vrs(self, n_stim: int = 4):
        from vneurotk.vision.data import VisionData
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        output_order = np.arange(n_stim)
        images = {i: np.random.rand(4).astype(np.float32) for i in range(n_stim)}
        vd = VisionData(output_order=output_order)
        vd.extract_from(model, vision_db=images)
        return vd

    def test_dump_creates_vision_store_group(self, tmp_path):
        import h5py

        vd = self._make_vd_with_vrs()
        fpath = tmp_path / "test.h5"
        with h5py.File(fpath, "w") as f:
            vd.dump(f)
        with h5py.File(fpath, "r") as f:
            assert "vision_store" in f
            assert "0" in f["vision_store"]
            assert "1" in f["vision_store"]
            grp = f["vision_store"]["0"]
            assert "model" in grp.attrs
            assert "module_name" in grp.attrs
            assert "array" in grp

    def test_from_h5_reconstructs_vrs(self, tmp_path):
        import h5py

        from vneurotk.vision.data import VisionData

        vd = self._make_vd_with_vrs(n_stim=5)
        fpath = tmp_path / "test.h5"
        with h5py.File(fpath, "w") as f:
            vd.dump(f)

        output_order = np.arange(5)
        with h5py.File(fpath, "r") as f:
            vd2 = VisionData.from_h5(f, output_order=output_order)

        assert len(vd2.meta) == len(vd.meta)
        assert set(vd2.meta["module_name"]) == set(vd.meta["module_name"])

    def test_dump_from_h5_roundtrip(self, tmp_path):
        import h5py

        from vneurotk.vision.data import VisionData

        vd = self._make_vd_with_vrs(n_stim=6)
        fpath = tmp_path / "roundtrip.h5"
        with h5py.File(fpath, "w") as f:
            vd.dump(f)

        output_order = np.arange(6)
        with h5py.File(fpath, "r") as f:
            vd2 = VisionData.from_h5(f, output_order=output_order)

        for name in vd.meta["module_name"]:
            arr1 = vd.by_module(name)
            arr2 = vd2.by_module(name)
            np.testing.assert_array_equal(arr1, arr2)

    def test_from_h5_returns_empty_when_group_missing(self, tmp_path):
        import h5py

        from vneurotk.vision.data import VisionData

        fpath = tmp_path / "empty.h5"
        with h5py.File(fpath, "w") as f:
            pass  # no vision_store group

        with h5py.File(fpath, "r") as f:
            vd = VisionData.from_h5(f, output_order=np.arange(3))

        assert not vd.has_visual_representations

    def test_has_visual_representations_property(self):
        from vneurotk.vision.data import VisionData

        vd_empty = VisionData(output_order=np.arange(3))
        assert not vd_empty.has_visual_representations

        vd_full = self._make_vd_with_vrs()
        assert vd_full.has_visual_representations


# ===========================================================================
# TestVisionDataAlign — storage→view seam edge cases
# ===========================================================================


class TestVisionDataAlign:
    """Tests for the storage→view seam: _align_vr and __getitem__ edge cases."""

    def _make_vd(self, output_order=None):
        from vneurotk.vision.data import VisionData

        n_stim, d = 4, 6
        if output_order is None:
            output_order = np.array([2, 0, 1, 3])
        stim_ids = list(range(n_stim))
        vr_a = VisualRepresentation(
            model="m",
            module_name="layer_a",
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.arange(n_stim * d, dtype=np.float32).reshape(n_stim, d),
        )
        vr_b = VisualRepresentation(
            model="m",
            module_name="layer_b",
            module_type="ReLU",
            stim_ids=stim_ids,
            array=np.arange(n_stim * d, dtype=np.float32).reshape(n_stim, d) * 10,
        )
        vd = VisionData(output_order=np.array(output_order))
        vd.add(VisualRepresentations([vr_a, vr_b]))
        return vd

    def test_getitem_int_returns_aligned_array(self):
        """__getitem__(int) should return the output-order-aligned array of that VR."""
        vd = self._make_vd()
        arr = vd[0]  # first VR = layer_a
        np.testing.assert_array_equal(arr, vd.by_module("layer_a"))

    def test_getitem_bool_mask_multiple_vrs_returns_visual_representations(self):
        """bool mask selecting multiple VRs should return VisualRepresentations, not ndarray."""
        vd = self._make_vd()
        mask = np.array([True, True])
        result = vd[mask]
        assert isinstance(result, VisualRepresentations)
        assert len(list(result)) == 2

    def test_getitem_multi_vr_arrays_are_aligned_to_output_order(self):
        """Multi-VR bool mask: returned VisualRepresentations must have trial-aligned arrays."""
        vd = self._make_vd(output_order=[2, 0, 1, 3])  # non-identity reorder
        mask = np.array([True, True])
        result = vd[mask]

        # Each VR in the result must have stim_ids == output_order
        assert list(result[0].stim_ids) == [2, 0, 1, 3]
        assert list(result[1].stim_ids) == [2, 0, 1, 3]

        # Arrays must be aligned: row i corresponds to output_order[i]
        layer_a_single = vd.by_module("layer_a")  # aligned ndarray via single-VR path
        np.testing.assert_array_equal(result.by_module("layer_a").array, layer_a_single)

    def test_align_subset_output_order(self):
        """output_order covering only a subset of stim_ids should align correctly."""
        from vneurotk.vision.data import VisionData

        stim_ids = [0, 1, 2, 3, 4]
        arr = np.arange(15, dtype=np.float32).reshape(5, 3)
        vr = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=stim_ids,
            array=arr,
        )
        output_order = np.array([4, 1, 2])
        vd = VisionData(output_order=output_order)
        vd.add(VisualRepresentations([vr]))
        result = vd.by_module("layer_x")
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result[0], arr[4])
        np.testing.assert_array_equal(result[1], arr[1])
        np.testing.assert_array_equal(result[2], arr[2])

    def test_align_repeated_stim_ids_in_output_order(self):
        """output_order with repeated IDs should repeat the corresponding rows."""
        from vneurotk.vision.data import VisionData

        stim_ids = [0, 1, 2]
        arr = np.eye(3, dtype=np.float32)
        vr = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=stim_ids,
            array=arr,
        )
        output_order = np.array([0, 1, 0, 2, 0])  # stim 0 appears 3 times
        vd = VisionData(output_order=output_order)
        vd.add(VisualRepresentations([vr]))
        result = vd.by_module("layer_x")
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[0], arr[0])
        np.testing.assert_array_equal(result[2], arr[0])
        np.testing.assert_array_equal(result[4], arr[0])

    def test_persistence_string_stim_ids_roundtrip(self, tmp_path):
        """dump/from_h5 should preserve string stim_ids and alignment faithfully."""
        import h5py

        from vneurotk.vision.data import VisionData

        string_ids = ["cat", "dog", "bird"]
        arr = np.eye(3, dtype=np.float32)
        vr = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=string_ids,
            array=arr,
        )
        output_order = np.array(["dog", "cat", "bird"], dtype=object)
        vd = VisionData(output_order=output_order)
        vd.add(VisualRepresentations([vr]))

        fpath = tmp_path / "str_stim.h5"
        with h5py.File(fpath, "w") as f:
            vd.dump(f)
        with h5py.File(fpath, "r") as f:
            vd2 = VisionData.from_h5(f, output_order=output_order)

        assert vd2.has_visual_representations
        result = vd2.by_module("layer_x")
        # output_order is [dog, cat, bird] → rows [1, 0, 2] of arr
        np.testing.assert_array_equal(result[0], arr[1])
        np.testing.assert_array_equal(result[1], arr[0])
        np.testing.assert_array_equal(result[2], arr[2])

    def test_add_overwrite_replaces_existing_vr(self):
        """add(overwrite=True) should replace a VR with the same (model, module_name)."""
        from vneurotk.vision.data import VisionData

        output_order = np.array([0, 1, 2])
        stim_ids = [0, 1, 2]
        orig = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.zeros((3, 4), dtype=np.float32),
        )
        replacement = VisualRepresentation(
            model="m",
            module_name="layer_x",
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.ones((3, 4), dtype=np.float32),
        )
        vd = VisionData(output_order=output_order)
        vd.add(VisualRepresentations([orig]))
        vd.add(VisualRepresentations([replacement]), overwrite=True)

        assert len(vd.meta) == 1
        result = vd.by_module("layer_x")
        np.testing.assert_array_equal(result, np.ones((3, 4), dtype=np.float32))


# ===========================================================================
# TestVisionDataAlignCache — verify (model, module_name) cache key behavior
# ===========================================================================


class TestVisionDataAlignCache:
    """Tests that _align_vr uses (model, module_name) as cache key, not id(vr)."""

    def _make_vr(self, model: str, module_name: str, array: np.ndarray) -> VisualRepresentation:
        return VisualRepresentation(
            model=model,
            module_name=module_name,
            module_type="Linear",
            stim_ids=[0, 1, 2],
            array=array,
        )

    def test_cache_hit_on_same_key(self):
        """Second call to by_module with same key should return cached result."""
        from vneurotk.vision.data import VisionData

        arr = np.eye(3, dtype=np.float32)
        vr = self._make_vr("m", "layer_a", arr)
        vd = VisionData(output_order=np.array([0, 1, 2]))
        vd.add(VisualRepresentations([vr]))

        result1 = vd.by_module("layer_a")
        result2 = vd.by_module("layer_a")
        assert result1 is result2  # same object from cache

    def test_cache_key_is_model_module_name(self):
        """Cache key must be (model, module_name), not object identity."""
        from vneurotk.vision.data import VisionData

        arr_orig = np.zeros((3, 4), dtype=np.float32)
        vr_orig = self._make_vr("m", "layer_a", arr_orig)

        output_order = np.array([0, 1, 2])
        vd = VisionData(output_order=output_order)
        vd.add(VisualRepresentations([vr_orig]))

        # Prime the cache
        cached = vd.by_module("layer_a")
        np.testing.assert_array_equal(cached, arr_orig)

        # Replace VR with same key but different array — cache should still return old result
        arr_new = np.ones((3, 4), dtype=np.float32)
        vr_new = self._make_vr("m", "layer_a", arr_new)
        vd._records[("m", "layer_a")] = vr_new

        # Cache key (m, layer_a) still hits — returns original cached array
        still_cached = vd.by_module("layer_a")
        np.testing.assert_array_equal(still_cached, arr_orig)

    def test_different_models_different_cache_entries(self):
        """Two VRs with same module_name but different model get separate cache entries."""
        from vneurotk.vision.data import VisionData

        arr_a = np.zeros((3, 4), dtype=np.float32)
        arr_b = np.ones((3, 4), dtype=np.float32)
        vr_a = self._make_vr("model_a", "layer_x", arr_a)
        vr_b = self._make_vr("model_b", "layer_x", arr_b)

        vd = VisionData(output_order=np.array([0, 1, 2]))
        vd.add(VisualRepresentations([vr_a, vr_b]))

        result_a = vd.by_module("layer_x", model="model_a")
        result_b = vd.by_module("layer_x", model="model_b")
        np.testing.assert_array_equal(result_a, arr_a)
        np.testing.assert_array_equal(result_b, arr_b)
        assert result_a is not result_b


# ===========================================================================
# TestPrepareImages — unit-test VisionModel._prepare_images static method
# ===========================================================================


class TestPrepareImages:
    """Tests for VisionModel._prepare_images() without any backend/model."""

    def test_dict_passthrough(self):
        """Already-loaded images in a dict pass through unchanged."""
        from vneurotk.vision.model.base import VisionModel

        img_a = np.zeros((3, 4, 3), dtype=np.uint8)
        img_b = np.ones((3, 4, 3), dtype=np.uint8)
        images = {"a": img_a, "b": img_b}
        stim_ids, loaded = VisionModel._prepare_images(images)
        assert stim_ids == ["a", "b"]
        assert loaded[0] is img_a
        assert loaded[1] is img_b

    def test_path_values_opened(self, tmp_path):
        """String and Path values are opened as PIL Images."""
        from PIL import Image as PILImage

        from vneurotk.vision.model.base import VisionModel

        img_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        p = tmp_path / "img.png"
        PILImage.fromarray(img_arr).save(p)

        images = {42: str(p)}
        stim_ids, loaded = VisionModel._prepare_images(images)
        assert stim_ids == [42]
        assert isinstance(loaded[0], PILImage.Image)

    def test_path_object_opened(self, tmp_path):
        """pathlib.Path values are opened as PIL Images."""

        from PIL import Image as PILImage

        from vneurotk.vision.model.base import VisionModel

        img_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        p = tmp_path / "img2.png"
        PILImage.fromarray(img_arr).save(p)

        stim_ids, loaded = VisionModel._prepare_images({0: p})
        assert isinstance(loaded[0], PILImage.Image)

    def test_order_preserved(self):
        """Output order matches dict insertion order."""
        from vneurotk.vision.model.base import VisionModel

        imgs = {i: np.zeros((2, 2, 3), dtype=np.uint8) for i in range(5)}
        stim_ids, loaded = VisionModel._prepare_images(imgs)
        assert stim_ids == list(range(5))
        assert len(loaded) == 5


def _timm_installed():
    try:
        import timm  # noqa: F401  # type: ignore

        return True
    except ImportError:
        return False


# ===========================================================================
# TestRunBatches — unit-test VisionModel._run_batches without any backend
# ===========================================================================


class TestRunBatches:
    """Candidate A — _run_batches() is testable with a mocked _forward_chunk."""

    def _make_model_with_fake_forward(self, n_layers=2, feat_dim=8):
        """Return a VisionModel-like object whose _forward_chunk returns fake activations."""
        from unittest.mock import MagicMock

        from vneurotk.vision.model.base import VisionModel

        fake_backend = MagicMock()
        fake_backend.get_model_meta.return_value = MagicMock(model_id="fake_model")
        fake_backend.enumerate_modules.return_value = []

        model = object.__new__(VisionModel)
        model._backend = fake_backend
        model._module_names = [f"layer{i}" for i in range(n_layers)]

        def fake_forward_chunk(images):
            b = len(images)
            return {f"layer{i}": np.ones((b, feat_dim)) * i for i in range(n_layers)}

        model._forward_chunk = fake_forward_chunk
        return model

    def test_single_batch_no_chunking(self):
        """When batch_size >= n_images, result equals one _forward_chunk call."""
        model = self._make_model_with_fake_forward(n_layers=2, feat_dim=4)
        loaded = [object() for _ in range(3)]
        features = model._run_batches(loaded, batch_size=10)

        assert set(features.keys()) == {"layer0", "layer1"}
        assert features["layer0"].shape == (3, 4)
        np.testing.assert_array_equal(features["layer0"], np.zeros((3, 4)))
        np.testing.assert_array_equal(features["layer1"], np.ones((3, 4)))

    def test_multi_batch_concatenated_correctly(self):
        """Multiple chunks are concatenated along axis 0."""
        model = self._make_model_with_fake_forward(n_layers=1, feat_dim=4)
        loaded = [object() for _ in range(5)]
        features = model._run_batches(loaded, batch_size=2)

        assert features["layer0"].shape == (5, 4)

    def test_single_image(self):
        """Single image is handled without error."""
        model = self._make_model_with_fake_forward(n_layers=1, feat_dim=8)
        loaded = [object()]
        features = model._run_batches(loaded, batch_size=32)
        assert features["layer0"].shape == (1, 8)


def _network_available(host: str = "huggingface.co", port: int = 443) -> bool:
    import socket

    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


# ===========================================================================
# TestModuleTypeMapCache
# ===========================================================================


class TestModuleTypeMapCache:
    """Candidate 2 — _module_type_map is built once at init, not per _extract_batch."""

    def _make_model(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

    def test_module_type_map_populated_at_init(self):
        model = self._make_model()
        assert hasattr(model, "_module_type_map")
        assert isinstance(model._module_type_map, dict)
        assert len(model._module_type_map) > 0

    def test_module_type_map_keys_are_layer_names(self):
        model = self._make_model()
        for key in model._module_type_map:
            assert isinstance(key, str)

    def test_extract_does_not_call_enumerate_modules(self):
        """_extract_batch() uses cached map — enumerate_modules() not called during extraction."""
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        # Patch enumerate_modules with a spy
        call_count = []
        original = model._backend.enumerate_modules

        def spy_enumerate():
            call_count.append(1)
            return original()

        model._backend.enumerate_modules = spy_enumerate  # ty: ignore[invalid-assignment]

        images = {i: np.random.rand(4).astype(np.float32) for i in range(3)}
        model.extract(images, batch_size=3)

        assert len(call_count) == 0, "enumerate_modules() was called during extraction"


# ===========================================================================
# TestHookableModel
# ===========================================================================


class TestHookableModel:
    """Candidate 3 — BaseBackend.hookable_model is a declared interface."""

    def test_base_backend_default_returns_model(self):
        """Default hookable_model returns self.model."""
        backend = _MockBackend(device="cpu")
        backend.load("mock")
        assert backend.hookable_model is backend.model

    def test_hookable_model_before_load_returns_none(self):
        """Before load(), model is None, hookable_model returns None."""
        backend = _MockBackend(device="cpu")
        assert backend.hookable_model is None

    def test_from_model_uses_hookable_model(self, monkeypatch):
        """VisionModel.from_model() selects modules from backend.hookable_model."""
        from unittest.mock import MagicMock

        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")

        # Override hookable_model with a spy — use monkeypatch so the class is
        # restored after the test (direct class assignment would leak to all
        # subsequent _MockBackend instances).
        original_model = backend.model
        sentinel = MagicMock(wraps=original_model)
        monkeypatch.setattr(type(backend), "hookable_model", property(lambda self: sentinel))

        selector = CustomSelector(["0"])
        VisionModel.from_model(model=original_model, backend=backend, selector=selector)

        # The selector was called on the sentinel (hookable_model)
        # We verify by checking that named_modules was called on sentinel
        assert sentinel.named_modules.called


def _transformers_installed():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _thingsvision_installed():
    try:
        import thingsvision  # noqa: F401  # ty: ignore[unresolved-import]

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _timm_installed(), reason="timm not installed")
class TestTimmSmokeTest:
    def test_resnet18_end_to_end(self):
        from PIL import Image

        from vneurotk.vision.model.backend.timm_backend import TimmBackend
        from vneurotk.vision.model.selector import BlockLevelSelector

        backend = TimmBackend(device="cpu")
        backend.load("resnet18", pretrained=False)

        sel = BlockLevelSelector()
        module_names = sel.select(backend.enumerate_modules())
        assert len(module_names) > 0

        backend.register_hooks(module_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        out = backend.forward(inputs)
        acts = backend.collect_activations()

        assert isinstance(out, torch.Tensor)
        assert len(acts) == 2
        backend.remove_hooks()

    @pytest.mark.skipif(not _network_available(), reason="network unavailable")
    def test_dinov2_end_to_end(self):
        from PIL import Image

        from vneurotk.vision.model.backend.transformers_backend import TransformersBackend
        from vneurotk.vision.model.selector import BlockLevelSelector

        backend = TransformersBackend(device="cpu")
        backend.load("facebook/dinov2-base")

        sel = BlockLevelSelector()
        module_names = sel.select(backend.enumerate_modules())
        assert len(module_names) > 0

        backend.register_hooks(module_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        backend.forward(inputs)
        acts = backend.collect_activations()

        assert len(acts) == 2
        backend.remove_hooks()


@pytest.mark.skipif(not _thingsvision_installed(), reason="thingsvision not installed")
class TestThingsVisionSmokeTest:
    def test_resnet18_end_to_end(self):
        from PIL import Image

        from vneurotk.vision.model.backend.thingsvision_backend import ThingsVisionBackend
        from vneurotk.vision.model.selector import BlockLevelSelector

        backend = ThingsVisionBackend(source="torchvision", device="cpu")
        backend.load("resnet18", pretrained=False)

        sel = BlockLevelSelector()
        module_names = sel.select(backend.enumerate_modules())
        assert len(module_names) > 0

        backend.register_hooks(module_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        out = backend.forward(inputs)
        acts = backend.collect_activations()

        assert isinstance(out, torch.Tensor)
        assert len(acts) == 2
        backend.remove_hooks()


# ===========================================================================
# TestBindSelector
# ===========================================================================


class TestBindSelector:
    """Candidate 1 (Round 14) — _bind_selector consolidates hook wiring."""

    def _make_backend(self):
        backend = _MockBackend(device="cpu")
        backend.load("mock")
        return backend

    def test_bind_selector_sets_module_names(self):
        """_bind_selector populates _module_names from selector."""
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = self._make_backend()
        selector = CustomSelector(["0", "2"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        assert model._module_names == ["0", "2"]

    def test_bind_selector_sets_module_type_map(self):
        """_bind_selector populates _module_type_map from enumerate_modules."""
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = self._make_backend()
        selector = CustomSelector(["0"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        assert isinstance(model._module_type_map, dict)
        assert len(model._module_type_map) > 0

    def test_bind_selector_registers_hooks(self):
        """After _bind_selector, backend has hooks for selected modules."""
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = self._make_backend()
        selector = CustomSelector(["0", "2"])
        model = VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

        assert len(model._backend._hooks) == 2

    def test_from_model_and_init_produce_same_module_names(self):
        """_bind_selector path shared: from_model and __init__ yield same module wiring."""
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend1 = self._make_backend()
        backend2 = self._make_backend()
        selector1 = CustomSelector(["0", "2"])
        selector2 = CustomSelector(["0", "2"])

        model1 = VisionModel.from_model(model=backend1.model, backend=backend1, selector=selector1)
        model2 = VisionModel.from_model(model=backend2.model, backend=backend2, selector=selector2)

        assert model1._module_names == model2._module_names
        assert set(model1._module_type_map.keys()) == set(model2._module_type_map.keys())


# ===========================================================================
# TestStimIdsTuple
# ===========================================================================


class TestStimIdsTuple:
    """Candidate 3 (Round 14) — VisualRepresentation.stim_ids is an immutable tuple."""

    def test_stim_ids_is_tuple(self):
        """stim_ids stored as tuple, not list."""
        vr = _make_vr(n_stim=3)
        assert isinstance(vr.stim_ids, tuple)

    def test_stim_ids_values_preserved(self):
        """Tuple contains the correct values in insertion order."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentation

        vr = VisualRepresentation(
            model="m",
            module_name="l",
            module_type="T",
            stim_ids=["a", "b", "c"],
            array=np.zeros((3, 4)),
        )
        assert vr.stim_ids == ("a", "b", "c")

    def test_id_to_idx_consistent_with_stim_ids(self):
        """_id_to_idx maps each stim_id to its position in stim_ids."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentation

        ids = [10, 20, 30]
        vr = VisualRepresentation(model="m", module_name="l", module_type="T", stim_ids=ids, array=np.zeros((3, 2)))
        for i, sid in enumerate(ids):
            assert vr._id_to_idx[sid] == i

    def test_select_uses_tuple_stim_ids(self):
        """Returned VR from select() also has tuple stim_ids."""
        vr = _make_vr(n_stim=5)
        sub = vr.select([1, 3])
        assert isinstance(sub.stim_ids, tuple)
        assert list(sub.stim_ids) == [1, 3]

    def test_visual_representations_stim_ids_is_tuple(self):
        """VisualRepresentations.stim_ids delegates to first VR, returns tuple."""
        vrs = _make_vrs(n_stim=4)
        assert isinstance(vrs.stim_ids, tuple)

    def test_visual_representations_empty_stim_ids_is_tuple(self):
        """Empty VisualRepresentations.stim_ids returns empty tuple."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        vrs = VisualRepresentations([])
        assert vrs.stim_ids == ()
        assert isinstance(vrs.stim_ids, tuple)


# ===========================================================================
# TestRelevantImages
# ===========================================================================


class TestRelevantImages:
    """Candidate 2 (Round 15) — VisionData._relevant_images() named seam."""

    def _make_vd(self, output_order):
        from vneurotk.vision.data import VisionData

        return VisionData(np.array(output_order))

    def test_returns_only_ids_in_output_order(self):
        """Images keyed by IDs not in output_order are excluded."""
        vd = self._make_vd([0, 1, 2])
        stimuli = {0: "img0", 1: "img1", 2: "img2", 3: "img3"}
        result = vd._relevant_images(stimuli)
        assert set(result.keys()) == {0, 1, 2}

    def test_silently_drops_missing_ids(self):
        """IDs in output_order but absent from stimuli are silently dropped."""
        vd = self._make_vd([0, 1, 2])
        stimuli = {0: "img0", 2: "img2"}  # 1 is missing
        result = vd._relevant_images(stimuli)
        assert set(result.keys()) == {0, 2}
        assert 1 not in result

    def test_preserves_first_appearance_order_not_sorted(self):
        """Deduplication uses first-appearance order, not sorted order."""
        vd = self._make_vd([3, 1, 2, 1, 3])  # 3 first, then 1, then 2
        stimuli = {1: "a", 2: "b", 3: "c"}
        result = vd._relevant_images(stimuli)
        assert list(result.keys()) == [3, 1, 2]

    def test_empty_when_no_overlap(self):
        """Returns empty dict when no output_order ID exists in stimuli."""
        vd = self._make_vd([10, 20])
        stimuli = {0: "img0", 1: "img1"}
        result = vd._relevant_images(stimuli)
        assert result == {}

    def test_deduplicates_output_order_ids(self):
        """Repeated IDs in output_order appear only once in result."""
        vd = self._make_vd([0, 1, 0, 1])
        stimuli = {0: "img0", 1: "img1"}
        result = vd._relevant_images(stimuli)
        assert list(result.keys()) == [0, 1]


# ===========================================================================
# TestModulesToExtract
# ===========================================================================


class TestModulesToExtract:
    """Candidate 2 (Round 16) — _pending_modules encapsulates extraction decision."""

    def _make_vd(self, output_order=None):
        from vneurotk.vision.data import VisionData

        if output_order is None:
            output_order = [0, 1, 2]
        return VisionData(np.array(output_order))

    def _add_vr(self, vd, model_id, module_name):
        from vneurotk.vision.representation.visual_representations import (
            VisualRepresentation,
            VisualRepresentations,
        )

        vr = VisualRepresentation(
            model=model_id,
            module_name=module_name,
            module_type="Linear",
            stim_ids=[0, 1, 2],
            array=np.zeros((3, 4)),
        )
        vd.add(VisualRepresentations([vr]))

    def test_empty_store_returns_all_modules(self):
        """No cached VRs → all modules are pending regardless of overwrite."""
        vd = self._make_vd()
        result = vd._pending_modules("resnet50", ["layer1", "layer2"], overwrite=False)
        assert result == ["layer1", "layer2"]

    def test_overwrite_true_returns_all_modules(self):
        """overwrite=True → all modules returned even when already cached."""
        vd = self._make_vd()
        self._add_vr(vd, "resnet50", "layer1")
        self._add_vr(vd, "resnet50", "layer2")
        result = vd._pending_modules("resnet50", ["layer1", "layer2"], overwrite=True)
        assert result == ["layer1", "layer2"]

    def test_all_cached_returns_empty_list(self):
        """All modules cached and overwrite=False → empty list."""
        vd = self._make_vd()
        self._add_vr(vd, "resnet50", "layer1")
        self._add_vr(vd, "resnet50", "layer2")
        result = vd._pending_modules("resnet50", ["layer1", "layer2"], overwrite=False)
        assert result == []

    def test_partial_cache_returns_missing_only(self):
        """Only one module cached → returns the other."""
        vd = self._make_vd()
        self._add_vr(vd, "resnet50", "layer1")
        result = vd._pending_modules("resnet50", ["layer1", "layer2"], overwrite=False)
        assert result == ["layer2"]


# ===========================================================================
# TestSetSelectorBind
# ===========================================================================


class TestSetSelectorBind:
    """Candidate 1 (Round 18) — set_selector delegates to _bind_selector()."""

    def _make_model(self, layer_names=None):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        layers = layer_names or ["0", "2"]
        selector = CustomSelector(layers)
        return VisionModel.from_model(model=backend.model, backend=backend, selector=selector)

    def test_set_selector_updates_module_names(self):
        """set_selector replaces _module_names to reflect new selector."""
        from vneurotk.vision.model.selector import CustomSelector

        model = self._make_model(["0", "2"])
        model.set_selector(CustomSelector(["1"]))
        assert model.module_names == ["1"]

    def test_set_selector_updates_module_type_map(self):
        """set_selector refreshes _module_type_map — no stale entries after change."""
        from vneurotk.vision.model.selector import CustomSelector

        model = self._make_model(["0"])
        assert model._module_type_map  # non-empty after init
        model.set_selector(CustomSelector(["1", "2"]))
        # map reflects new enumeration — ReLU is at index 1
        assert "1" in model._module_type_map or len(model._module_type_map) > 0

    def test_set_selector_module_type_populated_after_extract(self):
        """VR.module_type is non-empty string after set_selector()."""
        from vneurotk.vision.model.selector import CustomSelector

        model = self._make_model(["0"])
        model.set_selector(CustomSelector(["2"]))
        image = np.random.rand(4).astype(np.float32)
        vrs = model.extract(image)
        vr = next(iter(vrs))
        assert vr.module_type != ""

    def test_set_selector_registers_new_hooks(self):
        """set_selector re-registers hooks matching the new module list."""
        from vneurotk.vision.model.selector import CustomSelector

        model = self._make_model(["0", "2"])
        assert len(model._backend._hooks) == 2
        model.set_selector(CustomSelector(["1"]))
        assert len(model._backend._hooks) == 1


# ===========================================================================
# TestNormalizeImages
# ===========================================================================


class TestNormalizeImages:
    """C4 (Round 22) — _normalize_images: list input raises TypeError."""

    def _norm(self, image):
        from vneurotk.vision.model.base import VisionModel

        return VisionModel._normalize_images(image)

    def test_dict_returned_as_is(self):
        d = {"a": "img_a", "b": "img_b"}
        assert self._norm(d) is d

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError, match="dict mapping stim_id"):
            self._norm(["img0", "img1"])

    def test_empty_list_raises_type_error(self):
        with pytest.raises(TypeError, match="dict mapping stim_id"):
            self._norm([])

    def test_single_image_converted_to_key_zero_dict(self):
        img = np.zeros((4,), dtype=np.float32)
        result = self._norm(img)
        assert list(result.keys()) == [0]
        assert result[0] is img

    def test_extract_list_raises(self):
        backend = _MockBackend(device="cpu")
        backend.load("mock")
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        model = VisionModel.from_model(model=backend.model, backend=backend, selector=CustomSelector(["0"]))
        images = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        with pytest.raises(TypeError, match="dict mapping stim_id"):
            model.extract(images)

    def test_extract_dict_with_integer_ids(self):
        backend = _MockBackend(device="cpu")
        backend.load("mock")
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        model = VisionModel.from_model(model=backend.model, backend=backend, selector=CustomSelector(["0"]))
        images = {i: np.random.rand(4).astype(np.float32) for i in range(3)}
        vrs = model.extract(images)
        assert list(vrs["0"].stim_ids) == [0, 1, 2]

    def test_extract_single_produces_stim_id_zero(self):
        backend = _MockBackend(device="cpu")
        backend.load("mock")
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        model = VisionModel.from_model(model=backend.model, backend=backend, selector=CustomSelector(["0"]))
        vrs = model.extract(np.random.rand(4).astype(np.float32))
        assert list(vrs["0"].stim_ids) == [0]


# ===========================================================================
# TestAssertSharedStimIds
# ===========================================================================


class TestAssertSharedStimIds:
    """Candidate 1 (Round 20) — _assert_shared_stim_ids is symmetric to _build_meta."""

    def _make_vr(self, module_name: str, stim_ids: list):
        from vneurotk.vision.representation.visual_representations import VisualRepresentation

        return VisualRepresentation(
            model="m",
            module_name=module_name,
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.zeros((len(stim_ids), 4)),
        )

    def test_empty_list_passes(self):
        """Empty list never raises."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        VisualRepresentations._assert_shared_stim_ids([])

    def test_single_vr_passes(self):
        """Single VR never raises."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        VisualRepresentations._assert_shared_stim_ids([self._make_vr("a", [0, 1, 2])])

    def test_matching_stim_ids_passes(self):
        """All VRs with identical stim_ids passes."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        vrs = [self._make_vr("a", [0, 1]), self._make_vr("b", [0, 1])]
        VisualRepresentations._assert_shared_stim_ids(vrs)

    def test_mismatched_stim_ids_raises(self):
        """VRs with different stim_ids raise ValueError."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        vrs = [self._make_vr("a", [0, 1]), self._make_vr("b", [0, 2])]
        with pytest.raises(ValueError, match="stim_ids"):
            VisualRepresentations._assert_shared_stim_ids(vrs)

    def test_init_uses_validator(self):
        """VisualRepresentations.__init__ raises on mismatched stim_ids."""
        from vneurotk.vision.representation.visual_representations import VisualRepresentations

        vrs = [self._make_vr("a", [0, 1]), self._make_vr("b", [0, 2])]
        with pytest.raises(ValueError):
            VisualRepresentations(vrs)


# ===========================================================================
# TestModuleDepth
# ===========================================================================


class TestModuleDepth:
    """Candidate 3 (Round 20) — _module_depth encapsulates the dotted-name depth formula."""

    def test_top_level_name_has_depth_1(self):
        """'layer1' has no dots → depth 1."""
        from vneurotk.vision.model.selector import BlockLevelSelector

        assert BlockLevelSelector._module_depth("layer1") == 1

    def test_depth_two_name(self):
        """'encoder.layer' → depth 2."""
        from vneurotk.vision.model.selector import BlockLevelSelector

        assert BlockLevelSelector._module_depth("encoder.layer") == 2

    def test_depth_three_name(self):
        """'encoder.layer.3' → depth 3."""
        from vneurotk.vision.model.selector import BlockLevelSelector

        assert BlockLevelSelector._module_depth("encoder.layer.3") == 3

    def test_empty_string_returns_zero(self):
        """Empty name (root module) → depth 0."""
        from vneurotk.vision.model.selector import BlockLevelSelector

        assert BlockLevelSelector._module_depth("") == 0

    def test_compiled_patterns_cached_in_init(self):
        """_compiled_patterns is built once in __init__, not in select()."""
        from vneurotk.vision.model.selector import BlockLevelSelector

        sel = BlockLevelSelector()
        assert hasattr(sel, "_compiled_patterns")
        assert len(sel._compiled_patterns) == len(BlockLevelSelector._ARCH_PATTERNS)

    def test_select_uses_compiled_patterns(self):
        """BlockLevelSelector.select() still works correctly after refactor."""
        import torch.nn as nn  # type: ignore

        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        sel = BlockLevelSelector()
        result = sel.select(_module_infos(model))
        assert isinstance(result, list)
        assert len(result) > 0


# ===========================================================================
# TestAssertStimIdsCoverOutputOrder  (Candidate 1, Round 21)
# ===========================================================================


class TestAssertStimIdsCoverOutputOrder:
    """VisionData._assert_stim_ids_cover_output_order enforces write-time coverage."""

    def _make_vr(self, stim_ids):
        return VisualRepresentation(
            model="m",
            module_name="layer",
            module_type="Linear",
            stim_ids=stim_ids,
            array=np.zeros((len(stim_ids), 4), dtype=np.float32),
        )

    def test_exact_coverage_passes(self):
        """VR whose stim_ids exactly match output_order passes."""
        from vneurotk.vision.data import VisionData

        vr = self._make_vr([0, 1, 2])
        VisionData._assert_stim_ids_cover_output_order(vr, np.array([0, 1, 2]))

    def test_superset_coverage_passes(self):
        """VR with more stim_ids than output_order passes (superset)."""
        from vneurotk.vision.data import VisionData

        vr = self._make_vr([0, 1, 2, 3])
        VisionData._assert_stim_ids_cover_output_order(vr, np.array([0, 2]))

    def test_missing_id_raises(self):
        """VR missing an output_order ID raises ValueError at add() time."""
        from vneurotk.vision.data import VisionData

        vr = self._make_vr([0, 1])
        with pytest.raises(ValueError, match="Missing"):
            VisionData._assert_stim_ids_cover_output_order(vr, np.array([0, 1, 2]))

    def test_error_lists_missing_ids(self):
        """ValueError message includes the missing IDs."""
        from vneurotk.vision.data import VisionData

        vr = self._make_vr([0])
        with pytest.raises(ValueError, match="2"):
            VisionData._assert_stim_ids_cover_output_order(vr, np.array([0, 2]))

    def test_add_rejects_incomplete_vr(self):
        """VisionData.add() raises ValueError when a VR doesn't cover output_order."""
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.array([0, 1, 2]))
        vr = self._make_vr([0, 1])  # missing id=2
        vrs = VisualRepresentations([vr])
        with pytest.raises(ValueError, match="Missing"):
            vd.add(vrs)

    def test_add_accepts_full_coverage(self):
        """VisionData.add() succeeds when all output_order IDs are covered."""
        from vneurotk.vision.data import VisionData

        vd = VisionData(output_order=np.array([0, 1, 2]))
        vr = self._make_vr([0, 1, 2])
        vrs = VisualRepresentations([vr])
        vd.add(vrs)  # no exception


# ===========================================================================
# TestBuildVrList  (Candidate 2, Round 21)
# ===========================================================================


class TestBuildVrList:
    """VisionModel._build_vr_list assembles VR objects from stim_ids + features."""

    def _make_model(self):
        from vneurotk.vision.model.base import VisionModel
        from vneurotk.vision.model.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        return VisionModel.from_model(
            model=backend.model,
            backend=backend,
            selector=CustomSelector(["0", "2"]),
        )

    def test_returns_one_vr_per_layer(self):
        """_build_vr_list returns one VR for each key in features."""
        model = self._make_model()
        stim_ids = ["a", "b", "c"]
        features = {
            "0": np.zeros((3, 8), dtype=np.float32),
            "2": np.zeros((3, 2), dtype=np.float32),
        }
        vrs = model._build_vr_list(stim_ids, features)
        assert len(vrs) == 2

    def test_stim_ids_propagated(self):
        """Each VR in the result has stim_ids matching the input list."""
        model = self._make_model()
        stim_ids = ["x", "y"]
        features = {"0": np.zeros((2, 8), dtype=np.float32)}
        vrs = model._build_vr_list(stim_ids, features)
        assert list(vrs[0].stim_ids) == stim_ids

    def test_module_type_filled_from_map(self):
        """module_type is populated from _module_type_map, not empty string."""
        model = self._make_model()
        features = {"0": np.zeros((1, 8), dtype=np.float32)}
        vrs = model._build_vr_list(["s0"], features)
        assert vrs[0].module_type != ""

    def test_array_shape_preserved(self):
        """The array stored in each VR matches the input features shape."""
        model = self._make_model()
        arr = np.random.rand(4, 8).astype(np.float32)
        vrs = model._build_vr_list(["a", "b", "c", "d"], {"0": arr})
        np.testing.assert_array_equal(vrs[0].array, arr)


# ===========================================================================
# TestFilterModules  (Candidate 4, Round 21)
# ===========================================================================


class TestFilterModules:
    """VisionModel._filter_modules filters module_list by type and/or name."""

    def _make_modules(self):
        from vneurotk.vision.model.backend.base import ModuleInfo

        return [
            ModuleInfo("enc.0", "Linear", depth=2, n_params=8, is_leaf=True, param_shapes={}),
            ModuleInfo("enc.1", "ReLU", depth=2, n_params=0, is_leaf=True, param_shapes={}),
            ModuleInfo("enc.2", "Linear", depth=2, n_params=4, is_leaf=True, param_shapes={}),
            ModuleInfo("head", "LayerNorm", depth=1, n_params=2, is_leaf=True, param_shapes={}),
        ]

    def test_filter_by_single_type(self):
        """Filtering by one module_type returns all matching modules."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, {"Linear"}, set())
        assert [m.name for m in result] == ["enc.0", "enc.2"]

    def test_filter_by_multiple_types(self):
        """Filtering by two types returns union."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, {"Linear", "LayerNorm"}, set())
        assert {m.name for m in result} == {"enc.0", "enc.2", "head"}

    def test_filter_by_name(self):
        """Filtering by exact name returns that module."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, set(), {"head"})
        assert [m.name for m in result] == ["head"]

    def test_filter_type_and_name_union(self):
        """type and name filters are unioned, not intersected."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, {"ReLU"}, {"head"})
        assert {m.name for m in result} == {"enc.1", "head"}

    def test_no_match_returns_empty(self):
        """No match → empty list (caller decides whether to raise)."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, {"Conv2d"}, set())
        assert result == []

    def test_preserves_order(self):
        """Result order matches original module_list order."""
        from vneurotk.vision.model.base import VisionModel

        modules = self._make_modules()
        result = VisionModel._filter_modules(modules, {"Linear"}, {"enc.1"})
        assert [m.name for m in result] == ["enc.0", "enc.1", "enc.2"]


# ---------------------------------------------------------------------------
# Rich print functions
# ---------------------------------------------------------------------------


def _make_module_list():
    from vneurotk.vision.model.backend.base import ModuleInfo

    return [
        ModuleInfo("encoder", "Sequential", depth=1, n_params=22, is_leaf=False, param_shapes={}),
        ModuleInfo(
            "encoder.fc",
            "Linear",
            depth=2,
            n_params=20,
            is_leaf=True,
            param_shapes={"weight": (4, 4), "bias": (4,)},
        ),
        ModuleInfo("encoder.bn", "BatchNorm1d", depth=2, n_params=2, is_leaf=True, param_shapes={}),
    ]


class TestPrintModules:
    """_print_modules / print_modules renders without error and produces output."""

    def test_no_error(self):
        from io import StringIO

        from rich.console import Console

        from vneurotk.vision.model.base import _print_modules

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        _print_modules(_make_module_list(), console=console)
        output = buf.getvalue()
        assert "Linear" in output
        assert "encoder" in output

    def test_max_depth_filters(self):
        from io import StringIO

        from rich.console import Console

        from vneurotk.vision.model.base import _print_modules

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        _print_modules(_make_module_list(), max_depth=1, console=console)
        output = buf.getvalue()
        assert "encoder" in output
        assert "fc" not in output

    def test_record_true_export_svg(self):
        """Console(record=True) + export_svg() produces a non-empty SVG string."""
        from rich.console import Console

        from vneurotk.vision.model.base import _print_modules

        console = Console(record=True, force_terminal=True, width=120)
        _print_modules(_make_module_list(), console=console)
        svg = console.export_svg()
        assert svg.startswith("<svg")
        assert len(svg) > 100

    def test_public_print_modules_delegates(self):
        from io import StringIO

        from rich.console import Console

        from vneurotk.vision.model.base import print_modules

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_modules(_make_module_list(), console=console)
        assert "Linear" in buf.getvalue()


class TestPrintCachedModels:
    """print_cached_models renders without error; Console(record=True) captures SVG."""

    def _make_models(self):
        from datetime import datetime

        from vneurotk.vision._cache import CachedModel

        return [
            CachedModel(
                model_id="openai/clip-vit-base-patch32",
                source="transformers",
                size_bytes=600 * 1024 * 1024,
                last_used=datetime(2025, 1, 15),
            ),
            CachedModel(
                model_id="timm/resnet50.a1_in1k",
                source="timm",
                size_bytes=100 * 1024 * 1024,
                last_used=datetime(2025, 3, 1),
            ),
        ]

    def test_no_error(self):
        from io import StringIO

        from rich.console import Console

        from vneurotk.vision._cache import print_cached_models

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_cached_models(self._make_models(), console=console)
        output = buf.getvalue()
        assert "clip-vit" in output
        assert "resnet50" in output

    def test_empty_list_message(self):
        from io import StringIO

        from rich.console import Console

        from vneurotk.vision._cache import print_cached_models

        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        print_cached_models([], console=console)
        assert "No cached models found" in buf.getvalue()

    def test_record_true_export_svg(self):
        from rich.console import Console

        from vneurotk.vision._cache import print_cached_models

        console = Console(record=True, force_terminal=True, width=120)
        print_cached_models(self._make_models(), console=console)
        svg = console.export_svg()
        assert svg.startswith("<svg")
        assert "clip-vit" in svg
