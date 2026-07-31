"""HuggingFace Transformers vision model backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import torch

from vneurotk.vision.meta import ModelInfo
from vneurotk.vision.model.backend.base import BaseBackend

__all__ = ["TransformersBackend"]


class TransformersBackend(BaseBackend):
    """Backend powered by HuggingFace ``transformers``.

    Supports any vision model loadable via ``AutoModel.from_pretrained()``.
    CLIP and SigLIP models are detected by name and loaded with the
    appropriate model class; all hook management uses :attr:`hookable_model`.

    Parameters
    ----------
    device : str or torch.device
        Inference device (default ``"cpu"``).
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(device)
        self._model_name: str = ""
        self._processor = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a HuggingFace vision model.

        Parameters
        ----------
        model_name : str
            HuggingFace model ID, e.g. ``"openai/clip-vit-base-patch32"``
            or ``"facebook/dinov2-base"``.
        pretrained : bool
            If ``False``, load with randomized weights (for testing).
        """
        import torch

        try:
            from transformers import (
                AutoConfig,
                AutoModel,
                AutoProcessor,
                CLIPProcessor,
                CLIPVisionModelWithProjection,
                SiglipVisionModel,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TransformersBackend. Install it with: uv add 'vneurotk[vision]'"
            ) from exc

        logger.info("Loading transformers model: {} (pretrained={})", model_name, pretrained)

        n = model_name.lower()
        is_clip = "clip" in n
        is_siglip = "siglip" in n

        if is_clip:
            self._processor = CLIPProcessor.from_pretrained(model_name)
            if pretrained:
                self.model = CLIPVisionModelWithProjection.from_pretrained(model_name, use_safetensors=True)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.model = CLIPVisionModelWithProjection(getattr(config, "vision_config", config))
        elif is_siglip:
            self._processor = AutoProcessor.from_pretrained(model_name)
            if pretrained:
                self.model = SiglipVisionModel.from_pretrained(model_name, use_safetensors=True)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.model = SiglipVisionModel(getattr(config, "vision_config", config))
        else:
            self._processor = AutoProcessor.from_pretrained(model_name)
            if pretrained:
                self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.model = AutoModel.from_config(config)

        model: Any = self.model
        model.eval()
        try:
            model.to(self.device)
        except (ValueError, RuntimeError) as e:
            if self.device.type != "cpu":
                logger.warning("Moving model to {} failed ({}), falling back to CPU", self.device, e)
                self.device = torch.device("cpu")
                model.to(self.device)
            else:
                raise

        self._model_name = model_name
        self._pretrained = pretrained
        logger.info("Loaded transformers model: {}", model_name)

    def preprocess(self, image: Any) -> dict[str, Any]:
        """Preprocess one image or a list of images.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or list of these
            Single image or batch of images.

        Returns
        -------
        dict[str, Any]
            Processor output dict with batched ``pixel_values`` shape
            ``(B, C, H, W)`` and any other required inputs.
        """
        imgs = self._load_images(image)
        if self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self._processor(images=imgs, return_tensors="pt")
        return dict(inputs)

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run the model forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Any
            HuggingFace ``ModelOutput`` or ``Tensor``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        moved = self._move_to_device(inputs)
        import torch

        with torch.no_grad():
            return self.model(**moved)

    def get_model_meta(self) -> ModelInfo:
        """Return ModelInfo for the loaded transformers model."""
        return ModelInfo(model_id=self._model_name, backend="transformers")

    def dependency_names(self) -> tuple[str, ...]:
        """Return Transformers extraction dependencies."""
        return ("torch", "transformers", "Pillow")

    def get_model_revision(self) -> str:
        """Return an immutable loaded-weight revision when pretrained weights were used."""
        if self._pretrained is not True:
            return "unknown"
        return super().get_model_revision()

    def get_preprocessing_description(self) -> str:
        """Describe the locally loaded processor without registry access."""
        if self._processor is None:
            return "unknown"
        cls_name = type(self._processor).__name__
        image_processor = getattr(self._processor, "image_processor", self._processor)
        details: list[str] = []
        fields = (
            "do_resize",
            "size",
            "do_center_crop",
            "crop_size",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "resample",
        )
        for name in fields:
            value = getattr(image_processor, name, None)
            if value is not None:
                serialized = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
                details.append(f"{name}={serialized}")
        return cls_name if not details else f"{cls_name}({';'.join(details)})"

    # ------------------------------------------------------------------
    # Hook management override — uses hookable_model for CLIP sub-encoders
    # ------------------------------------------------------------------

    def register_hooks(self, layer_names: list[str]) -> None:
        """Attach hooks on :attr:`hookable_model`.

        Parameters
        ----------
        layer_names : list[str]
            Module names as returned by :meth:`enumerate_modules`.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before registering hooks.")

        self.remove_hooks()
        self._activations.clear()

        import torch.nn as nn
        from torch import Tensor

        named = dict(self.hookable_model.named_modules())
        missing = [n for n in layer_names if n not in named]
        if missing:
            raise ValueError(f"Layer(s) not found in model: {missing}")

        for name in layer_names:
            module = named[name]

            def _hook(mod: nn.Module, inp: Any, output: Any, _n: str = name) -> None:  # noqa: ARG001
                act = output[0] if isinstance(output, tuple) else output
                if hasattr(act, "last_hidden_state"):
                    act = act.last_hidden_state
                if isinstance(act, Tensor):
                    self._activations[_n] = act.detach().cpu()

            handle = module.register_forward_hook(_hook)
            self._hooks.append(handle)

        logger.debug("Registered {} hooks", len(layer_names))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
