"""timm-based vision model backend."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from vneurotk.vision.meta import ModelInfo
from vneurotk.vision.model.backend.base import BaseBackend

__all__ = ["TimmBackend"]


class TimmBackend(BaseBackend):
    """Backend powered by the ``timm`` library.

    Any model available via ``timm.create_model()`` is supported.
    Preprocessing uses the model's registered data config so no
    ImageNet mean/std are hard-coded.

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
        self._transform = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a timm model.

        Parameters
        ----------
        model_name : str
            E.g. ``"vit_base_patch16_224"`` or ``"resnet50"``.
        pretrained : bool
            Load pretrained weights.
        """
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except ImportError as exc:
            raise ImportError("timm is required for TimmBackend.  Install with: uv add timm") from exc

        logger.info("Loading timm model: {} (pretrained={})", model_name, pretrained)
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        self.model.to(self.device)

        self._model_name = model_name
        cfg = self.model.pretrained_cfg if hasattr(self.model, "pretrained_cfg") else {}
        data_config = resolve_data_config(cfg or {})
        self._transform = create_transform(**data_config)
        logger.info("Loaded timm model: {}", model_name)

    def preprocess(self, image: Any) -> dict[str, Any]:
        """Preprocess one image or a list of images.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or str or Path or list of these
            Single image or batch of images.

        Returns
        -------
        dict[str, Any]
            ``{"pixel_values": Tensor}`` with shape ``(B, C, H, W)``.
        """
        import torch

        imgs = self._load_images(image)
        if self._transform is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        tensors = [self._transform(img) for img in imgs]
        return {"pixel_values": torch.stack(tensors, dim=0)}

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run the timm model forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Tensor
            Model output, shape ``(1, n_classes)`` or ``(1, D)``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        px = self._move_to_device(inputs)["pixel_values"]
        with torch.no_grad():
            return self.model(px)

    def get_model_meta(self) -> ModelInfo:
        """Return ModelInfo for the loaded timm model."""
        return ModelInfo(model_id=self._model_name, backend="timm")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
