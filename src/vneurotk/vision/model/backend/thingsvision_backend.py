"""ThingsVision backend for DNN feature extraction."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from vneurotk.vision.meta import ModelInfo
from vneurotk.vision.model.backend.base import BaseBackend

__all__ = ["ThingsVisionBackend"]


class ThingsVisionBackend(BaseBackend):
    """Backend powered by the ``thingsvision`` library.

    ``thingsvision`` must be installed before instantiating this class;
    a missing installation raises ``ImportError`` immediately (fail-fast).

    Parameters
    ----------
    source : str
        Model source string for thingsvision, e.g. ``"timm"`` or
        ``"torchvision"``.
    device : str or torch.device
        Inference device (default ``"cpu"``).
    """

    def __init__(
        self,
        source: str = "torchvision",
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            from thingsvision.model_class import Model  # noqa: F401  # ty: ignore[unresolved-import]
        except ImportError as exc:
            raise ImportError(
                "thingsvision is required for ThingsVisionBackend.  Install with: uv add thingsvision"
            ) from exc

        super().__init__(device)
        self._tv_source = source
        self._model_name: str = ""
        self._extractor = None
        self._transform = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a model via thingsvision.

        Parameters
        ----------
        model_name : str
            Model name as accepted by ``thingsvision.get_extractor()``.
        pretrained : bool
            Load pretrained weights.
        """
        from thingsvision.model_class import Model  # ty: ignore[unresolved-import]

        logger.info(
            "Loading thingsvision model: {} (source={}, pretrained={})",
            model_name,
            self._tv_source,
            pretrained,
        )

        device_str = str(self.device)
        self._tv_model = Model(
            model_name=model_name,
            pretrained=pretrained,
            device=device_str,
            backend="pt",
        )
        self.model = self._tv_model.model
        self.model.eval()
        self._transform = self._tv_model.get_transformations()
        self._model_name = model_name
        logger.info("Loaded thingsvision model: {}", model_name)

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
        """Run the model forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        px = self._move_to_device(inputs)["pixel_values"]
        with torch.no_grad():
            return self.model(px)

    def get_model_meta(self) -> ModelInfo:
        """Return ModelInfo for the loaded thingsvision model."""
        return ModelInfo(model_id=self._model_name, backend="thingsvision")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
