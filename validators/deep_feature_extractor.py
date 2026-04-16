"""
Deep feature extractor for signature images using EfficientNet-B0.

EfficientNet-B0 pre-trained on ImageNet is used as a fixed feature backbone.
The global-average-pool output (1 280-dim vector) captures high-level visual
patterns that are far more discriminative than hand-crafted metrics (SSIM /
ORB / contour) for verifying whether two signatures belong to the same person.

Inference runs on Apple MPS → CUDA → CPU, in that priority order.
The model is loaded once (singleton) and shared across requests.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

logger = logging.getLogger(__name__)


# ── ImageNet normalisation used during EfficientNet pre-training ─────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DeepFeatureExtractor:
    """
    Singleton wrapper around EfficientNet-B0 (feature extraction mode).

    Usage
    -----
    extractor = DeepFeatureExtractor.instance()
    vec = extractor.extract(gray_image_np)   # → np.ndarray shape (1280,)
    sim = extractor.cosine_similarity(vec_a, vec_b)   # → float [0, 1]
    """

    _instance: "DeepFeatureExtractor | None" = None

    def __init__(self) -> None:
        self.device = _best_device()
        logger.info("DeepFeatureExtractor: loading EfficientNet-B0 on %s …", self.device)

        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = tv_models.efficientnet_b0(weights=weights)

        # Drop the classifier; keep features + avgpool → 1280-dim output
        self.model: nn.Module = nn.Sequential(
            backbone.features,
            backbone.avgpool,
            nn.Flatten(),
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("DeepFeatureExtractor: ready (device=%s, output_dim=1280).", self.device)

    @classmethod
    def instance(cls) -> "DeepFeatureExtractor":
        """Return the module-level singleton (lazy-initialised)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def extract(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Extract a 1 280-dim L2-normalised feature vector from a preprocessed
        grayscale binary signature image (uint8, values 0/255).

        Parameters
        ----------
        gray_img:
            Grayscale numpy array (H × W), dtype uint8.
            Typically the output of  SignatureValidator._load_and_preprocess().

        Returns
        -------
        np.ndarray
            Shape (1280,), L2-normalised, dtype float32.
        """
        # Repeat gray channel → 3-channel so ImageNet transforms apply
        rgb = np.stack([gray_img] * 3, axis=-1)   # (H, W, 3)

        tensor = _TRANSFORM(rgb).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        with torch.no_grad():
            feat = self.model(tensor)              # (1, 1280)

        vec = feat.squeeze(0).cpu().numpy().astype(np.float32)

        # L2 normalise → cosine similarity reduces to dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalised vectors → [0, 1].

        Because both vectors are already L2-normalised in  extract(),
        cosine similarity is simply the dot product, clamped to [0, 1].
        """
        score = float(np.dot(vec_a, vec_b))
        return float(np.clip(score, 0.0, 1.0))
