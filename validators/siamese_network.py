"""
Siamese Neural Network for offline signature verification.

Architecture
------------
Two identical CNN branches share weights. Each branch maps a preprocessed
signature image to a 64-dim L2-normalised embedding.  A pair's similarity
is derived from the Euclidean distance between the two embeddings:

    similarity = 1 / (1 + distance)          → [0, 1]

Training uses Contrastive Loss (LeCun 2005):
    L = y·d² + (1-y)·max(margin - d, 0)²
where y=1 for a genuine pair (same signer), y=0 for a forged/impostor pair.

Input
-----
Grayscale binary images of size (TARGET_H × TARGET_W) produced by
SignatureValidator._load_and_preprocess(), normalised to [0, 1].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
EMBED_DIM   = 64
CONTRASTIVE_MARGIN = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SiameseCNN(nn.Module):
    """
    Shared CNN backbone that maps a (1 × H × W) signature image to a
    64-dim L2-normalised embedding.

    Input shape: (batch, 1, TARGET_H, TARGET_W)  — typically (B, 1, 150, 300)
    Output shape: (batch, EMBED_DIM=64)
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1,   32, kernel=5, pool=True),   # →  32 × H/2  × W/2
            _ConvBlock(32,  64, kernel=5, pool=True),   # →  64 × H/4  × W/4
            _ConvBlock(64, 128, kernel=3, pool=True),   # → 128 × H/8  × W/8
            _ConvBlock(128,256, kernel=3, pool=True),   # → 256 × H/16 × W/16
        )
        # Global Average Pooling → 1×1 per channel (MPS-safe, standard for embeddings)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))         # → 256 × 1 × 1

        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, EMBED_DIM),
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)   # L2 normalise → unit sphere

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_one(x1), self.forward_one(x2)


# ═══════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks (LeCun et al., 2005).

    Parameters
    ----------
    margin:
        Minimum distance expected between embeddings of different signers.
        Embeddings of the same signer should have distance ≈ 0.
    """

    def __init__(self, margin: float = CONTRASTIVE_MARGIN) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embed1: torch.Tensor,
        embed2: torch.Tensor,
        label: torch.Tensor,   # 1 = same signer, 0 = different signer
    ) -> torch.Tensor:
        dist  = F.pairwise_distance(embed1, embed2, p=2)
        pos   = label       * dist.pow(2)
        neg   = (1 - label) * F.relu(self.margin - dist).pow(2)
        return (pos + neg).mean() / 2


# ═══════════════════════════════════════════════════════════════════════════
# Inference wrapper (singleton)
# ═══════════════════════════════════════════════════════════════════════════

def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SiameseValidator:
    """
    Inference-only wrapper around a trained SiameseCNN.

    Usage
    -----
    sv = SiameseValidator.instance()
    score = sv.compare(gray_img_a, gray_img_b)   # float [0, 1]
    """

    _instance: "SiameseValidator | None" = None

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.device     = _best_device()
        self._model: SiameseCNN | None = None
        self._loaded    = False

    @classmethod
    def instance(cls) -> "SiameseValidator":
        if cls._instance is None:
            from config import settings
            cls._instance = cls(model_path=settings.siamese_model_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force reload after training."""
        cls._instance = None

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.model_path.exists()

    def compare(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> float:
        """
        Compute similarity between two preprocessed grayscale signature
        images (dtype uint8, values 0/255).

        Returns
        -------
        float in [0, 1].  Higher = more similar.
        None is never returned; returns 0.0 on any error.
        """
        model = self._get_model()
        if model is None:
            return 0.0

        try:
            t1 = self._to_tensor(img_a)
            t2 = self._to_tensor(img_b)
            with torch.no_grad():
                e1, e2 = model(t1, t2)
            dist = float(F.pairwise_distance(e1, e2, p=2).item())
            # Convert distance [0, ∞) → similarity [0, 1]
            return float(1.0 / (1.0 + dist))
        except Exception as exc:
            logger.warning("SiameseValidator.compare error: %s", exc)
            return 0.0

    # ------------------------------------------------------------------

    def _get_model(self) -> SiameseCNN | None:
        if self._loaded:
            return self._model
        self._loaded = True

        if not self.model_path.exists():
            logger.info(
                "Siamese model not found at '%s'. "
                "Call POST /siamese/train to train it from existing signatures.",
                self.model_path,
            )
            return None

        try:
            net = SiameseCNN()
            state = torch.load(str(self.model_path), map_location="cpu")
            net.load_state_dict(state)
            net.to(self.device)
            net.eval()
            self._model = net
            logger.info(
                "Siamese model loaded from '%s' (device=%s).",
                self.model_path,
                self.device,
            )
            return self._model
        except Exception as exc:
            logger.error("Failed to load Siamese model: %s", exc)
            return None

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert uint8 grayscale (H×W) → float32 tensor (1×1×H×W) on device."""
        x = img.astype(np.float32) / 255.0
        t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        return t.to(self.device)
