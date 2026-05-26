"""
SigVer-based signature feature extractor using the pretrained SigNet model.
https://github.com/luizgh/sigver

Model file: models/signet.pth
Download: python -c "from validators.sigver_validator import SigverValidator; SigverValidator.download_model()"
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

_MODEL_URL = "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
_GDRIVE_FILE_ID = "0B29vNACcjvzVc1RfVkg5dUh3b1E"


class SigverValidator:
    """Singleton that extracts 2048-dim SigNet features and computes cosine similarity."""

    _instance: "SigverValidator | None" = None

    def __init__(self, model_path: str | Path):
        import torch
        from sigver.featurelearning.models import SigNet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict, _, _ = torch.load(str(model_path), map_location=self.device)
        self.model = SigNet().eval().to(self.device)
        self.model.load_state_dict(state_dict)
        self._torch = torch

    @classmethod
    def instance(cls, model_path: str | Path | None = None) -> "SigverValidator":
        if cls._instance is None:
            from config import settings
            path = Path(model_path) if model_path else settings.sigver_model_path
            if not path.exists():
                raise FileNotFoundError(
                    f"SigNet model not found at '{path}'. "
                    "Run: python scripts/download_signet.py"
                )
            cls._instance = cls(path)
        return cls._instance

    # SigNet was trained on 155×220 (H×W) images
    INPUT_H = 155
    INPUT_W = 220

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract a 2048-dim L2-normalised feature vector from a preprocessed grayscale image."""
        import torch
        import cv2

        # Resize to the fixed input size SigNet expects (155×220)
        resized = cv2.resize(img, (self.INPUT_W, self.INPUT_H), interpolation=cv2.INTER_AREA)

        # uint8 [0,255] → float32 [0,1], shape (1,1,H,W)
        tensor = (
            torch.from_numpy(resized.astype(np.float32) / 255.0)
            .unsqueeze(0)   # H,W → 1,H,W
            .unsqueeze(0)   # → 1,1,H,W
            .to(self.device)
        )
        with torch.no_grad():
            feat = self.model(tensor)   # (1, 2048)
        feat = feat.squeeze(0).cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), 0.0, 1.0))

    def compare(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        feat_a = self.extract(img_a)
        feat_b = self.extract(img_b)
        return self.cosine_similarity(feat_a, feat_b)
