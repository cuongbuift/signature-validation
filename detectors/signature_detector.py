"""
YOLO-based signature region detector.

Detects the bounding box of a handwritten signature in a document image,
then crops and returns that region so downstream preprocessing and comparison
work on the signature only — ignoring surrounding printed text, stamps, etc.

Usage
-----
detector = SignatureDetector(model_path="models/signature_yolo.pt")
cropped  = detector.detect_and_crop("path/to/image.jpg")
# cropped is a BGR numpy array, or None if no signature found
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SignatureDetector:
    """
    Wraps a YOLOv8 model trained to detect handwritten signatures.

    Parameters
    ----------
    model_path:
        Path to the YOLO weights file (*.pt).  If the file does not exist
        the detector logs a warning and `detect_and_crop` always returns
        ``None`` (caller should fall back to the original image).
    conf:
        Minimum confidence threshold for a detection to be accepted.
    padding:
        Fractional padding added around the detected box (0.05 = 5 %).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        padding: float = 0.08,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.padding = padding
        self._model = None          # lazy-loaded on first call
        self._available = False     # set True once the model loads OK

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_crop(
        self,
        image_path: str | Path,
    ) -> np.ndarray | None:
        """
        Run YOLO on *image_path* and return the cropped signature region
        as a BGR numpy array.

        Returns
        -------
        np.ndarray
            Cropped BGR image containing the detected signature.
        None
            If the model is not available, or no signature was detected.
            Callers should use the original image in this case.
        """
        model = self._get_model()
        if model is None:
            return None

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning("Cannot read image: %s", image_path)
            return None

        try:
            results = model(str(image_path), conf=self.conf, verbose=False)
        except Exception as exc:
            logger.warning("YOLO inference failed: %s", exc)
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            logger.debug("No signature detected in %s", image_path)
            return None

        # Pick the box with the highest confidence score
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = (
            boxes.xyxy[best_idx].cpu().numpy().astype(int)
        )

        h, w = img_bgr.shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1

        # Apply fractional padding
        pad_x = int(box_w * self.padding)
        pad_y = int(box_h * self.padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        cropped = img_bgr[y1:y2, x1:x2]
        if cropped.size == 0:
            logger.warning("Empty crop for %s (box %s)", image_path, (x1, y1, x2, y2))
            return None

        logger.debug(
            "Signature detected in %s: box=(%d,%d,%d,%d) conf=%.3f",
            image_path,
            x1, y1, x2, y2,
            float(boxes.conf[best_idx]),
        )
        return cropped

    def is_available(self) -> bool:
        """Return True if the model file exists and loaded successfully."""
        return self._available or (self._get_model() is not None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the YOLO model on first call."""
        if self._model is not None:
            return self._model

        if not self.model_path.exists():
            logger.warning(
                "YOLO signature model not found at '%s'. "
                "Signature detection will be skipped (using full image). "
                "Run  python download_model.py  to download the model.",
                self.model_path,
            )
            return None

        try:
            from ultralytics import YOLO  # imported here to keep startup fast

            self._model = YOLO(str(self.model_path))
            self._available = True
            logger.info("YOLO signature model loaded from '%s'.", self.model_path)
            return self._model
        except Exception as exc:
            logger.error("Failed to load YOLO model from '%s': %s", self.model_path, exc)
            return None
