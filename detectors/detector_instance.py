"""
Module-level singleton for the signature detector.

Import `detector` from this module everywhere you need detection;
the YOLO model is lazy-loaded on first use and shared across requests.
"""
from config import settings
from .signature_detector import SignatureDetector

detector = SignatureDetector(
    model_path=settings.yolo_model_path,
    conf=settings.yolo_conf,
    padding=settings.yolo_padding,
)
