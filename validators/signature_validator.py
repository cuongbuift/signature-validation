"""
Signature validation using multi-metric comparison:
  1. Deep CNN features (EfficientNet-B0) – high-level semantic similarity
  2. SSIM  (Structural Similarity Index)  – overall structural likeness
  3. ORB   (Oriented FAST and Rotated BRIEF) – keypoint feature matching
  4. Contour similarity – shape / stroke comparison

Overall score = trung bình của SIAMESE + DEEP CNN + SSIM (ORB và Contour chỉ hiển thị, không ảnh hưởng tổng hợp).
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, field

from config import settings


@dataclass
class SignatureScores:
    siamese_score: float
    deep_score: float
    ssim_score: float
    orb_score: float
    contour_score: float
    overall_score: float
    is_valid: bool
    detail: dict = field(default_factory=dict)


class SignatureValidator:
    TARGET_SIZE: tuple[int, int] = settings.target_size

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        siamese_weight: float = 0.35,
        deep_weight: float = 0.30,
        ssim_weight: float = 0.20,
        orb_weight: float = 0.10,
        contour_weight: float = 0.05,
    ):
        self.threshold    = similarity_threshold
        self.w_siamese    = siamese_weight
        self.w_deep       = deep_weight
        self.w_ssim       = ssim_weight
        self.w_orb        = orb_weight
        self.w_contour    = contour_weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        input_path: str | Path,
        reference_paths: list[str | Path],
    ) -> SignatureScores:
        """
        Validate *input_path* against one or two reference signatures.
        Returns the best-matching score across all references.
        """
        if not reference_paths:
            raise ValueError("Cần ít nhất 1 chữ ký mẫu để so sánh.")

        input_img     = self._load_and_preprocess(input_path)
        input_feat    = self._deep_features(input_img)

        best: SignatureScores | None = None
        per_ref: list[dict] = []

        for ref_path in reference_paths:
            ref_img      = self._load_and_preprocess(ref_path)
            ref_feat     = self._deep_features(ref_img)
            siamese_sim  = self._siamese_score(input_img, ref_img)
            scores       = self._compare(input_img, ref_img, input_feat, ref_feat, siamese_sim)

            per_ref.append({
                "reference": str(ref_path),
                "siamese":   round(scores["siamese"], 4),
                "deep":      round(scores["deep"],    4),
                "ssim":      round(scores["ssim"],    4),
                "orb":       round(scores["orb"],     4),
                "contour":   round(scores["contour"], 4),
                "overall":   round(scores["overall"], 4),
            })

            if best is None or scores["overall"] > best.overall_score:
                best = SignatureScores(
                    siamese_score = scores["siamese"],
                    deep_score    = scores["deep"],
                    ssim_score    = scores["ssim"],
                    orb_score     = scores["orb"],
                    contour_score = scores["contour"],
                    overall_score = scores["overall"],
                    is_valid      = scores["overall"] >= self.threshold,
                    detail        = {},
                )

        best.detail = {"per_reference": per_ref, "threshold": self.threshold}
        best.is_valid = best.overall_score >= self.threshold
        return best

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, path: str | Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise ValueError(f"Không thể đọc ảnh: {path}")

        # 1. Remove red stamps (dấu mộc đỏ) using HSV color masking
        img_bgr = self._remove_red_stamp(img_bgr)

        # 2. Convert to grayscale
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Binarize (Otsu threshold)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Remove printed text (thin strokes) — keep only thick handwritten strokes
        binary = self._remove_thin_strokes(binary)

        # 5. Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # 6. Crop to bounding box of signature strokes
        cropped = self._crop_to_content(cleaned)

        # 7. Resize to standard canvas (keep aspect ratio, pad with white)
        resized = self._resize_with_padding(cropped, self.TARGET_SIZE)

        return resized

    @staticmethod
    def _remove_red_stamp(img_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        result   = img_bgr.copy()
        result[red_mask > 0] = [255, 255, 255]
        return result

    @staticmethod
    def _remove_thin_strokes(binary: np.ndarray, min_thickness: int = 2) -> np.ndarray:
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_thickness, min_thickness))
        eroded  = cv2.erode(binary, k, iterations=1)
        restored = cv2.dilate(eroded, k, iterations=1)
        return restored

    @staticmethod
    def _crop_to_content(binary: np.ndarray, margin: int = 10) -> np.ndarray:
        coords = cv2.findNonZero(binary)
        if coords is None:
            return binary
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(binary.shape[1] - x, w + 2 * margin)
        h = min(binary.shape[0] - y, h + 2 * margin)
        return binary[y : y + h, x : x + w]

    @staticmethod
    def _resize_with_padding(img: np.ndarray, target: tuple[int, int]) -> np.ndarray:
        tw, th = target
        h, w   = img.shape
        scale  = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas  = np.zeros((th, tw), dtype=np.uint8)
        x_off   = (tw - new_w) // 2
        y_off   = (th - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return canvas

    # ------------------------------------------------------------------
    # Deep CNN feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_features(img: np.ndarray) -> np.ndarray | None:
        """
        Extract a 1280-dim L2-normalised EfficientNet-B0 feature vector.
        Returns None if the extractor is unavailable (import error, etc.).
        """
        try:
            from validators.deep_feature_extractor import DeepFeatureExtractor
            return DeepFeatureExtractor.instance().extract(img)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------

    def _compare(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        feat_a: np.ndarray | None,
        feat_b: np.ndarray | None,
        siamese_sim: float = 0.0,
    ) -> dict:
        ssim_score    = self._ssim(img_a, img_b)
        orb_score     = self._orb(img_a, img_b)
        contour_score = self._contour(img_a, img_b)
        deep_score    = self._deep_sim(feat_a, feat_b)

        # Weighted sum using configured weights; normalise by the sum of weights
        # whose metric is actually available (Siamese/Deep may be 0 if model not loaded).
        weighted = 0.0
        weight_sum = 0.0

        if siamese_sim > 0 or self.w_siamese > 0:
            weighted   += self.w_siamese * siamese_sim
            weight_sum += self.w_siamese

        deep_available = feat_a is not None and feat_b is not None
        if deep_available or self.w_deep > 0:
            weighted   += self.w_deep * (deep_score if deep_available else 0.0)
            weight_sum += self.w_deep

        weighted   += self.w_ssim * ssim_score + self.w_orb * orb_score + self.w_contour * contour_score
        weight_sum += self.w_ssim + self.w_orb + self.w_contour

        overall = float(weighted / weight_sum) if weight_sum > 0 else 0.0

        return {
            "siamese": float(siamese_sim),
            "deep":    float(deep_score),
            "ssim":    float(ssim_score),
            "orb":     float(orb_score),
            "contour": float(contour_score),
            "overall": float(overall),
        }

    @staticmethod
    def _siamese_score(img_a: np.ndarray, img_b: np.ndarray) -> float:
        try:
            from validators.siamese_network import SiameseValidator
            return SiameseValidator.instance().compare(img_a, img_b)
        except Exception:
            return 0.0

    @staticmethod
    def _deep_sim(feat_a: np.ndarray | None, feat_b: np.ndarray | None) -> float:
        if feat_a is None or feat_b is None:
            return 0.0
        from validators.deep_feature_extractor import DeepFeatureExtractor
        return DeepFeatureExtractor.cosine_similarity(feat_a, feat_b)

    @staticmethod
    def _ssim(a: np.ndarray, b: np.ndarray) -> float:
        score, _ = ssim(a, b, full=True)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _orb(a: np.ndarray, b: np.ndarray) -> float:
        orb = cv2.ORB_create(nfeatures=500)
        kp_a, des_a = orb.detectAndCompute(a, None)
        kp_b, des_b = orb.detectAndCompute(b, None)
        if des_a is None or des_b is None or len(kp_a) == 0 or len(kp_b) == 0:
            return 0.0
        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_a, des_b)
        if not matches:
            return 0.0
        max_dist = max(m.distance for m in matches) or 1
        good     = [m for m in matches if m.distance < 0.75 * max_dist]
        score    = len(good) / max(len(kp_a), len(kp_b))
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _contour(a: np.ndarray, b: np.ndarray) -> float:
        cnts_a, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_b, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_a or not cnts_b:
            return 0.0
        main_a = max(cnts_a, key=cv2.contourArea)
        main_b = max(cnts_b, key=cv2.contourArea)
        dist   = cv2.matchShapes(main_a, main_b, cv2.CONTOURS_MATCH_I2, 0)
        return float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))
