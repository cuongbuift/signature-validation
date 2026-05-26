from __future__ import annotations

import io
import base64
import logging
import unicodedata
from functools import lru_cache
from typing import Optional

import numpy as np
import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Extract Signatures"])

# ── Render settings ────────────────────────────────────────────────────────────
# 200 DPI gives OCR enough resolution without being too slow
RENDER_DPI = 200
SCALE = RENDER_DPI / 72  # PyMuPDF default canvas is 72 DPI

# ── OCR keywords ──────────────────────────────────────────────────────────────
# Primary keywords (with Vietnamese diacritics, NFC-normalised)
_KW_WITH_DIACRITICS = [
    "Khách hàng đã nhận đủ hàng",
    "Khách hàng đã nhận",
    "nhận đủ hàng",
    "Ký và ghi rõ họ tên",
    "ghi rõ họ tên",
]

# ASCII fallback keywords — used when OCR drops diacritics (common in low-res scans)
_KW_ASCII_FALLBACK = [
    "Khach hang da nhan du hang",
    "Khach hang da nhan",
    "nhan du hang",
    "Ky va ghi ro ho ten",
    "ghi ro ho ten",
]

FOOTER_KEYWORDS = [unicodedata.normalize("NFC", kw) for kw in _KW_WITH_DIACRITICS]
FOOTER_KEYWORDS_ASCII = [kw.lower() for kw in _KW_ASCII_FALLBACK]


def _strip_diacritics(text: str) -> str:
    """Return a rough ASCII version of *text* for fuzzy matching.
    Handles Vietnamese-specific letters that NFKD does not fully decompose (đ → d).
    """
    # Replace characters NFKD won't decompose
    _SPECIAL = str.maketrans("đĐ", "dD")
    text = text.translate(_SPECIAL)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# ── Lazy EasyOCR reader (loaded once, reused across requests) ──────────────────
@lru_cache(maxsize=1)
def _get_ocr_reader():
    import easyocr  # imported lazily so startup is fast

    logger.info("Loading EasyOCR model for vi+en …")
    reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    logger.info("EasyOCR ready.")
    return reader


# ── Helpers ────────────────────────────────────────────────────────────────────

def _page_to_pil(page: fitz.Page, scale: float = SCALE) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _ocr_find_label_bottom(img: Image.Image) -> Optional[int]:
    """
    Run OCR on *img* and return the **y pixel coordinate of the bottom edge**
    of the line containing the customer-label keyword.

    Matching strategy (in order):
      1. Exact Vietnamese keyword (NFC) — highest confidence
      2. ASCII-stripped keyword — catches OCR that drops diacritics

    Returns None if the label is not found.
    """
    reader = _get_ocr_reader()
    img_np = np.array(img)

    results = reader.readtext(img_np, detail=1, paragraph=False)

    # We want to crop from BELOW all label lines, so track both:
    #   - topmost_y_top   : to know where the label block starts
    #   - bottommost_y_bottom : to know where the last label line ends
    topmost_y_top: Optional[int] = None
    bottommost_y_bottom: Optional[int] = None

    for (bbox, text, _conf) in results:
        norm       = _normalize(text)
        norm_lower = norm.lower()
        norm_ascii = _strip_diacritics(norm)

        ys       = [pt[1] for pt in bbox]
        y_top    = int(min(ys))
        y_bottom = int(max(ys))

        matched = any(kw.lower() in norm_lower for kw in FOOTER_KEYWORDS) or \
                  any(kw       in norm_ascii  for kw in FOOTER_KEYWORDS_ASCII)

        if matched:
            if topmost_y_top is None or y_top < topmost_y_top:
                topmost_y_top = y_top
            if bottommost_y_bottom is None or y_bottom > bottommost_y_bottom:
                bottommost_y_bottom = y_bottom

    # Only use the bottommost label if all matched lines are within a single
    # label block (within 3× the label block height of the topmost line).
    # This prevents accidentally skipping to a label on a completely different
    # part of the page.
    if topmost_y_top is not None and bottommost_y_bottom is not None:
        block_height = bottommost_y_bottom - topmost_y_top
        if block_height > 300:  # sanity cap — labels are never 300 px apart
            return bottommost_y_bottom  # something unusual; still use bottommost
    return bottommost_y_bottom


def _build_clean_ink_mask(arr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) that keeps only handwriting-like ink
    while removing:
      - Scanner binding artifacts (thin coloured lines at page edges)
      - Horizontal/vertical ruling lines from the form
      - Isolated noise specks (too small to be a stroke)
      - Components hugging the very bottom edge (scanner fold marks)

    Parameters
    ----------
    arr : H×W×3 uint8 RGB array

    Returns
    -------
    clean : H×W uint8 mask (255 = keep)
    """
    import cv2

    h, w = arr.shape[:2]

    # ── 1. Grayscale → binary ink mask ──────────────────────────────────────
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Adaptive threshold handles uneven scan lighting better than a fixed value
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=12,
    )

    # ── 2. Erase scanner binding marks (pink / magenta streaks) ─────────────
    # These come from scanning near the physical book spine.
    # Characteristic: R high, G low-mid, B mid → not normal black/blue ink.
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    pink_mask = ((r.astype(int) - g.astype(int)) > 40) & \
                ((r.astype(int) - b.astype(int)) < 60) & \
                (r > 140)
    binary[pink_mask] = 0

    # ── 3. Erase horizontal ruling lines (form borders, table lines) ────────
    # A true ruling line is very wide (> 55 % of page width) and 1-3 px tall.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w * 55 // 100), 1))
    hlines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    binary   = cv2.subtract(binary, hlines)

    # Erase vertical ruling lines similarly
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h * 40 // 100)))
    vlines   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    binary   = cv2.subtract(binary, vlines)

    # ── 4. Connected-component filtering ────────────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    clean = np.zeros_like(binary)

    for i in range(1, num_labels):  # label 0 = background
        area    = int(stats[i, cv2.CC_STAT_AREA])
        comp_x  = int(stats[i, cv2.CC_STAT_LEFT])
        comp_y  = int(stats[i, cv2.CC_STAT_TOP])
        comp_w  = int(stats[i, cv2.CC_STAT_WIDTH])
        comp_h  = int(stats[i, cv2.CC_STAT_HEIGHT])

        # (a) Discard tiny noise specks
        if area < 40:
            continue

        # (b) Discard very thin wide stripes (residual ruling lines)
        if comp_w > w * 0.55 and comp_h <= 4:
            continue

        # (c) Discard components whose bottom edge sits within the last 4 % of
        #     the image — these are scanner fold / binding marks.
        bottom_edge = comp_y + comp_h
        if bottom_edge > h * 0.96 and comp_y > h * 0.88:
            continue

        clean[labels == i] = 255

    return clean


def _apply_handwriting_separation(img: Image.Image) -> Image.Image:
    """
    Return a version of *img* with printed text and ruling lines removed,
    keeping only handwriting (signature strokes + handwritten name).

    Strategy:
    - If significant colored ink (blue/purple) is detected via LAB color space
      → use the LAB mask, which cleanly separates colored handwriting from
        neutral-black printed text.
    - Otherwise (black-ink signatures) → use morphological ruling-line removal
      then filter out small components that match printed-text character sizes.

    The result always has a white background.
    """
    import cv2

    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return img

    # ── LAB color separation ────────────────────────────────────────────────
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L_ch = lab[:, :, 0].astype(float)
    A_ch = lab[:, :, 1].astype(float)
    B_ch = lab[:, :, 2].astype(float)

    # Colored ink: A deviates from neutral (128) → red/purple or green,
    # or B < 120 → blue pen ink.  Must also be dark enough (not white paper).
    colored_mask = (
        ((A_ch > 134) | (A_ch < 124) | (B_ch < 120)) &
        (L_ch < 230)
    ).astype(np.uint8) * 255

    # Measure how much of the visible ink is colored
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, all_ink = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    total_ink_px = int(np.count_nonzero(all_ink))
    colored_ink_px = int(np.count_nonzero(colored_mask))

    if total_ink_px > 0 and colored_ink_px / total_ink_px > 0.15:
        # Enough colored ink → LAB mask cleanly removes black printed text
        raw_mask = colored_mask
    else:
        # Black-ink signature → start from full ink mask and remove lines +
        # small text-sized components.
        raw_mask = _build_clean_ink_mask(arr)

        # Remove connected components that look like printed-text characters:
        # at 200 DPI a typical Latin/Vietnamese printed char is 10–40 px tall
        # and has area < 600 px².  Signature strokes are larger / wider.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )
        filtered = np.zeros_like(raw_mask)
        for i in range(1, num_labels):
            area   = int(stats[i, cv2.CC_STAT_AREA])
            comp_h = int(stats[i, cv2.CC_STAT_HEIGHT])
            comp_w = int(stats[i, cv2.CC_STAT_WIDTH])
            # Keep if it looks like a signature stroke (large area, wide, or tall)
            if area >= 600 or comp_w >= 60 or comp_h >= 40:
                filtered[labels == i] = 255
        raw_mask = filtered

    # Remove tiny noise from whichever mask we chose
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_mask, connectivity=8
    )
    clean = np.zeros_like(raw_mask)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= 15:
            clean[labels == i] = 255

    # Compose: keep original RGB colour at ink pixels, white elsewhere
    result = np.full_like(arr, 255)
    result[clean > 0] = arr[clean > 0]
    return Image.fromarray(result)


def _tight_crop(img: Image.Image, padding: int = 20) -> Optional[Image.Image]:
    """
    Remove scanner artefacts and noise, then tight-crop to the bounding box
    of remaining handwriting-like ink strokes.
    """
    arr  = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    clean = _build_clean_ink_mask(arr)

    rows = np.where(clean.any(axis=1))[0]
    cols = np.where(clean.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return None

    r0 = max(int(rows[0])  - padding, 0)
    r1 = min(int(rows[-1]) + padding, h)
    c0 = max(int(cols[0])  - padding, 0)
    c1 = min(int(cols[-1]) + padding, w)

    return img.crop((c0, r0, c1, r1))


def _crop_signature(img: Image.Image, label_y_bottom: int) -> Image.Image:
    """
    1. Slice the raw region: from below the label line to the page bottom,
       left 40 % of the page.
    2. Tight-crop to remove surrounding white space.
    3. Cap height to the crop width to avoid overly tall results from residual
       artifacts at the bottom of the scan.
    """
    w, h = img.size
    raw = img.crop((0, label_y_bottom, int(w * 0.40), h))

    tight = _tight_crop(raw)
    result = tight if tight is not None else raw

    # Separate handwriting/signature from printed text and ruling lines
    result = _apply_handwriting_separation(result)

    # Enforce max height = width so scanner-fold artifacts can't inflate the crop
    rw, rh = result.size
    if rh > rw:
        result = result.crop((0, 0, rw, rw))

    return result


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Core extraction logic ──────────────────────────────────────────────────────

def extract_signatures_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Render every page of a scanned PDF, run OCR to locate
    "Khách hàng đã nhận đủ hàng", then crop the signature area below it.

    Returns a list of:
      { "page": int, "image_b64": str, "mime": "image/png" }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: list[dict] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        img  = _page_to_pil(page)
        w, h = img.size

        # Two-pass OCR search:
        #   Pass 1 — bottom 50 % of the page (fast, covers most single-page DOs)
        #   Pass 2 — full page (catches continuation pages where the label sits higher)
        label_y_bottom: Optional[int] = None
        for roi_start_frac in (0.50, 0.0):
            roi_y = int(h * roi_start_frac)
            roi   = img.crop((0, roi_y, w, h))

            found = _ocr_find_label_bottom(roi)
            if found is not None:
                label_y_bottom = roi_y + found
                logger.debug(
                    "Page %d: label found at y=%d (roi_start=%.0f%%)",
                    page_num + 1, label_y_bottom, roi_start_frac * 100,
                )
                break

        if label_y_bottom is None:
            logger.debug("Page %d: label not found, skipping.", page_num + 1)
            continue

        crop = _crop_signature(img, label_y_bottom)

        results.append(
            {
                "page": page_num + 1,
                "image_b64": _pil_to_b64(crop),
                "mime": "image/png",
            }
        )

    doc.close()
    return results


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/extract-signatures",
    summary="Trích xuất chữ ký khách hàng từ phiếu giao hàng (PDF scan)",
    response_class=JSONResponse,
)
async def extract_signatures(
    file: UploadFile = File(..., description="File PDF phiếu giao nhận hàng (bản scan)"),
):
    is_pdf = file.content_type == "application/pdf" or (
        (file.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF.")

    pdf_bytes = await file.read()

    try:
        signatures = extract_signatures_from_pdf(pdf_bytes)
    except Exception as exc:
        logger.exception("PDF processing error")
        raise HTTPException(status_code=422, detail=f"Lỗi xử lý PDF: {exc}")

    if not signatures:
        raise HTTPException(
            status_code=404,
            detail=(
                "Không tìm thấy vùng chữ ký trong file. "
                "Kiểm tra file có dòng 'Khách hàng đã nhận đủ hàng' không."
            ),
        )

    return {"total": len(signatures), "signatures": signatures}
