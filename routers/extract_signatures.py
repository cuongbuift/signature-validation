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

    best_y_bottom: Optional[int] = None
    best_y_top: Optional[int] = None

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
            # Keep the topmost match (first label line on the page)
            if best_y_top is None or y_top < best_y_top:
                best_y_bottom = y_bottom
                best_y_top    = y_top

    return best_y_bottom


def _tight_crop(img: Image.Image, padding: int = 18) -> Optional[Image.Image]:
    """
    Auto-crop an image to the tight bounding box of its non-white content.
    `padding` (pixels) is added on all sides so marks aren't touching the edge.
    Returns None if no ink is found (blank region).
    """
    arr = np.array(img.convert("RGB"))

    # A pixel is considered "ink" when it is darker than the threshold on any channel.
    # Threshold 230 catches light-grey strokes too.
    ink_mask = np.any(arr < 230, axis=2)  # shape (H, W), True = ink pixel

    rows = np.where(ink_mask.any(axis=1))[0]  # rows that contain ink
    cols = np.where(ink_mask.any(axis=0))[0]  # cols that contain ink

    if rows.size == 0 or cols.size == 0:
        return None  # nothing to crop

    h, w = arr.shape[:2]
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
    """
    w, h = img.size
    raw = img.crop((0, label_y_bottom, int(w * 0.40), h))

    tight = _tight_crop(raw)
    return tight if tight is not None else raw


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
