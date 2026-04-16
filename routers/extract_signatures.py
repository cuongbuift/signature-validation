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

# ── OCR keyword (Vietnamese, normalised to NFC) ───────────────────────────────
CUSTOMER_LABEL = unicodedata.normalize("NFC", "Khách hàng đã nhận đủ hàng")

# Keywords that indicate a signature footer area even when OCR is imperfect
FOOTER_KEYWORDS = [
    unicodedata.normalize("NFC", kw)
    for kw in [
        "Khách hàng đã nhận",
        "nhận đủ hàng",
        "Ký và ghi rõ",
        "ghi rõ họ tên",
    ]
]


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
    Returns None if the label is not found.
    """
    reader = _get_ocr_reader()
    img_np = np.array(img)

    # detail=1 → returns (bbox, text, confidence)
    results = reader.readtext(img_np, detail=1, paragraph=False)

    best_y_bottom: Optional[int] = None
    best_y_top: Optional[int] = None

    for (bbox, text, conf) in results:
        norm = _normalize(text)
        # bbox is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        ys = [pt[1] for pt in bbox]
        y_top    = int(min(ys))
        y_bottom = int(max(ys))

        for kw in FOOTER_KEYWORDS:
            if kw.lower() in norm.lower():
                if best_y_bottom is None or y_top < (best_y_top or 9999):
                    best_y_bottom = y_bottom
                    best_y_top    = y_top
                break

    return best_y_bottom


def _crop_signature(img: Image.Image, label_y_bottom: int) -> Image.Image:
    """
    Given the bottom-y of the label text, crop the customer-signature region:
      - vertically: from label_y_bottom to label_y_bottom + ~200 px (at 200 DPI)
      - horizontally: left 40 % of the page
    """
    w, h = img.size
    x0 = 0
    x1 = int(w * 0.40)
    y0 = label_y_bottom
    y1 = h  # extend to the bottom of the page so nothing gets cut off
    return img.crop((x0, y0, x1, y1))


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

        # --- OCR: search in the bottom 35 % of the page only (faster + accurate)
        w, h  = img.size
        roi_y = int(h * 0.65)
        roi   = img.crop((0, roi_y, w, h))

        label_y_in_roi = _ocr_find_label_bottom(roi)
        if label_y_in_roi is None:
            logger.debug("Page %d: label not found by OCR, skipping.", page_num + 1)
            continue

        # Convert ROI-relative y back to full-image y
        label_y_bottom = roi_y + label_y_in_roi

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
