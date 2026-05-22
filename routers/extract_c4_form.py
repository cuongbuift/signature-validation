from __future__ import annotations

import io
import base64
import logging
import re
import unicodedata
from typing import Optional

import cv2
import numpy as np
import fitz
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from routers.extract_signatures import (
    _get_ocr_reader,
    _page_to_pil,
    _strip_diacritics,
    _tight_crop,
    SCALE,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Extract C4 Form"])


# ── Lazy VietOCR predictor (loaded once) ──────────────────────────────────────

from functools import lru_cache

@lru_cache(maxsize=1)
def _get_vietocr():
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    cfg = Cfg.load_config_from_name("vgg_transformer")
    cfg["device"] = "cpu"
    cfg["predictor"]["beamsearch"] = False
    logger.info("Loading VietOCR model …")
    predictor = Predictor(cfg)
    logger.info("VietOCR ready.")
    return predictor


# ── Text normalisation helpers ─────────────────────────────────────────────────

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _ascii(text: str) -> str:
    return _strip_diacritics(_nfc(text))


# ── OCR helper: returns list of (text, x0, y0, x1, y1) ────────────────────────

def _ocr_blocks(img: Image.Image) -> list[dict]:
    """EasyOCR detects bounding boxes; VietOCR recognizes Vietnamese text."""
    reader = _get_ocr_reader()
    viet = _get_vietocr()

    # EasyOCR: detection only — we use its bboxes but replace its text with VietOCR
    easy_results = reader.readtext(np.array(img), detail=1, paragraph=False)

    blocks = []
    for (bbox, easy_text, conf) in easy_results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

        # Crop the detected region and let VietOCR recognize it
        pad = 4
        w, h = img.size
        crop = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                         min(w, x1 + pad), min(h, y1 + pad)))
        try:
            text = viet.predict(crop)
        except Exception:
            text = easy_text  # fallback to EasyOCR text on error

        blocks.append({
            "text": _nfc(text),
            "ascii": _ascii(text),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "conf": conf,
        })

    blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    return blocks


def _find_block(blocks: list[dict], keyword: str, ascii_keyword: str | None = None) -> Optional[dict]:
    """Return first block whose text contains *keyword* (case-insensitive)."""
    kw_lower = keyword.lower()
    kw_ascii = ascii_keyword or _ascii(keyword)
    for b in blocks:
        if kw_lower in b["text"].lower() or kw_ascii in b["ascii"]:
            return b
    return None


def _value_right_of(blocks: list[dict], anchor: dict, y_tol: int = 15) -> str:
    """Collect all block texts on the same line (within y_tol px) to the right of anchor."""
    parts = [
        b["text"] for b in blocks
        if abs(b["y0"] - anchor["y0"]) <= y_tol and b["x0"] > anchor["x1"]
    ]
    return " ".join(parts).strip()


def _value_below(blocks: list[dict], anchor: dict, x_range: tuple[int, int],
                 max_dy: int = 60) -> str:
    """Collect text blocks directly below anchor within x_range (x0..x1)."""
    x0, x1 = x_range
    parts = [
        b["text"] for b in blocks
        if b["y0"] > anchor["y1"]
        and b["y0"] - anchor["y1"] <= max_dy
        and b["x0"] >= x0 and b["x1"] <= x1 + 80
    ]
    return " ".join(parts).strip()


# ── Customer info extraction (page 1) ─────────────────────────────────────────

def _extract_customer_info(page_img: Image.Image) -> dict:
    blocks = _ocr_blocks(page_img)
    w, _h = page_img.size

    def find(kw, ascii_kw=None):
        return _find_block(blocks, kw, ascii_kw)

    # 1. Tên đăng ký kinh doanh
    b = find("đăng ký kinh doanh", "dang ky kinh doanh")
    ten_dkk = _value_right_of(blocks, b) if b else ""

    # 1.1. Tên cửa hàng
    b = find("tên cửa hàng", "ten cua hang")
    ten_cua_hang = _value_right_of(blocks, b) if b else ""

    # 2. Giấy phép kinh doanh + Ngày cấp + Nơi cấp
    b = find("giấy phép", "giay phep")
    gp_number = ""
    gp_ngay_cap = ""
    gp_noi_cap = ""
    if b:
        # value after "Giấy phép đăng ký kinh doanh:" label on same line
        same_line = [
            bl for bl in blocks
            if abs(bl["y0"] - b["y0"]) <= 15 and bl["x0"] > b["x1"]
        ]
        same_line.sort(key=lambda bl: bl["x0"])
        # first part is GP number, then "Ngày cấp:", date, "Nơi cấp:", place
        raw = " ".join(bl["text"] for bl in same_line)
        # try to split around "Ngày cấp" / "Noi cap"
        ngay_m = re.split(r"ngày\s*cấp|ngay\s*cap", raw, flags=re.IGNORECASE)
        if len(ngay_m) >= 2:
            gp_number = ngay_m[0].strip().strip(":")
            rest = ngay_m[1]
            noi_m = re.split(r"nơi\s*cấp|noi\s*cap", rest, flags=re.IGNORECASE)
            if len(noi_m) >= 2:
                gp_ngay_cap = noi_m[0].strip().strip(":")
                gp_noi_cap = noi_m[1].strip().strip(":")
            else:
                gp_ngay_cap = rest.strip().strip(":")
        else:
            gp_number = raw.strip()

        # Also try dedicated "Ngày cấp" and "Nơi cấp" blocks near the row
        b_ngay = _find_block(blocks, "ngày cấp", "ngay cap")
        if b_ngay and abs(b_ngay["y0"] - b["y0"]) <= 15:
            gp_ngay_cap = _value_right_of(blocks, b_ngay) or gp_ngay_cap
        b_noi = _find_block(blocks, "nơi cấp", "noi cap")
        if b_noi and abs(b_noi["y0"] - b["y0"]) <= 15:
            gp_noi_cap = _value_right_of(blocks, b_noi) or gp_noi_cap

    # Normalize: ngày cấp must be DD/MM/YYYY only — strip any trailing garbage
    date_m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", gp_ngay_cap)
    gp_ngay_cap = date_m.group(0) if date_m else gp_ngay_cap

    # 3. Địa chỉ kinh doanh
    b = find("địa chỉ kinh doanh", "dia chi kinh doanh")
    dia_chi = _value_right_of(blocks, b) if b else ""
    if not dia_chi and b:
        dia_chi = _value_below(blocks, b, (0, w))

    return {
        "ten_dang_ky_kinh_doanh": ten_dkk,
        "ten_cua_hang": ten_cua_hang,
        "giay_phep_kinh_doanh": {
            "so": gp_number,
            "ngay_cap": gp_ngay_cap,
            "noi_cap": gp_noi_cap,
        },
        "dia_chi_kinh_doanh": dia_chi,
    }


# ── Signature box extraction (page 2, section C) ──────────────────────────────

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _is_blank_cell(cell: Image.Image, dark_ratio_threshold: float = 0.001) -> bool:
    """Return True if the cell has almost no dark pixels (effectively empty)."""
    gray = cv2.cvtColor(np.array(cell.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_pixels = np.count_nonzero(binary)
    total = binary.size
    return (dark_pixels / total) < dark_ratio_threshold


def _is_slash_cell(cell: Image.Image) -> bool:
    """
    Return True if the cell contains a diagonal slash (meaning 'not applicable').
    Conditions:
      1. Dark pixel ratio < 8% (a slash is sparse; a real signature is denser)
      2. HoughLinesP finds a diagonal line (25–65°) spanning ≥35% of the cell diagonal
    """
    gray = cv2.cvtColor(np.array(cell.convert("RGB")), cv2.COLOR_RGB2GRAY)
    ch, cw = gray.shape
    if cw < 10 or ch < 10:
        return False

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # A real signature has more ink than a simple slash mark
    dark_ratio = np.count_nonzero(binary) / binary.size
    if dark_ratio >= 0.08:
        return False

    min_len = int(min(cw, ch) * 0.35)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                             threshold=25, minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return False

    diagonal = np.hypot(cw, ch)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx < 1:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if 25 <= angle <= 65:
            length = np.hypot(dx, dy)
            if length >= diagonal * 0.35:
                return True
    return False


_KY_LABEL_PATTERNS = [
    "ky va ghi ro ho ten",
    "ky va ghi ro",
    "ghi ro ho ten",
    "ho ten",
    "ky ten",
]


def _find_ky_label_bottom(cell_img: Image.Image) -> Optional[int]:
    """
    Find the bottom y-coordinate (in cell-local pixels) of the '(Ký và ghi rõ họ tên)'
    label. Tries broad pattern matching across all OCR blocks; returns the maximum y1
    of any matching block, or None if nothing matches.
    """
    reader = _get_ocr_reader()
    results = reader.readtext(np.array(cell_img), detail=1, paragraph=False)
    best_y1 = None
    for (bbox, text, _conf) in results:
        ascii_text = _ascii(text).lower()
        if any(pat in ascii_text for pat in _KY_LABEL_PATTERNS):
            ys = [pt[1] for pt in bbox]
            y1 = int(max(ys))
            if best_y1 is None or y1 > best_y1:
                best_y1 = y1
    return best_y1


def _crop_sig_box(page_img: Image.Image, y_top: int, y_bottom: int,
                  x_left: int, x_right: int,
                  label_ratio: Optional[float] = None) -> tuple[Optional[str], Optional[float]]:
    """
    Crop a signature cell. Returns (b64_or_None, label_ratio).
    label_ratio is label_y1 / cell_height — reuse across paired cells so both
    lần 1 and lần 2 share the same vertical offset even if OCR only detects
    the label in one of them.
    """
    img_w, img_h = page_img.size
    x_left   = max(0, x_left)
    x_right  = min(img_w, x_right)
    y_top    = max(0, y_top)
    y_bottom = min(img_h, y_bottom)
    cell = page_img.crop((x_left, y_top, x_right, y_bottom))

    if _is_blank_cell(cell) or _is_slash_cell(cell):
        return None, label_ratio

    cell_h = y_bottom - y_top

    # Try to detect the label; fall back to caller-supplied ratio if unavailable
    found_label_y1 = _find_ky_label_bottom(cell)
    if found_label_y1 is not None:
        label_ratio = found_label_y1 / cell_h if cell_h > 0 else label_ratio

    if label_ratio is not None and cell_h > 0:
        sig_y_top = y_top + int(label_ratio * cell_h) + 4
        sig_y_top = min(sig_y_top, y_bottom - 10)
        cell = page_img.crop((x_left, sig_y_top, x_right, y_bottom))

    tight = _tight_crop(cell, padding=15)
    final = tight if tight is not None else cell
    return _pil_to_b64(final), label_ratio


def _detect_sig_cells(page_img: Image.Image) -> list[tuple[int, int, int, int]]:
    """
    Detect signature cell bounding boxes by finding table borders with OpenCV.
    Returns list of (x0, y0, x1, y1) sorted top-to-bottom, left-to-right.
    Only keeps cells large enough to be signature boxes (not header/text rows).
    """
    gray = cv2.cvtColor(np.array(page_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Binarize (dark lines on white background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines (width >= 8% of page width)
    min_h_len = w // 12
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Detect vertical lines (height >= 3% of page height)
    min_v_len = h // 30
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Combine into a grid mask and dilate slightly to close small gaps
    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    # Find contours of closed cells
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum cell size: at least 15% page width × 4% page height
    min_cell_w = int(w * 0.15)
    min_cell_h = int(h * 0.04)

    cells = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw >= min_cell_w and ch >= min_cell_h:
            cells.append((x, y, x + cw, y + ch))

    # Deduplicate near-identical boxes (keep largest when overlap > 80%)
    cells.sort(key=lambda c: (c[1], c[0]))
    filtered = []
    for cell in cells:
        x0, y0, x1, y1 = cell
        duplicate = False
        for fx0, fy0, fx1, fy1 in filtered:
            ix0, iy0 = max(x0, fx0), max(y0, fy0)
            ix1, iy1 = min(x1, fx1), min(y1, fy1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                area = (x1 - x0) * (y1 - y0)
                if inter / area > 0.80:
                    duplicate = True
                    break
        if not duplicate:
            filtered.append(cell)

    return filtered


def _extract_signatures_section_c(page_img: Image.Image) -> dict:
    """
    Crop signature cells from Section C using detected table borders.
    The form has a fixed 4-row × 2-column grid (người chịu trách nhiệm + 3 ủy quyền).

    Falls back to OCR-based y-range detection if grid is not found.
    """
    GROUP_KEYS = [
        "nguoi_chiu_trach_nhiem",
        "uy_quyen_1",
        "uy_quyen_2",
        "uy_quyen_3",
    ]

    cells = _detect_sig_cells(page_img)
    w, h = page_img.size

    # Expect 8 cells (4 rows × 2 cols); accept 6–8
    if len(cells) >= 6:
        # Sort: group into rows by y0, then left/right within each row
        cells.sort(key=lambda c: (c[1], c[0]))

        # Cluster into rows: cells whose y0 are within 20px of each other → same row
        rows: list[list[tuple]] = []
        for cell in cells:
            placed = False
            for row in rows:
                if abs(cell[1] - row[0][1]) <= 20:
                    row.append(cell)
                    placed = True
                    break
            if not placed:
                rows.append([cell])

        # Sort each row left-to-right, keep at most 2 cells per row
        rows = [sorted(row, key=lambda c: c[0])[:2] for row in rows]
        # Keep only rows that have 2 cells (both signature columns present)
        rows = [row for row in rows if len(row) == 2]
        rows = rows[:4]  # at most 4 groups

        results: dict = {}
        for i, key in enumerate(GROUP_KEYS):
            if i >= len(rows):
                results[key] = {"chu_ky_lan_1": None, "chu_ky_lan_2": None}
                continue
            left_cell, right_cell = rows[i]
            # Process lần 1 first; share its label_ratio with lần 2
            b64_left,  ratio = _crop_sig_box(page_img, *_cell_inner(left_cell))
            b64_right, _     = _crop_sig_box(page_img, *_cell_inner(right_cell), label_ratio=ratio)
            results[key] = {
                "chu_ky_lan_1": {"image_b64": b64_left,  "mime": "image/png"} if b64_left  else None,
                "chu_ky_lan_2": {"image_b64": b64_right, "mime": "image/png"} if b64_right else None,
            }
        return results

    # ── Fallback: use OCR header positions ────────────────────────────────────
    logger.warning("Section C: grid detection found %d cells, falling back to OCR", len(cells))
    blocks = _ocr_blocks(page_img)
    group_keywords = [
        ("nguoi_chiu_trach_nhiem", ["chịu trách nhiệm", "chiu trach nhiem"]),
        ("uy_quyen_1", ["ủy quyền thứ 1", "uy quyen thu 1"]),
        ("uy_quyen_2", ["ủy quyền thứ 2", "uy quyen thu 2"]),
        ("uy_quyen_3", ["ủy quyền thứ 3", "uy quyen thu 3"]),
    ]
    group_y: dict[str, int] = {}
    for key, kws in group_keywords:
        for kw in kws:
            b = _find_block(blocks, kw)
            if b:
                group_y[key] = b["y0"]
                break
    if not group_y:
        logger.warning("Section C: no group headers found")
        return {}
    y_positions = sorted(group_y.items(), key=lambda t: t[1])
    half_w = w // 2
    results = {}
    for i, (key, y_top) in enumerate(y_positions):
        y_bottom = y_positions[i + 1][1] if i + 1 < len(y_positions) else min(h, y_top + int(h * 0.20))
        sig_y_top = y_top + 30
        b64_left,  ratio = _crop_sig_box(page_img, sig_y_top, y_bottom, 0,      half_w)
        b64_right, _     = _crop_sig_box(page_img, sig_y_top, y_bottom, half_w, w, label_ratio=ratio)
        results[key] = {
            "chu_ky_lan_1": {"image_b64": b64_left,  "mime": "image/png"} if b64_left  else None,
            "chu_ky_lan_2": {"image_b64": b64_right, "mime": "image/png"} if b64_right else None,
        }
    return results


def _cell_inner(cell: tuple[int, int, int, int], margin: int = 6) -> tuple[int, int, int, int]:
    """Return (y_top, y_bottom, x_left, x_right) with a small inset to exclude borders."""
    x0, y0, x1, y1 = cell
    return y0 + margin, y1 - margin, x0 + margin, x1 - margin


# ── Main extraction entry point ────────────────────────────────────────────────

def extract_c4_form(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)

    if n_pages < 1:
        raise ValueError("PDF rỗng.")

    page1_img = _page_to_pil(doc[0])
    customer_info = _extract_customer_info(page1_img)

    # Section C is on the last page (page 2 for standard C4 form)
    sig_page_img = _page_to_pil(doc[n_pages - 1])
    signatures = _extract_signatures_section_c(sig_page_img)

    doc.close()
    return {"customer_info": customer_info, "signatures": signatures}


# ── FastAPI endpoint ───────────────────────────────────────────────────────────

@router.post(
    "/extract-c4-form",
    summary="Trích xuất thông tin khách hàng và chữ ký từ Phiếu Thông Tin Khách Hàng (Form C4)",
    response_class=JSONResponse,
)
async def extract_c4_form_endpoint(
    file: UploadFile = File(..., description="File PDF Phiếu Thông Tin Khách Hàng (Form C4)"),
):
    is_pdf = file.content_type == "application/pdf" or (
        (file.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF.")

    pdf_bytes = await file.read()
    try:
        result = extract_c4_form(pdf_bytes)
    except Exception as exc:
        logger.exception("C4 form processing error")
        raise HTTPException(status_code=422, detail=f"Lỗi xử lý PDF: {exc}")

    return result
