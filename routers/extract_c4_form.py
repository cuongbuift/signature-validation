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
    _apply_handwriting_separation,
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
    print("[DEBUG] All OCR blocks (page 1):")
    for bl in blocks:
        print(f"[DEBUG]   y0={bl['y0']:4d} text='{bl['text']}'  ascii='{bl['ascii']}'")

    def find(kw, ascii_kw=None):
        return _find_block(blocks, kw, ascii_kw)

    # 1. Tên đăng ký kinh doanh (C4) / Tên đơn vị kinh doanh (C2)
    b = find("đăng ký kinh doanh", "dang ky kinh doanh") \
        or find("đơn vị kinh doanh", "don vi kinh doanh")
    ten_dkk = _value_right_of(blocks, b) if b else ""

    # 1.1. Tên cửa hàng — OCR may split the label into fragments across blocks,
    # so we find the "2. Tên" label block then take whatever follows the last ":" on that line.
    b = (find("tên cửa hàng", "ten cua hang")
         or find("cửa hàng", "cua hang")
         or find("2. tên", "2. ten"))
    if b:
        same_line = [bl for bl in blocks
                     if abs(bl["y0"] - b["y0"]) <= 15 and bl["x0"] > b["x0"]]
        same_line.sort(key=lambda bl: bl["x0"])
        # Find the rightmost block that ends with ":" — that's the last label fragment
        colon_blocks = [bl for bl in same_line if bl["text"].rstrip().endswith(":")]
        if colon_blocks:
            last_colon = max(colon_blocks, key=lambda bl: bl["x0"])
            ten_cua_hang = " ".join(
                bl["text"] for bl in same_line if bl["x0"] > last_colon["x1"]
            ).strip()
        else:
            ten_cua_hang = " ".join(bl["text"] for bl in same_line).strip()
    else:
        ten_cua_hang = ""

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
            part0 = ngay_m[0].strip().strip(":")
            nums = re.findall(r"\b\d{6,}\b", part0)
            gp_number = nums[0] if nums else part0
            rest = ngay_m[1]
            noi_m = re.split(r"nơi\s*cấp|noi\s*cap", rest, flags=re.IGNORECASE)
            if len(noi_m) >= 2:
                gp_ngay_cap = noi_m[0].strip().strip(":")
                gp_noi_cap = noi_m[1].strip().strip(":")
            else:
                gp_ngay_cap = rest.strip().strip(":")
        else:
            # Extract only the numeric GP number from raw (ignore any label fragments)
            nums = re.findall(r"\b\d{6,}\b", raw)
            gp_number = nums[0] if nums else raw.strip()

        # C2: number may be embedded in the label block itself (e.g. "Giấy phép … 0318996463")
        if not gp_number:
            nums = re.findall(r"\b\d{6,}\b", b["text"])
            if nums:
                gp_number = nums[-1]

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

    # 4. Mã khách hàng (Search key) — top-right box of page 1
    ma_khach_hang = ""
    b = find("mã khách hàng", "ma khach hang") or find("search key", "search key")
    if b:
        # Value is to the right of the label on the same line
        val_right = _value_right_of(blocks, b)
        if val_right:
            nums = re.findall(r"\b\d{6,}\b", val_right)
            ma_khach_hang = nums[0] if nums else val_right.strip()
        if not ma_khach_hang:
            # Sometimes the value is on the next block below
            val_below = _value_below(blocks, b, (b["x0"], b["x1"] + 300), max_dy=40)
            nums = re.findall(r"\b\d{6,}\b", val_below)
            ma_khach_hang = nums[0] if nums else val_below.strip()
        if not ma_khach_hang:
            # The numeric value may sit on the "(Search key)" sub-line which is slightly below
            val_right_loose = " ".join(
                bl["text"] for bl in blocks
                if bl["x0"] > b["x1"] - 10
                and b["y0"] - 20 <= bl["y0"] <= b["y1"] + 60
            )
            nums = re.findall(r"\b\d{6,}\b", val_right_loose)
            ma_khach_hang = nums[0] if nums else ""

    # Fallback: scan top-right quadrant (top 15% height, right 45% width) for a standalone number
    if not ma_khach_hang:
        top_right = [
            bl for bl in blocks
            if bl["x0"] > w * 0.55 and bl["y1"] < _h * 0.15
        ]
        for bl in top_right:
            nums = re.findall(r"\b\d{6,}\b", bl["text"])
            if nums:
                ma_khach_hang = nums[0]
                break

    return {
        "ten_dang_ky_kinh_doanh": ten_dkk,
        "ten_cua_hang": ten_cua_hang,
        "giay_phep_kinh_doanh": {
            "so": gp_number,
            "ngay_cap": gp_ngay_cap,
            "noi_cap": gp_noi_cap,
        },
        "dia_chi_kinh_doanh": dia_chi,
        "ma_khach_hang": ma_khach_hang,
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

    cell_h = y_bottom - y_top

    # Try to detect the label; fall back to caller-supplied ratio if unavailable
    found_label_y1 = _find_ky_label_bottom(cell)
    if found_label_y1 is not None:
        label_ratio = found_label_y1 / cell_h if cell_h > 0 else label_ratio

    # Determine the signature sub-area (below the label line)
    if label_ratio is not None and cell_h > 0:
        sig_y_top = y_top + int(label_ratio * cell_h)
        sig_y_top = min(sig_y_top, y_bottom - 10)
    else:
        sig_y_top = y_top

    sig_area = page_img.crop((x_left, sig_y_top - 30, x_right, y_bottom + 30))

    # Check blank/slash only on the signature sub-area (excludes label text)
    if _is_blank_cell(sig_area) or _is_slash_cell(sig_area):
        return None, label_ratio

    tight = _tight_crop(sig_area, padding=15)
    final = tight if tight is not None else sig_area

    # Remove printed text and ruling lines, keep only handwriting/signature
    final = _apply_handwriting_separation(final)

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

    # Cell size bounds: signature cells are 15–52% page width, 4–30% page height
    min_cell_w = int(w * 0.15)
    max_cell_w = int(w * 0.52)
    min_cell_h = int(h * 0.04)
    max_cell_h = int(h * 0.30)
    # "wide" cells: full-row cells where the vertical divider was missed
    wide_cell_w_min = int(w * 0.60)

    cells = []
    wide_cells = []   # full-width cells to be split later
    rejected = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if min_cell_w <= cw <= max_cell_w and min_cell_h <= ch <= max_cell_h:
            cells.append((x, y, x + cw, y + ch))
        elif wide_cell_w_min <= cw and min_cell_h <= ch <= max_cell_h:
            wide_cells.append((x, y, x + cw, y + ch))
        else:
            rejected.append((x, y, x + cw, y + ch, cw, ch,
                             f"w={cw/w:.2%} h={ch/h:.2%}"))
    print(f"[DEBUG] _detect_sig_cells: {len(cells)} normal, {len(wide_cells)} wide, {len(rejected)} rejected")

    # Deduplicate: sort smallest-area first so inner cells are kept over outer borders.
    # If two boxes overlap >80% of the smaller one's area, the larger is dropped.
    cells.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    filtered: list[tuple] = []
    for cell in cells:
        x0, y0, x1, y1 = cell
        area = (x1 - x0) * (y1 - y0)
        skip = False
        for fc in filtered:
            fx0, fy0, fx1, fy1 = fc
            ix0, iy0 = max(x0, fx0), max(y0, fy0)
            ix1, iy1 = min(x1, fx1), min(y1, fy1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                f_area = (fx1 - fx0) * (fy1 - fy0)
                if inter / min(area, f_area) > 0.80:
                    skip = True
                    break
        if not skip:
            filtered.append(cell)

    # If wide cells exist, infer the vertical divider x from normal cells and split them
    if wide_cells and filtered:
        # Collect left-column and right-column x bounds from normal cells
        left_x1s  = [c[2] for c in filtered if (c[0] + c[2]) // 2 < w // 2]
        right_x0s = [c[0] for c in filtered if (c[0] + c[2]) // 2 >= w // 2]
        if left_x1s and right_x0s:
            divider = (int(np.median(left_x1s)) + int(np.median(right_x0s))) // 2
        elif left_x1s:
            divider = int(np.median(left_x1s))
        elif right_x0s:
            divider = int(np.median(right_x0s))
        else:
            divider = w // 2
        print(f"[DEBUG] inferred vertical divider x={divider}, splitting {len(wide_cells)} wide cells")
        def _overlaps_existing(cell, existing, threshold=0.60):
            x0, y0, x1, y1 = cell
            area = (x1 - x0) * (y1 - y0)
            if area == 0:
                return True
            for fc in existing:
                fx0, fy0, fx1, fy1 = fc
                ix0, iy0 = max(x0, fx0), max(y0, fy0)
                ix1, iy1 = min(x1, fx1), min(y1, fy1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    if inter / area > threshold:
                        return True
            return False

        for wc in wide_cells:
            wx0, wy0, wx1, wy1 = wc
            left_half  = (wx0, wy0, divider, wy1)
            right_half = (divider, wy0, wx1, wy1)
            if not _overlaps_existing(left_half, filtered):
                filtered.append(left_half)
            if not _overlaps_existing(right_half, filtered):
                filtered.append(right_half)

    print(f"[DEBUG] _detect_sig_cells final: {len(filtered)} cells (page {w}x{h})")
    for c in filtered:
        print(f"[DEBUG]   cell y0={c[1]} y1={c[3]} x0={c[0]} x1={c[2]}  w={(c[2]-c[0])/w*100:.1f}% h={(c[3]-c[1])/h*100:.1f}%")
    return filtered


def _extract_signatures_section_c(page_img: Image.Image) -> dict:
    """
    Crop signature cells from Section C using detected table borders.
    The form has a fixed 4-row × 2-column grid (người chịu trách nhiệm + 3 ủy quyền).
    """
    GROUP_KEYS = [
        "nguoi_chiu_trach_nhiem",
        "uy_quyen_1",
        "uy_quyen_2",
        "uy_quyen_3",
    ]

    cells = _detect_sig_cells(page_img)
    w, h = page_img.size
    logger.info("Section C: detected %d grid cells", len(cells))

    if len(cells) >= 6:
        # Sort top-to-bottom, left-to-right
        cells.sort(key=lambda c: (c[1], c[0]))

        # Cluster into rows: cells whose y0 are within 30px of each other
        rows: list[list[tuple]] = []
        for cell in cells:
            placed = False
            for row in rows:
                if abs(cell[1] - row[0][1]) <= 30:
                    row.append(cell)
                    placed = True
                    break
            if not placed:
                rows.append([cell])

        print(f"[DEBUG] Section C: {len(rows)} raw rows after clustering:")
        for ri, row in enumerate(rows):
            print(f"[DEBUG]   row {ri}: {len(row)} cells, y0s={[c[1] for c in row]}")

        # Sort cells within each row left-to-right
        rows = [sorted(row, key=lambda c: c[0]) for row in rows]

        # Infer column bounds from rows that have 2 cells
        complete_rows = [r for r in rows if len(r) >= 2]
        if complete_rows:
            left_x0  = int(np.median([r[0][0] for r in complete_rows]))
            left_x1  = int(np.median([r[0][2] for r in complete_rows]))
            right_x0 = int(np.median([r[1][0] for r in complete_rows]))
            right_x1 = int(np.median([r[1][2] for r in complete_rows]))
        else:
            left_x0 = left_x1 = right_x0 = right_x1 = 0

        # For rows with only 1 cell, synthesize the missing paired cell
        for row in rows:
            if len(row) == 1:
                sole = row[0]
                sole_cx = (sole[0] + sole[2]) // 2
                page_mid = w // 2
                if sole_cx < page_mid and right_x0 > 0:
                    # left cell present → synthesize right cell
                    row.append((right_x0, sole[1], right_x1, sole[3]))
                    print(f"[DEBUG]   synthesized RIGHT cell for row y0={sole[1]}")
                elif sole_cx >= page_mid and left_x0 > 0:
                    # right cell present → synthesize left cell
                    row.insert(0, (left_x0, sole[1], left_x1, sole[3]))
                    print(f"[DEBUG]   synthesized LEFT cell for row y0={sole[1]}")

        # Keep rows that have exactly 2 cells
        rows = [r[:2] for r in rows if len(r) >= 2]

        print(f"[DEBUG] Section C: {len(rows)} valid 2-cell rows detected")

        # If more than 4 rows, pick the 4 that are most likely to be signature rows
        # (tallest cells → most ink area available for signatures)
        if len(rows) > 4:
            # Section C is always in the upper portion of the signature page.
            # Filter out rows in the bottom 40% of the page (date/personnel section),
            # then keep the top 4 remaining rows.
            upper_rows = [r for r in rows if r[0][1] < h * 0.60]
            if len(upper_rows) >= 4:
                rows = upper_rows[:4]
            else:
                # Fallback: keep topmost 4 rows
                rows = rows[:4]

        # Collect best label_ratio from taller cells (more reliable detection)
        best_ratio: Optional[float] = None
        for row in rows:
            left_cell = row[0]
            cell_crop = page_img.crop((left_cell[0], left_cell[1], left_cell[2], left_cell[3]))
            label_y1 = _find_ky_label_bottom(cell_crop)
            cell_h = left_cell[3] - left_cell[1]
            if label_y1 is not None and cell_h > 0:
                best_ratio = label_y1 / cell_h
                break

        print(f"[DEBUG] best_ratio={best_ratio}, rows selected:")
        for ri, row in enumerate(rows):
            print(f"[DEBUG]   row {ri}: y0={row[0][1]} y1={row[0][3]} h={row[0][3]-row[0][1]}px")

        results: dict = {}
        for i, key in enumerate(GROUP_KEYS):
            if i >= len(rows):
                results[key] = {"chu_ky_lan_1": None, "chu_ky_lan_2": None}
                continue
            left_cell, right_cell = rows[i]
            b64_left,  ratio_left  = _crop_sig_box(page_img, *_cell_inner(left_cell),  label_ratio=best_ratio)
            b64_right, ratio_right = _crop_sig_box(page_img, *_cell_inner(right_cell), label_ratio=best_ratio)
            rl = f"{ratio_left:.2f}" if ratio_left is not None else "N/A"
            rr = f"{ratio_right:.2f}" if ratio_right is not None else "N/A"
            print(f"[DEBUG]   {key}: left={'OK' if b64_left else 'None'} ratio={rl}, right={'OK' if b64_right else 'None'} ratio={rr}")
            results[key] = {
                "chu_ky_lan_1": {"image_b64": b64_left,  "mime": "image/png"} if b64_left  else None,
                "chu_ky_lan_2": {"image_b64": b64_right, "mime": "image/png"} if b64_right else None,
            }
        return results

    logger.warning("Section C: only %d cells found, cannot extract signatures", len(cells))
    return {k: {"chu_ky_lan_1": None, "chu_ky_lan_2": None} for k in GROUP_KEYS}


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
    summary="Trích xuất thông tin khách hàng và chữ ký từ Phiếu Thông Tin Khách Hàng (Form C2 - C4)",
    response_class=JSONResponse,
)
async def extract_c4_form_endpoint(
    file: UploadFile = File(..., description="File PDF Phiếu Thông Tin Khách Hàng (Form C2 - C4)"),
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
