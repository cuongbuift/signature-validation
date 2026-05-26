from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Optional, List

import cv2
from PIL import Image
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy.orm import Session

import re

import fitz

from database import get_db
from detectors import detector
from models import CustomerRecord
from routers.extract_c4_form import extract_c4_form
from routers.extract_signatures import extract_signatures_from_pdf
from validators import SignatureValidator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/c4-customers", tags=["C4 Customers"])

# Storage directories
STORAGE = Path("storage")
PDF_DIR = STORAGE / "c4_pdfs"
SIG_DIR = STORAGE / "c4_signatures"
PDF_DIR.mkdir(parents=True, exist_ok=True)
SIG_DIR.mkdir(parents=True, exist_ok=True)

# Signature keys and their field names on the model
SIG_KEYS = [
    "sig_ct_lan1", "sig_ct_lan2",
    "sig_uq1_lan1", "sig_uq1_lan2",
    "sig_uq2_lan1", "sig_uq2_lan2",
    "sig_uq3_lan1", "sig_uq3_lan2",
]

# Map from extraction response keys to model field names
_SIG_MAP = {
    ("nguoi_chiu_trach_nhiem", "chu_ky_lan_1"): "sig_ct_lan1",
    ("nguoi_chiu_trach_nhiem", "chu_ky_lan_2"): "sig_ct_lan2",
    ("uy_quyen_1",              "chu_ky_lan_1"): "sig_uq1_lan1",
    ("uy_quyen_1",              "chu_ky_lan_2"): "sig_uq1_lan2",
    ("uy_quyen_2",              "chu_ky_lan_1"): "sig_uq2_lan1",
    ("uy_quyen_2",              "chu_ky_lan_2"): "sig_uq2_lan2",
    ("uy_quyen_3",              "chu_ky_lan_1"): "sig_uq3_lan1",
    ("uy_quyen_3",              "chu_ky_lan_2"): "sig_uq3_lan2",
}


def _record_to_dict(rec: CustomerRecord) -> dict:
    return {
        "id": rec.id,
        "original_filename": rec.original_filename,
        "ma_khach_hang": rec.ma_khach_hang,
        "ten_dang_ky_kinh_doanh": rec.ten_dang_ky_kinh_doanh,
        "ten_cua_hang": rec.ten_cua_hang,
        "giay_phep_so": rec.giay_phep_so,
        "giay_phep_ngay_cap": rec.giay_phep_ngay_cap,
        "giay_phep_noi_cap": rec.giay_phep_noi_cap,
        "dia_chi_kinh_doanh": rec.dia_chi_kinh_doanh,
        "signatures": {k: (getattr(rec, k) is not None) for k in SIG_KEYS},
        "created_at": rec.created_at.isoformat(),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", summary="Upload PDF Form C4, trích xuất và lưu thông tin khách hàng")
async def create_customer(
    file: UploadFile = File(...),
    ma_khach_hang: str = Form(""),
    ten_dang_ky_kinh_doanh: str = Form(""),
    ten_cua_hang: str = Form(""),
    giay_phep_so: str = Form(""),
    giay_phep_ngay_cap: str = Form(""),
    giay_phep_noi_cap: str = Form(""),
    dia_chi_kinh_doanh: str = Form(""),
    db: Session = Depends(get_db),
):
    is_pdf = file.content_type == "application/pdf" or (
        (file.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF.")

    pdf_bytes = await file.read()

    # Extract from PDF
    try:
        extracted = extract_c4_form(pdf_bytes)
    except Exception as exc:
        logger.exception("C4 extract error")
        raise HTTPException(status_code=422, detail=f"Lỗi trích xuất PDF: {exc}")

    # Create DB record first to get ID for file paths
    rec = CustomerRecord(
        pdf_path="",
        original_filename=file.filename or "form.pdf",
        ma_khach_hang=ma_khach_hang or extracted["customer_info"].get("ma_khach_hang", ""),
        ten_dang_ky_kinh_doanh=ten_dang_ky_kinh_doanh or extracted["customer_info"].get("ten_dang_ky_kinh_doanh", ""),
        ten_cua_hang=ten_cua_hang or extracted["customer_info"].get("ten_cua_hang", ""),
        giay_phep_so=giay_phep_so or extracted["customer_info"].get("giay_phep_kinh_doanh", {}).get("so", ""),
        giay_phep_ngay_cap=giay_phep_ngay_cap or extracted["customer_info"].get("giay_phep_kinh_doanh", {}).get("ngay_cap", ""),
        giay_phep_noi_cap=giay_phep_noi_cap or extracted["customer_info"].get("giay_phep_kinh_doanh", {}).get("noi_cap", ""),
        dia_chi_kinh_doanh=dia_chi_kinh_doanh or extracted["customer_info"].get("dia_chi_kinh_doanh", ""),
    )
    db.add(rec)
    db.flush()  # get rec.id

    # Save PDF
    pdf_path = PDF_DIR / f"{rec.id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    rec.pdf_path = str(pdf_path)

    # Save signature images
    sigs = extracted.get("signatures", {})
    for (group_key, sig_key), field in _SIG_MAP.items():
        group = sigs.get(group_key)
        if not group:
            continue
        sig_data = group.get(sig_key)
        if not sig_data or not sig_data.get("image_b64"):
            continue
        img_path = SIG_DIR / f"{rec.id}_{field}.png"
        img_path.write_bytes(base64.b64decode(sig_data["image_b64"]))
        setattr(rec, field, str(img_path))

    db.commit()
    db.refresh(rec)
    return _record_to_dict(rec)


@router.get("", summary="Danh sách khách hàng C4")
def list_customers(
    skip: int = 0,
    limit: int = 200,
    db: Session = Depends(get_db),
):
    records = db.query(CustomerRecord).order_by(CustomerRecord.id.desc()).offset(skip).limit(limit).all()
    return [_record_to_dict(r) for r in records]


@router.get("/{record_id}", summary="Chi tiết khách hàng C4")
def get_customer(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    return _record_to_dict(rec)


@router.put("/{record_id}", summary="Cập nhật thông tin khách hàng C4")
def update_customer(
    record_id: int,
    ma_khach_hang: Optional[str] = None,
    ten_dang_ky_kinh_doanh: Optional[str] = None,
    ten_cua_hang: Optional[str] = None,
    giay_phep_so: Optional[str] = None,
    giay_phep_ngay_cap: Optional[str] = None,
    giay_phep_noi_cap: Optional[str] = None,
    dia_chi_kinh_doanh: Optional[str] = None,
    db: Session = Depends(get_db),
):
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    if ma_khach_hang is not None:
        rec.ma_khach_hang = ma_khach_hang
    if ten_dang_ky_kinh_doanh is not None:
        rec.ten_dang_ky_kinh_doanh = ten_dang_ky_kinh_doanh
    if ten_cua_hang is not None:
        rec.ten_cua_hang = ten_cua_hang
    if giay_phep_so is not None:
        rec.giay_phep_so = giay_phep_so
    if giay_phep_ngay_cap is not None:
        rec.giay_phep_ngay_cap = giay_phep_ngay_cap
    if giay_phep_noi_cap is not None:
        rec.giay_phep_noi_cap = giay_phep_noi_cap
    if dia_chi_kinh_doanh is not None:
        rec.dia_chi_kinh_doanh = dia_chi_kinh_doanh
    db.commit()
    db.refresh(rec)
    return _record_to_dict(rec)


@router.delete("/{record_id}", status_code=204, summary="Xóa khách hàng C4")
def delete_customer(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    # Delete files
    for path_field in ["pdf_path"] + SIG_KEYS:
        p = getattr(rec, path_field, None)
        if p:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
    db.delete(rec)
    db.commit()


@router.get("/{record_id}/pdf", summary="Tải file PDF gốc")
def get_pdf(record_id: int, db: Session = Depends(get_db)):
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    if not rec.pdf_path or not Path(rec.pdf_path).exists():
        raise HTTPException(status_code=404, detail="File PDF không tồn tại.")
    return FileResponse(rec.pdf_path, media_type="application/pdf",
                        filename=rec.original_filename)


@router.get("/{record_id}/signatures/{sig_key}", summary="Lấy ảnh chữ ký")
def get_signature_image(record_id: int, sig_key: str, db: Session = Depends(get_db)):
    if sig_key not in SIG_KEYS:
        raise HTTPException(status_code=400, detail=f"sig_key không hợp lệ: {sig_key}")
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")
    path = getattr(rec, sig_key, None)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Chữ ký không tồn tại.")
    return FileResponse(path, media_type="image/png")


_SIG_LABELS = {
    "sig_ct_lan1":  "CT — Lần 1",
    "sig_ct_lan2":  "CT — Lần 2",
    "sig_uq1_lan1": "UQ1 — Lần 1",
    "sig_uq1_lan2": "UQ1 — Lần 2",
    "sig_uq2_lan1": "UQ2 — Lần 1",
    "sig_uq2_lan2": "UQ2 — Lần 2",
    "sig_uq3_lan1": "UQ3 — Lần 1",
    "sig_uq3_lan2": "UQ3 — Lần 2",
}


@router.post("/{record_id}/validate-do", summary="Xác thực chữ ký từ file DO với chữ ký khách hàng")
async def validate_do(
    record_id: int,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    rec = db.get(CustomerRecord, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi.")

    customer_sigs = {
        k: getattr(rec, k)
        for k in SIG_KEYS
        if getattr(rec, k) and Path(getattr(rec, k)).exists()
    }
    if not customer_sigs:
        raise HTTPException(status_code=400, detail="Khách hàng chưa có chữ ký mẫu.")

    from config import settings

    validator = SignatureValidator(
        similarity_threshold=settings.similarity_threshold,
        siamese_weight=settings.siamese_weight,
        deep_weight=settings.deep_weight,
        ssim_weight=settings.ssim_weight,
        orb_weight=settings.orb_weight,
        contour_weight=settings.contour_weight,
    )

    tmp_dir = STORAGE / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dos_results = []
    any_do_valid = False

    for upload_file in files:
        is_pdf = upload_file.content_type == "application/pdf" or (
            (upload_file.filename or "").lower().endswith(".pdf")
        )
        if not is_pdf:
            dos_results.append({
                "filename": upload_file.filename or "unknown",
                "error": "Chỉ hỗ trợ file PDF.",
                "extracted_count": 0,
                "extracted_signatures": [],
                "is_valid": False,
            })
            continue

        pdf_bytes = await upload_file.read()
        try:
            extracted = extract_signatures_from_pdf(pdf_bytes)
        except Exception as exc:
            logger.exception("DO extract error")
            dos_results.append({
                "filename": upload_file.filename or "unknown.pdf",
                "error": f"Lỗi trích xuất: {exc}",
                "extracted_count": 0,
                "extracted_signatures": [],
                "is_valid": False,
            })
            continue

        do_sigs_results = []
        do_valid = False

        for sig_item in extracted:
            img_bytes = base64.b64decode(sig_item["image_b64"])
            tmp_path = tmp_dir / f"do_sig_{uuid.uuid4().hex}.png"
            tmp_path.write_bytes(img_bytes)

            try:
                cropped = detector.detect_and_crop(tmp_path)
                if cropped is not None:
                    cv2.imwrite(str(tmp_path), cropped)

                comparisons = []
                sig_valid = False
                best_score = 0.0

                for sig_key, sig_path in customer_sigs.items():
                    try:
                        result = validator.validate(
                            input_path=tmp_path,
                            reference_paths=[sig_path],
                        )
                        if result.is_valid:
                            sig_valid = True
                            do_valid = True
                            any_do_valid = True
                        best_score = max(best_score, result.overall_score)
                        comparisons.append({
                            "sig_key": sig_key,
                            "sig_label": _SIG_LABELS.get(sig_key, sig_key),
                            "is_valid": result.is_valid,
                            "overall_score": round(result.overall_score, 4),
                            "siamese_score": round(result.siamese_score, 4),
                            "deep_score": round(result.deep_score, 4),
                            "ssim_score": round(result.ssim_score, 4),
                            "orb_score": round(result.orb_score, 4),
                            "contour_score": round(result.contour_score, 4),
                        })
                    except Exception as exc:
                        logger.exception(f"Validation error for {sig_key}")
                        comparisons.append({
                            "sig_key": sig_key,
                            "sig_label": _SIG_LABELS.get(sig_key, sig_key),
                            "is_valid": False,
                            "overall_score": 0.0,
                            "error": str(exc),
                        })

                do_sigs_results.append({
                    "page": sig_item["page"],
                    "image_b64": sig_item["image_b64"],
                    "comparisons": comparisons,
                    "best_score": round(best_score, 4),
                    "is_valid": sig_valid,
                })
            finally:
                tmp_path.unlink(missing_ok=True)

        dos_results.append({
            "filename": upload_file.filename or "unknown.pdf",
            "extracted_count": len(do_sigs_results),
            "extracted_signatures": do_sigs_results,
            "is_valid": do_valid,
        })

    return {
        "customer_id": record_id,
        "do_valid": any_do_valid,
        "threshold": settings.similarity_threshold,
        "dos": dos_results,
    }


_OCR_DIGIT_FIX = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1', '/': '0', ' ': '', '\t': ''})


def _fix_ocr_digits(text: str) -> str:
    """Fix common OCR misreads in what should be an all-digit string."""
    return text.translate(_OCR_DIGIT_FIX)


def _extract_ma_kh_from_do(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Extract (ma_khach_hang, do_so) from a DO PDF using OCR on the top-left strip.
    Returns (ma_kh, do_so) — either may be "" if not found.
    """
    from routers.extract_signatures import _get_ocr_reader, SCALE
    import numpy as np

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat)
    full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    w, h = full_img.size
    topleft  = full_img.crop((0, 0, int(w * 0.55), int(h * 0.08)))
    topright = full_img.crop((int(w * 0.50), 0, w, int(h * 0.08)))

    reader = _get_ocr_reader()

    ma_kh = ""
    for (_, text, _conf) in reader.readtext(np.array(topleft), detail=1, paragraph=False):
        # Apply digit-fixing before matching so OCR glitches like "10/094989" → "100094989"
        candidate = _fix_ocr_digits(text.strip())
        m = re.match(r"(\d{6,10})", candidate)
        if m:
            ma_kh = m.group(1)
            break

    do_so = ""
    for (_, text, _conf) in reader.readtext(np.array(topright), detail=1, paragraph=False):
        m = re.search(r"DO\s*[Ss]ố\s*[:\-]?\s*(\S+)", text) or re.search(r"(\d[A-Z]{2}\d{4}-\d{5})", text)
        if m:
            do_so = m.group(1)
            break

    return ma_kh, do_so


@router.post("/check-do-batch", summary="Kiểm tra DO hàng loạt — tự nhận diện khách hàng từ mã KH trên DO")
async def check_do_batch(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    from config import settings

    validator = SignatureValidator(
        similarity_threshold=settings.similarity_threshold,
        siamese_weight=settings.siamese_weight,
        deep_weight=settings.deep_weight,
        ssim_weight=settings.ssim_weight,
        orb_weight=settings.orb_weight,
        contour_weight=settings.contour_weight,
    )

    tmp_dir = STORAGE / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for upload_file in files:
        filename = upload_file.filename or "unknown.pdf"
        is_pdf = upload_file.content_type == "application/pdf" or filename.lower().endswith(".pdf")
        if not is_pdf:
            results.append({"filename": filename, "error": "Chỉ hỗ trợ file PDF.", "is_valid": False})
            continue

        pdf_bytes = await upload_file.read()

        # 1. Identify customer from mã KH (OCR on top-left strip)
        ma_kh, do_so = _extract_ma_kh_from_do(pdf_bytes)

        rec: CustomerRecord | None = None
        if ma_kh:
            rec = db.query(CustomerRecord).filter(CustomerRecord.ma_khach_hang == ma_kh).first()

        base_info = {
            "filename": filename,
            "do_so": do_so,
            "ma_khach_hang": ma_kh,
            "customer_found": rec is not None,
            "customer_id": rec.id if rec else None,
            "customer_name": (rec.ten_dang_ky_kinh_doanh or rec.ten_cua_hang or "") if rec else "",
        }

        if not ma_kh:
            results.append({**base_info, "error": "Không trích xuất được mã khách hàng từ DO.", "is_valid": False, "extracted_signatures": []})
            continue

        if not rec:
            results.append({**base_info, "error": f"Không tìm thấy khách hàng với mã '{ma_kh}' trong hệ thống.", "is_valid": False, "extracted_signatures": []})
            continue

        customer_sigs = {
            k: getattr(rec, k) for k in SIG_KEYS
            if getattr(rec, k) and Path(getattr(rec, k)).exists()
        }
        if not customer_sigs:
            results.append({**base_info, "error": "Khách hàng chưa có chữ ký mẫu.", "is_valid": False, "extracted_signatures": []})
            continue

        # 2. Extract signatures from DO
        try:
            extracted = extract_signatures_from_pdf(pdf_bytes)
        except Exception as exc:
            logger.exception("DO extract error")
            results.append({**base_info, "error": f"Lỗi trích xuất chữ ký: {exc}", "is_valid": False, "extracted_signatures": []})
            continue

        do_sigs_results = []
        do_valid = False

        for sig_item in extracted:
            img_bytes = base64.b64decode(sig_item["image_b64"])
            tmp_path = tmp_dir / f"do_chk_{uuid.uuid4().hex}.png"
            tmp_path.write_bytes(img_bytes)

            try:
                cropped = detector.detect_and_crop(tmp_path)
                if cropped is not None:
                    cv2.imwrite(str(tmp_path), cropped)

                comparisons = []
                sig_valid = False
                best_score = 0.0

                for sig_key, sig_path in customer_sigs.items():
                    try:
                        result = validator.validate(input_path=tmp_path, reference_paths=[sig_path])
                        if result.is_valid:
                            sig_valid = True
                            do_valid = True
                        best_score = max(best_score, result.overall_score)
                        comparisons.append({
                            "sig_key": sig_key,
                            "sig_label": _SIG_LABELS.get(sig_key, sig_key),
                            "is_valid": result.is_valid,
                            "overall_score": round(result.overall_score, 4),
                            "siamese_score": round(result.siamese_score, 4),
                            "deep_score": round(result.deep_score, 4),
                            "ssim_score": round(result.ssim_score, 4),
                        })
                    except Exception as exc:
                        logger.exception(f"Validation error for {sig_key}")
                        comparisons.append({
                            "sig_key": sig_key,
                            "sig_label": _SIG_LABELS.get(sig_key, sig_key),
                            "is_valid": False,
                            "overall_score": 0.0,
                            "error": str(exc),
                        })

                do_sigs_results.append({
                    "page": sig_item["page"],
                    "image_b64": sig_item["image_b64"],
                    "comparisons": comparisons,
                    "best_score": round(best_score, 4),
                    "is_valid": sig_valid,
                })
            finally:
                tmp_path.unlink(missing_ok=True)

        results.append({
            **base_info,
            "extracted_count": len(do_sigs_results),
            "extracted_signatures": do_sigs_results,
            "is_valid": do_valid,
        })

    overall_valid = any(r.get("is_valid") for r in results)
    return {
        "overall_valid": overall_valid,
        "threshold": settings.similarity_threshold,
        "results": results,
    }


@router.post("/extract-preview", summary="Chỉ trích xuất (không lưu) để xem trước")
async def extract_preview(file: UploadFile = File(...)):
    is_pdf = file.content_type == "application/pdf" or (
        (file.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF.")
    pdf_bytes = await file.read()
    try:
        result = extract_c4_form(pdf_bytes)
    except Exception as exc:
        logger.exception("C4 preview extract error")
        raise HTTPException(status_code=422, detail=f"Lỗi trích xuất PDF: {exc}")
    return result
