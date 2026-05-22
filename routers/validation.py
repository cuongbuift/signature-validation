from __future__ import annotations
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

import cv2
from config import settings
from database import get_db
from detectors import detector
from models import Employee, ReferenceSignature, ValidationRecord
from schemas import ValidationResult, ValidationRecordOut
from validators import SignatureValidator

router = APIRouter(tags=["Validation"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}


@router.post(
    "/validate",
    response_model=ValidationResult,
    summary="Xác thực chữ ký trên phiếu giao hàng",
)
async def validate_signature(
    employee_code: str = Form(..., description="Mã nhân viên"),
    file: UploadFile = File(..., description="Ảnh chữ ký trên phiếu giao hàng"),
    delivery_ref: str | None = Form(None, description="Mã phiếu giao hàng"),
    threshold: float | None = Form(
        None,
        ge=0.0,
        le=1.0,
        description="Ngưỡng chấp nhận (override config, 0.0–1.0)",
    ),
    db: Session = Depends(get_db),
):
    # 1. Get employee
    emp = db.query(Employee).filter(Employee.employee_code == employee_code).first()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân viên '{employee_code}'.")

    # 2. Get reference signatures
    refs: list[ReferenceSignature] = (
        db.query(ReferenceSignature)
        .filter(ReferenceSignature.employee_id == emp.id)
        .order_by(ReferenceSignature.order)
        .all()
    )
    if not refs:
        raise HTTPException(
            status_code=400,
            detail="Nhân viên chưa có chữ ký mẫu. Vui lòng upload chữ ký mẫu trước.",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng không hỗ trợ: {file.content_type}.",
        )

    # 3. Save input file temporarily
    tmp_dir = settings.storage_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".jpg"
    tmp_path = tmp_dir / f"input_{uuid.uuid4().hex}{ext}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 3b. Detect and crop signature region with YOLO (if model is available)
    cropped = detector.detect_and_crop(tmp_path)
    if cropped is not None:
        cv2.imwrite(str(tmp_path), cropped)

    # 4. Load validation config from settings (config.py)
    used_threshold = threshold if threshold is not None else settings.similarity_threshold

    # 5. Run validation
    validator = SignatureValidator(
        similarity_threshold=used_threshold,
        siamese_weight=settings.siamese_weight,
        deep_weight=settings.deep_weight,
        ssim_weight=settings.ssim_weight,
        orb_weight=settings.orb_weight,
        contour_weight=settings.contour_weight,
    )

    try:
        result = validator.validate(
            input_path=tmp_path,
            reference_paths=[r.file_path for r in refs],
        )
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Lỗi xử lý ảnh: {exc}")

    # 6. Persist record (keep input file for audit)
    record = ValidationRecord(
        employee_id=emp.id,
        delivery_ref=delivery_ref,
        input_file_path=str(tmp_path),
        is_valid=result.is_valid,
        overall_score=result.overall_score,
        siamese_score=result.siamese_score,
        deep_score=result.deep_score,
        ssim_score=result.ssim_score,
        orb_score=result.orb_score,
        contour_score=result.contour_score,
        threshold_used=used_threshold,
    )
    db.add(record)
    db.commit()

    # Enrich per_reference with signature_id so the UI can fetch images
    per_ref = result.detail.get("per_reference", [])
    for i, entry in enumerate(per_ref):
        if i < len(refs):
            entry["signature_id"] = refs[i].id

    return ValidationResult(
        is_valid=result.is_valid,
        overall_score=round(result.overall_score, 4),
        siamese_score=round(result.siamese_score, 4),
        deep_score=round(result.deep_score, 4),
        ssim_score=round(result.ssim_score, 4),
        orb_score=round(result.orb_score, 4),
        contour_score=round(result.contour_score, 4),
        threshold_used=used_threshold,
        employee_code=employee_code,
        delivery_ref=delivery_ref,
        detail=result.detail,
    )


@router.get(
    "/validate/history/{employee_code}",
    response_model=list[ValidationRecordOut],
    summary="Lịch sử validation của nhân viên",
)
def validation_history(
    employee_code: str,
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
):
    emp = db.query(Employee).filter(Employee.employee_code == employee_code).first()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân viên '{employee_code}'.")

    records = (
        db.query(ValidationRecord)
        .filter(ValidationRecord.employee_id == emp.id)
        .order_by(ValidationRecord.validated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return records

