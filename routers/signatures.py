from __future__ import annotations
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from detectors import detector
from models import Employee, ReferenceSignature
from schemas import SignatureOut

router = APIRouter(prefix="/employees", tags=["Reference Signatures"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}


@router.post(
    "/{employee_code}/signatures",
    response_model=SignatureOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload chữ ký mẫu từ hợp đồng",
)
async def upload_reference_signature(
    employee_code: str,
    file: UploadFile = File(..., description="Ảnh chữ ký mẫu (JPEG/PNG/WebP/TIFF)"),
    contract_ref: str | None = Form(None, description="Mã hợp đồng"),
    db: Session = Depends(get_db),
):
    emp = _get_employee_or_404(employee_code, db)

    # Check limit
    existing = (
        db.query(ReferenceSignature)
        .filter(ReferenceSignature.employee_id == emp.id)
        .all()
    )
    if len(existing) >= settings.max_reference_signatures:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Nhân viên đã có {settings.max_reference_signatures} chữ ký mẫu. "
                "Xóa chữ ký cũ trước khi thêm mới."
            ),
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Định dạng không hỗ trợ: {file.content_type}. Chấp nhận: JPEG, PNG, WebP, TIFF.",
        )

    # Persist file
    dest_dir = settings.storage_dir / employee_code
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"ref_{len(existing) + 1}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = dest_dir / filename

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Detect and crop signature region with YOLO (if model is available)
    cropped = detector.detect_and_crop(dest_path)
    if cropped is not None:
        cv2.imwrite(str(dest_path), cropped)

    sig = ReferenceSignature(
        employee_id=emp.id,
        file_path=str(dest_path),
        contract_ref=contract_ref,
        order=len(existing) + 1,
    )
    db.add(sig)
    db.commit()
    db.refresh(sig)
    return sig


@router.get(
    "/{employee_code}/signatures",
    response_model=list[SignatureOut],
    summary="Danh sách chữ ký mẫu của nhân viên",
)
def list_signatures(employee_code: str, db: Session = Depends(get_db)):
    emp = _get_employee_or_404(employee_code, db)
    return emp.signatures


@router.get(
    "/{employee_code}/signatures/{signature_id}/image",
    summary="Lấy ảnh chữ ký mẫu",
    response_class=FileResponse,
)
def get_signature_image(
    employee_code: str, signature_id: int, db: Session = Depends(get_db)
):
    emp = _get_employee_or_404(employee_code, db)
    sig = (
        db.query(ReferenceSignature)
        .filter(
            ReferenceSignature.id == signature_id,
            ReferenceSignature.employee_id == emp.id,
        )
        .first()
    )
    if not sig:
        raise HTTPException(status_code=404, detail="Không tìm thấy chữ ký mẫu.")
    path = Path(sig.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File ảnh không tồn tại trên server.")
    return FileResponse(path)


@router.delete(
    "/{employee_code}/signatures/{signature_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Xóa chữ ký mẫu",
)
def delete_signature(
    employee_code: str, signature_id: int, db: Session = Depends(get_db)
):
    emp = _get_employee_or_404(employee_code, db)
    sig = (
        db.query(ReferenceSignature)
        .filter(
            ReferenceSignature.id == signature_id,
            ReferenceSignature.employee_id == emp.id,
        )
        .first()
    )
    if not sig:
        raise HTTPException(status_code=404, detail="Không tìm thấy chữ ký mẫu.")

    # Remove file from disk
    path = Path(sig.file_path)
    if path.exists():
        path.unlink()

    db.delete(sig)
    db.commit()


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_employee_or_404(employee_code: str, db: Session) -> Employee:
    emp = db.query(Employee).filter(Employee.employee_code == employee_code).first()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân viên '{employee_code}'.")
    return emp
