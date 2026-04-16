from __future__ import annotations
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import ValidationConfig
from schemas import ConfigOut, ConfigUpdate

router = APIRouter(prefix="/config", tags=["Validation Config"])


@router.get("", response_model=ConfigOut, summary="Xem cấu hình validation hiện tại")
def get_config(db: Session = Depends(get_db)):
    return _get_or_create_default(db)


@router.put("", response_model=ConfigOut, summary="Cập nhật cấu hình validation")
def update_config(payload: ConfigUpdate, db: Session = Depends(get_db)):
    cfg = _get_or_create_default(db)

    new_siamese = payload.siamese_weight if payload.siamese_weight is not None else cfg.siamese_weight
    new_deep    = payload.deep_weight    if payload.deep_weight    is not None else cfg.deep_weight
    new_ssim    = payload.ssim_weight    if payload.ssim_weight    is not None else cfg.ssim_weight
    new_orb     = payload.orb_weight     if payload.orb_weight     is not None else cfg.orb_weight
    new_contour = payload.contour_weight if payload.contour_weight is not None else cfg.contour_weight

    # Validate weights sum ≈ 1.0
    total = new_siamese + new_deep + new_ssim + new_orb + new_contour
    if not (0.99 <= total <= 1.01):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tổng các trọng số (siamese + deep + ssim + orb + contour) phải bằng 1.0, "
                f"hiện tại = {total:.2f}."
            ),
        )

    if payload.similarity_threshold is not None:
        cfg.similarity_threshold = payload.similarity_threshold
    cfg.siamese_weight = new_siamese
    cfg.deep_weight    = new_deep
    cfg.ssim_weight    = new_ssim
    cfg.orb_weight     = new_orb
    cfg.contour_weight = new_contour
    cfg.updated_at     = datetime.utcnow()

    db.commit()
    db.refresh(cfg)
    return cfg


@router.post("/reset", response_model=ConfigOut, summary="Reset cấu hình về mặc định")
def reset_config(db: Session = Depends(get_db)):
    cfg = _get_or_create_default(db)
    cfg.similarity_threshold = settings.similarity_threshold
    cfg.siamese_weight = settings.siamese_weight
    cfg.deep_weight    = settings.deep_weight
    cfg.ssim_weight    = settings.ssim_weight
    cfg.orb_weight     = settings.orb_weight
    cfg.contour_weight = settings.contour_weight
    cfg.updated_at     = datetime.utcnow()
    db.commit()
    db.refresh(cfg)
    return cfg


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_or_create_default(db: Session) -> ValidationConfig:
    cfg = db.query(ValidationConfig).filter(ValidationConfig.name == "default").first()
    if cfg is None:
        cfg = ValidationConfig(
            name="default",
            similarity_threshold=settings.similarity_threshold,
            siamese_weight=settings.siamese_weight,
            deep_weight=settings.deep_weight,
            ssim_weight=settings.ssim_weight,
            orb_weight=settings.orb_weight,
            contour_weight=settings.contour_weight,
        )
        db.add(cfg)
        db.commit()
        db.refresh(cfg)
    return cfg
