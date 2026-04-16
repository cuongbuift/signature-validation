"""
API endpoints for the Siamese signature verification model.

POST /siamese/train   — Train (or retrain) from existing reference signatures.
GET  /siamese/status  — Model availability + training metadata.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config import settings
from database import get_db
from models import Employee, ReferenceSignature

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/siamese", tags=["Siamese Model"])


# ── Response schemas ──────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = 60
    augment_factor: int = 15


class TrainResponse(BaseModel):
    status: str
    message: str
    epochs_run: int = 0
    best_val_loss: float = 0.0
    n_employees: int = 0
    n_pairs: int = 0
    duration_seconds: float = 0.0
    train_loss_history: list[float] = []
    val_loss_history: list[float]   = []


class StatusResponse(BaseModel):
    model_available: bool
    model_path: str
    last_trained: str | None
    n_employees_at_training: int | None
    n_pairs_at_training: int | None


# ── Meta file path ────────────────────────────────────────────────────────────

_META_PATH = settings.siamese_model_path.with_suffix(".meta.json")

def _read_meta() -> dict:
    if _META_PATH.exists():
        try:
            return json.loads(_META_PATH.read_text())
        except Exception:
            pass
    return {}

def _write_meta(data: dict) -> None:
    _META_PATH.parent.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(json.dumps(data, indent=2))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse, summary="Trạng thái Siamese model")
def get_status():
    meta = _read_meta()
    return StatusResponse(
        model_available=settings.siamese_model_path.exists(),
        model_path=str(settings.siamese_model_path),
        last_trained=meta.get("last_trained"),
        n_employees_at_training=meta.get("n_employees"),
        n_pairs_at_training=meta.get("n_pairs"),
    )


@router.post("/train", response_model=TrainResponse, summary="Huấn luyện Siamese model từ chữ ký mẫu")
def train_model(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Build training pairs from all reference signatures currently in the
    database, then train the Siamese network.

    Requires at least **2 employees** with reference signatures.

    Training runs synchronously (may take 30–120 s on CPU/MPS depending on
    the number of signatures and epochs).
    """
    # Collect reference data from DB
    employees = (
        db.query(Employee)
        .filter(Employee.is_active == True)
        .all()
    )

    reference_data: dict[int, list[str]] = {}
    for emp in employees:
        paths = [
            sig.file_path
            for sig in emp.signatures
            if Path(sig.file_path).exists()
        ]
        if paths:
            reference_data[emp.id] = paths

    if len(reference_data) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cần ít nhất 2 nhân viên có chữ ký mẫu hợp lệ trên disk để huấn luyện. "
                f"Hiện tại: {len(reference_data)} nhân viên."
            ),
        )

    # Train
    from validators.siamese_trainer import SiameseTrainer
    trainer = SiameseTrainer(
        reference_data=reference_data,
        model_path=settings.siamese_model_path,
    )
    report = trainer.train(
        epochs=payload.epochs,
        augment_factor=payload.augment_factor,
    )

    # Persist meta
    if report.status == "ok":
        _write_meta({
            "last_trained": datetime.utcnow().isoformat(),
            "n_employees": report.n_employees,
            "n_pairs": report.n_pairs,
            "epochs": report.epochs_run,
            "best_val_loss": report.best_val_loss,
        })

    return TrainResponse(
        status=report.status,
        message=report.message,
        epochs_run=report.epochs_run,
        best_val_loss=report.best_val_loss,
        n_employees=report.n_employees,
        n_pairs=report.n_pairs,
        duration_seconds=report.duration_seconds,
        train_loss_history=report.train_loss_history,
        val_loss_history=report.val_loss_history,
    )
