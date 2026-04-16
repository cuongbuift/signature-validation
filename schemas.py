from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ── Employee ──────────────────────────────────────────────────────────────────

class EmployeeCreate(BaseModel):
    employee_code: str = Field(..., min_length=1, max_length=50, examples=["NV001"])
    full_name: str = Field(..., min_length=1, max_length=200, examples=["Nguyen Van A"])


class EmployeeUpdate(BaseModel):
    full_name: str | None = Field(None, min_length=1, max_length=200)


class EmployeeOut(BaseModel):
    id: int
    employee_code: str
    full_name: str
    created_at: datetime
    is_active: bool
    signature_count: int = 0

    model_config = {"from_attributes": True}


# ── Reference Signature ───────────────────────────────────────────────────────

class SignatureOut(BaseModel):
    id: int
    employee_id: int
    contract_ref: str | None
    order: int
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    is_valid: bool
    overall_score: float = Field(..., ge=0.0, le=1.0)
    siamese_score: float
    deep_score: float
    ssim_score: float
    orb_score: float
    contour_score: float
    threshold_used: float
    employee_code: str
    delivery_ref: str | None
    detail: dict


class ValidationRecordOut(BaseModel):
    id: int
    employee_id: int
    delivery_ref: str | None
    is_valid: bool
    overall_score: float
    siamese_score: float | None
    deep_score: float | None
    ssim_score: float | None
    orb_score: float | None
    contour_score: float | None
    threshold_used: float
    validated_at: datetime

    model_config = {"from_attributes": True}


# ── Config ────────────────────────────────────────────────────────────────────

class ConfigUpdate(BaseModel):
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    siamese_weight: float | None = Field(None, ge=0.0, le=1.0)
    deep_weight: float | None = Field(None, ge=0.0, le=1.0)
    ssim_weight: float | None = Field(None, ge=0.0, le=1.0)
    orb_weight: float | None = Field(None, ge=0.0, le=1.0)
    contour_weight: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("siamese_weight", "deep_weight", "ssim_weight", "orb_weight", "contour_weight", mode="before")
    @classmethod
    def check_weights(cls, v):
        return v  # Cross-field sum validation done in the router


class ConfigOut(BaseModel):
    name: str
    similarity_threshold: float
    siamese_weight: float
    deep_weight: float
    ssim_weight: float
    orb_weight: float
    contour_weight: float
    updated_at: datetime

    model_config = {"from_attributes": True}
