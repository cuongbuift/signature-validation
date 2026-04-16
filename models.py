from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    employee_code: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    signatures: Mapped[list["ReferenceSignature"]] = relationship(
        "ReferenceSignature", back_populates="employee", cascade="all, delete-orphan"
    )
    validations: Mapped[list["ValidationRecord"]] = relationship(
        "ValidationRecord", back_populates="employee"
    )


class ReferenceSignature(Base):
    __tablename__ = "reference_signatures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    employee_id: Mapped[int] = mapped_column(Integer, ForeignKey("employees.id"), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    contract_ref: Mapped[str] = mapped_column(String(200), nullable=True)  # Mã hợp đồng
    order: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 hoặc 2
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    employee: Mapped["Employee"] = relationship("Employee", back_populates="signatures")


class ValidationConfig(Base):
    __tablename__ = "validation_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.75)
    siamese_weight: Mapped[float] = mapped_column(Float, default=0.35)
    deep_weight: Mapped[float] = mapped_column(Float, default=0.30)
    ssim_weight: Mapped[float] = mapped_column(Float, default=0.20)
    orb_weight: Mapped[float] = mapped_column(Float, default=0.10)
    contour_weight: Mapped[float] = mapped_column(Float, default=0.05)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ValidationRecord(Base):
    __tablename__ = "validation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    employee_id: Mapped[int] = mapped_column(Integer, ForeignKey("employees.id"), nullable=False)
    delivery_ref: Mapped[str] = mapped_column(String(200), nullable=True)  # Mã phiếu giao hàng
    input_file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    siamese_score: Mapped[float] = mapped_column(Float, nullable=True)
    deep_score: Mapped[float] = mapped_column(Float, nullable=True)
    ssim_score: Mapped[float] = mapped_column(Float, nullable=True)
    orb_score: Mapped[float] = mapped_column(Float, nullable=True)
    contour_score: Mapped[float] = mapped_column(Float, nullable=True)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    validated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    employee: Mapped["Employee"] = relationship("Employee", back_populates="validations")
