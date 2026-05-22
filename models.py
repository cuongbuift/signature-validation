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


class CustomerRecord(Base):
    __tablename__ = "customer_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Source PDF
    pdf_path: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(300), nullable=False)

    # Extracted customer info (editable)
    ten_dang_ky_kinh_doanh: Mapped[str] = mapped_column(String(300), nullable=True)
    ten_cua_hang: Mapped[str] = mapped_column(String(300), nullable=True)
    giay_phep_so: Mapped[str] = mapped_column(String(100), nullable=True)
    giay_phep_ngay_cap: Mapped[str] = mapped_column(String(50), nullable=True)
    giay_phep_noi_cap: Mapped[str] = mapped_column(String(300), nullable=True)
    dia_chi_kinh_doanh: Mapped[str] = mapped_column(String(500), nullable=True)

    # Signature image file paths (null = not present)
    sig_ct_lan1: Mapped[str] = mapped_column(String(500), nullable=True)  # chịu trách nhiệm lần 1
    sig_ct_lan2: Mapped[str] = mapped_column(String(500), nullable=True)
    sig_uq1_lan1: Mapped[str] = mapped_column(String(500), nullable=True)  # ủy quyền 1 lần 1
    sig_uq1_lan2: Mapped[str] = mapped_column(String(500), nullable=True)
    sig_uq2_lan1: Mapped[str] = mapped_column(String(500), nullable=True)
    sig_uq2_lan2: Mapped[str] = mapped_column(String(500), nullable=True)
    sig_uq3_lan1: Mapped[str] = mapped_column(String(500), nullable=True)
    sig_uq3_lan2: Mapped[str] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
