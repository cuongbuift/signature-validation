from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./signature_validation.db"

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_dir: Path = Path("storage/signatures")

    # ── Validation ────────────────────────────────────────────────────────────
    similarity_threshold: float = 0.75
    siamese_weight: float = 0.35
    deep_weight: float = 0.30
    ssim_weight: float = 0.20
    orb_weight: float = 0.10
    contour_weight: float = 0.05

    # ── Siamese model ─────────────────────────────────────────────────────────
    siamese_model_path: Path = Path("models/siamese.pt")
    max_reference_signatures: int = 2

    # ── Image processing ──────────────────────────────────────────────────────
    # Env format: SIGNATURE_TARGET_SIZE=300,150
    signature_target_size: str = "300,150"

    # ── YOLO signature detector ───────────────────────────────────────────────
    yolo_model_path: Path = Path("models/signature_yolo.pt")
    yolo_conf: float = 0.25          # minimum detection confidence
    yolo_padding: float = 0.08       # fractional padding around detected box

    @property
    def target_size(self) -> tuple[int, int]:
        parts = [p.strip() for p in self.signature_target_size.split(",")]
        if len(parts) != 2:
            raise ValueError("SIGNATURE_TARGET_SIZE must be 'width,height' (e.g. 300,150)")
        return (int(parts[0]), int(parts[1]))


settings = Settings()
