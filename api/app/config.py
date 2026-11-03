"""Application settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field


def _list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# Detection mode controls which model is used and how results are interpreted.
#   ppe    – original behaviour: PPE compliance only
#   office – detect office objects (inventory / asset tracking)
#   both   – run a single model that covers both class sets
DETECTION_MODES = {"ppe", "office", "both"}


@dataclass(frozen=True)
class Settings:
    # Database: PostgreSQL in Docker, SQLite file when running without Docker.
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./siteguard.db")

    # Detection mode (ppe | office | both)
    detection_mode: str = os.getenv("DETECTION_MODE", "both")

    # Path to trained PPE weights. If the file does not exist, the API falls back
    # to the generic COCO model (it finds people, but not hard hats or vests).
    model_path: str = os.getenv("MODEL_PATH", "models/best.pt")

    # Separate model for office objects.  Falls back to same model_path / COCO.
    office_model_path: str = os.getenv("OFFICE_MODEL_PATH", "models/office_best.pt")

    fallback_model: str = os.getenv("FALLBACK_MODEL", "yolov8n.pt")

    conf_threshold: float = float(os.getenv("CONF_THRESHOLD", "0.35"))
    img_size: int = int(os.getenv("IMG_SIZE", "640"))

    # ── PPE classes ──────────────────────────────────────────────────────────
    # Class names that count as a safety violation (names from the dataset).
    violation_classes: list[str] = field(
        default_factory=lambda: _list(
            os.getenv("VIOLATION_CLASSES", "NO-Hardhat,NO-Safety Vest")
        )
    )
    person_classes: list[str] = field(
        default_factory=lambda: _list(os.getenv("PERSON_CLASSES", "Person,person"))
    )

    # ── Office / asset classes ────────────────────────────────────────────────
    # These are inventory items — their presence is logged but not flagged as a
    # "violation".  COCO already knows most of them (laptop, keyboard, mouse,
    # cell phone, cup, tv/monitor).
    office_classes: list[str] = field(
        default_factory=lambda: _list(
            os.getenv(
                "OFFICE_CLASSES",
                "laptop,keyboard,mouse,cell phone,cup,monitor,tv",
            )
        )
    )

    # Optional: flag when a valuable item is left unattended (no person nearby).
    flag_unattended: bool = os.getenv("FLAG_UNATTENDED", "false").lower() == "true"


settings = Settings()
