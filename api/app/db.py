"""Database tables (SQLAlchemy). Works with PostgreSQL and SQLite."""
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String,
                        create_engine)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Inference(Base):
    """One image/frame processed by the model."""
    __tablename__ = "inferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(50), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_name: Mapped[str] = mapped_column(String(100))
    latency_ms: Mapped[float] = mapped_column(Float)
    n_persons: Mapped[int] = mapped_column(Integer, default=0)
    n_violations: Mapped[int] = mapped_column(Integer, default=0)
    # "api" = real model output; "synthetic" = demo rows from scripts/simulate_cameras.py --synthetic
    source: Mapped[str] = mapped_column(String(20), default="api", index=True)

    detections: Mapped[list["DetectionRow"]] = relationship(
        back_populates="inference", cascade="all, delete-orphan")


class DetectionRow(Base):
    """One bounding box found in an image."""
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inference_id: Mapped[int] = mapped_column(ForeignKey("inferences.id", ondelete="CASCADE"), index=True)
    class_name: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    is_violation: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    inference: Mapped[Inference] = relationship(back_populates="detections")


def make_engine(url: str | None = None):
    url = url or settings.database_url
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kwargs)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db(eng=None) -> None:
    Base.metadata.create_all(eng or engine)
