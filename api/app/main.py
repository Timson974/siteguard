"""SiteGuard REST API.

Endpoints
  GET  /health                 service is up
  GET  /model                  which model(s) are loaded and their classes
  POST /detect                 send an image, get PPE + office detections (and save them)
  POST /detect/office          office-objects only (inventory / asset tracking)
  GET  /stats/summary          totals
  GET  /stats/by-camera        violations per camera
  GET  /stats/by-day           violations per day and type
  GET  /stats/by-hour          violations per hour of day
  GET  /stats/by-object        office object counts (inventory view)
  GET  /violations/recent      latest violations
  POST /demo/synthetic         add labelled fake rows to try the dashboard
  DELETE /demo/synthetic       remove them
Interactive docs: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from . import demo, stats
from .config import settings
from .db import DetectionRow, Inference, SessionLocal, init_db
from .detector import decode_image, encode_jpeg_b64
from .rules import apply_rules, flag_unattended_assets, summarise

logging.basicConfig(level=logging.INFO)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_app(detector=None) -> FastAPI:
    """`detector` can be injected (tests use a fake one so YOLO is not needed)."""
    state = {"detector": detector}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db()
        if state["detector"] is None:
            from .detector import Detector
            state["detector"] = Detector()
        yield

    app = FastAPI(
        title="SiteGuard API",
        version="0.2.0",
        lifespan=lifespan,
        description=(
            "PPE violation detection + office asset tracking for construction "
            "sites and office environments."
        ),
    )

    def det():
        return state["detector"]

    # ── Health / info ─────────────────────────────────────────────────────────

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model")
    def model_info():
        d = det()
        return {
            "model":             d.model_name,
            "ppe_model":         d.is_ppe_model,
            "detection_mode":    d.mode,
            "classes":           d.class_names,
            "violation_classes": settings.violation_classes,
            "office_classes":    settings.office_classes,
            "conf_threshold":    settings.conf_threshold,
        }

    # ── Detection endpoints ───────────────────────────────────────────────────

    @app.post("/detect")
    async def detect(
        file:        UploadFile          = File(...),
        camera_id:   str                 = Form("cam_01"),
        captured_at: Optional[datetime]  = Form(None),
        save:        bool                = Form(True),
        annotate:    bool                = Form(False),
        db:          Session             = Depends(get_db),
    ):
        """Full detection: PPE violations + office object inventory."""
        try:
            image = decode_image(await file.read())
        except ValueError:
            raise HTTPException(status_code=400, detail="File is not a readable image")

        detections, latency_ms, annotated_img = det().predict(image)
        apply_rules(
            detections,
            settings.violation_classes,
            person_classes=settings.person_classes,
            office_classes=settings.office_classes,
        )
        if settings.flag_unattended:
            flag_unattended_assets(
                detections,
                office_classes=settings.office_classes,
                person_classes=settings.person_classes,
            )
        summary = summarise(detections, settings.person_classes, settings.office_classes)
        captured_at = captured_at or datetime.now(timezone.utc)

        inference_id = None
        if save:
            inf = Inference(
                camera_id=camera_id,
                captured_at=captured_at,
                model_name=det().model_name,
                latency_ms=latency_ms,
                n_persons=summary["n_persons"],
                n_violations=summary["n_violations"],
                source="api",
            )
            inf.detections = [DetectionRow(**d.to_dict()) for d in detections]
            db.add(inf)
            db.commit()
            inference_id = inf.id

        response = {
            "inference_id": inference_id,
            "camera_id":    camera_id,
            "captured_at":  captured_at.isoformat(),
            "model":        det().model_name,
            "mode":         det().mode,
            "latency_ms":   round(latency_ms, 1),
            **summary,
            "detections":   [d.to_dict() for d in detections],
        }
        if annotate:
            response["annotated_image_b64"] = encode_jpeg_b64(annotated_img)
        return response

    @app.post("/detect/office")
    async def detect_office(
        file:        UploadFile          = File(...),
        camera_id:   str                 = Form("cam_01"),
        captured_at: Optional[datetime]  = Form(None),
        save:        bool                = Form(True),
        annotate:    bool                = Form(False),
        db:          Session             = Depends(get_db),
    ):
        """Office / asset mode: returns only office-class detections.

        Useful for inventory scans — no PPE logic is applied.
        """
        try:
            image = decode_image(await file.read())
        except ValueError:
            raise HTTPException(status_code=400, detail="File is not a readable image")

        detections, latency_ms, annotated_img = det().predict(image)

        # Keep only office objects
        office_set = {c.lower() for c in settings.office_classes}
        detections = [d for d in detections if d.class_name.lower() in office_set]
        for d in detections:
            d.detection_type = "office"

        captured_at = captured_at or datetime.now(timezone.utc)
        office_counts: dict[str, int] = {}
        for d in detections:
            office_counts[d.class_name] = office_counts.get(d.class_name, 0) + 1

        inference_id = None
        if save:
            inf = Inference(
                camera_id=camera_id,
                captured_at=captured_at,
                model_name=det().model_name,
                latency_ms=latency_ms,
                n_persons=0,
                n_violations=0,
                source="office",
            )
            inf.detections = [DetectionRow(**d.to_dict()) for d in detections]
            db.add(inf)
            db.commit()
            inference_id = inf.id

        response = {
            "inference_id":   inference_id,
            "camera_id":      camera_id,
            "captured_at":    captured_at.isoformat(),
            "model":          det().model_name,
            "mode":           "office",
            "latency_ms":     round(latency_ms, 1),
            "n_office_objects": len(detections),
            "office_by_type": office_counts,
            "detections":     [d.to_dict() for d in detections],
        }
        if annotate:
            response["annotated_image_b64"] = encode_jpeg_b64(annotated_img)
        return response

    # ── Stats endpoints ───────────────────────────────────────────────────────

    @app.get("/stats/summary")
    def stats_summary(include_synthetic: bool = False, db: Session = Depends(get_db)):
        return stats.summary(db, include_synthetic)

    @app.get("/stats/by-camera")
    def stats_by_camera(include_synthetic: bool = False, db: Session = Depends(get_db)):
        return stats.by_camera(db, include_synthetic)

    @app.get("/stats/by-day")
    def stats_by_day(include_synthetic: bool = False, db: Session = Depends(get_db)):
        return stats.by_day(db, include_synthetic)

    @app.get("/stats/by-hour")
    def stats_by_hour(include_synthetic: bool = False, db: Session = Depends(get_db)):
        return stats.by_hour(db, include_synthetic)

    @app.get("/stats/by-object")
    def stats_by_object(db: Session = Depends(get_db)):
        """Office asset inventory — counts per detected class from office-mode inferences."""
        return stats.by_object(db)

    @app.get("/violations/recent")
    def violations_recent(
        limit:              int  = Query(50, le=500),
        include_synthetic:  bool = False,
        db:                 Session = Depends(get_db),
    ):
        return stats.recent(db, limit, include_synthetic)

    # ── Demo endpoints ────────────────────────────────────────────────────────

    @app.post("/demo/synthetic")
    def demo_insert(
        n:    int = Query(300, le=5000),
        days: int = Query(14, le=90),
        db:   Session = Depends(get_db),
    ):
        return {"inserted": demo.insert_synthetic(db, n, days), "source": "synthetic"}

    @app.delete("/demo/synthetic")
    def demo_delete(db: Session = Depends(get_db)):
        return {"deleted": demo.delete_synthetic(db)}

    return app


app = create_app()
