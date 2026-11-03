"""Synthetic demo rows (source='synthetic') so the dashboard can be tried before
the model is trained. Hidden by default in the dashboard and excluded from stats
unless include_synthetic=true. Never report these numbers as results."""
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete
from sqlalchemy.orm import Session

from .db import DetectionRow, Inference

CAMERAS = ["cam_01", "cam_02", "cam_03", "cam_04"]
# cam_03 is made "worse" on purpose so the charts show a visible pattern
RISK = {"cam_01": 0.3, "cam_02": 0.4, "cam_03": 0.9, "cam_04": 0.2}


def random_time(days: int) -> datetime:
    now = datetime.now(timezone.utc)
    day = now - timedelta(days=random.randint(0, days - 1))
    return day.replace(hour=random.randint(7, 18), minute=random.randint(0, 59), second=0, microsecond=0)


def insert_synthetic(db: Session, n: int, days: int = 14) -> int:
    for _ in range(n):
        cam = random.choice(CAMERAS)
        n_people = random.randint(1, 6)
        dets = [DetectionRow(class_name="Person", confidence=0.9, x1=0, y1=0, x2=1, y2=1, is_violation=False)
                for _ in range(n_people)]
        for _ in range(n_people):
            if random.random() < RISK[cam] * 0.3:
                dets.append(DetectionRow(class_name=random.choice(["NO-Hardhat", "NO-Safety Vest"]),
                                         confidence=round(random.uniform(0.4, 0.95), 2),
                                         x1=0, y1=0, x2=1, y2=1, is_violation=True))
        inf = Inference(camera_id=cam, captured_at=random_time(days), model_name="synthetic", latency_ms=0,
                        n_persons=n_people, n_violations=sum(d.is_violation for d in dets), source="synthetic")
        inf.detections = dets
        db.add(inf)
    db.commit()
    return n


def delete_synthetic(db: Session) -> int:
    ids = [i for (i,) in db.query(Inference.id).filter(Inference.source == "synthetic").all()]
    if ids:
        db.execute(delete(DetectionRow).where(DetectionRow.inference_id.in_(ids)))
        db.execute(delete(Inference).where(Inference.id.in_(ids)))
        db.commit()
    return len(ids)
