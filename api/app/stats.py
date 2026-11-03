"""Analytics queries in plain SQL (work on both PostgreSQL and SQLite).

The same queries, with comments, are in sql/analytics.sql.
"""
from sqlalchemy import text
from sqlalchemy.orm import Session


def _source_filter(include_synthetic: bool) -> str:
    return "" if include_synthetic else "AND i.source = 'api'"


def summary(db: Session, include_synthetic: bool = False) -> dict:
    row = db.execute(text(f"""
        SELECT COUNT(*)                          AS frames,
               COALESCE(SUM(i.n_persons), 0)     AS persons,
               COALESCE(SUM(i.n_violations), 0)  AS violations,
               COALESCE(AVG(i.latency_ms), 0)    AS avg_latency_ms,
               COUNT(DISTINCT i.camera_id)       AS cameras
        FROM inferences i
        WHERE 1 = 1 {_source_filter(include_synthetic)}
    """)).mappings().one()
    return dict(row)


def by_camera(db: Session, include_synthetic: bool = False) -> list[dict]:
    rows = db.execute(text(f"""
        SELECT i.camera_id,
               COUNT(*)                 AS frames,
               SUM(i.n_violations)      AS violations,
               ROUND(SUM(i.n_violations) * 1.0 / COUNT(*), 2) AS violations_per_frame
        FROM inferences i
        WHERE 1 = 1 {_source_filter(include_synthetic)}
        GROUP BY i.camera_id
        ORDER BY violations DESC
    """)).mappings().all()
    return [dict(r) for r in rows]


def by_day(db: Session, include_synthetic: bool = False) -> list[dict]:
    rows = db.execute(text(f"""
        SELECT DATE(i.captured_at)  AS day,
               d.class_name         AS violation_type,
               COUNT(*)             AS violations
        FROM detections d
        JOIN inferences i ON i.id = d.inference_id
        WHERE d.is_violation {_source_filter(include_synthetic)}
        GROUP BY DATE(i.captured_at), d.class_name
        ORDER BY day
    """)).mappings().all()
    return [{**dict(r), "day": str(r["day"])} for r in rows]


def by_hour(db: Session, include_synthetic: bool = False) -> list[dict]:
    """Violations by hour of day: when on the shift do problems happen?"""
    dialect = db.get_bind().dialect.name
    hour = "EXTRACT(HOUR FROM i.captured_at)" if dialect == "postgresql" \
        else "CAST(strftime('%H', i.captured_at) AS INTEGER)"
    rows = db.execute(text(f"""
        SELECT {hour} AS hour, SUM(i.n_violations) AS violations
        FROM inferences i
        WHERE 1 = 1 {_source_filter(include_synthetic)}
        GROUP BY {hour}
        ORDER BY hour
    """)).mappings().all()
    return [{"hour": int(r["hour"]), "violations": int(r["violations"] or 0)} for r in rows]


def by_object(db: Session) -> list[dict]:
    """Office asset inventory: total detections per object class (source='office')."""
    rows = db.execute(text("""
        SELECT d.class_name        AS object_type,
               COUNT(*)            AS count,
               ROUND(AVG(d.confidence), 3) AS avg_confidence
        FROM detections d
        JOIN inferences i ON i.id = d.inference_id
        WHERE i.source = 'office'
        GROUP BY d.class_name
        ORDER BY count DESC
    """)).mappings().all()
    return [dict(r) for r in rows]


def recent(db: Session, limit: int = 50, include_synthetic: bool = False) -> list[dict]:
    rows = db.execute(text(f"""
        SELECT i.id, i.camera_id, i.captured_at, d.class_name, d.confidence
        FROM detections d
        JOIN inferences i ON i.id = d.inference_id
        WHERE d.is_violation {_source_filter(include_synthetic)}
        ORDER BY i.captured_at DESC
        LIMIT :limit
    """), {"limit": limit}).mappings().all()
    return [{**dict(r), "captured_at": str(r["captured_at"])} for r in rows]
