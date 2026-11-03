"""Business rules: turn raw detections into safety violations or asset events.

Kept separate from the model code so it can be unit-tested without loading YOLO.
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    is_violation: bool = False
    # "ppe" | "office" | "person" | "other"
    detection_type: str = "other"

    def to_dict(self) -> dict:
        return asdict(self)


# ── PPE rules ─────────────────────────────────────────────────────────────────

def apply_rules(
    detections: list[Detection],
    violation_classes: list[str],
    person_classes: Optional[list[str]] = None,
    office_classes: Optional[list[str]] = None,
) -> list[Detection]:
    """Mark each detection as a violation / type.

    The PPE dataset already has explicit "negative" classes (e.g. "NO-Hardhat"),
    so a violation is simply a confident detection of one of those classes.
    Matching is case-insensitive.
    """
    wanted_violations = {c.lower() for c in violation_classes}
    wanted_persons    = {c.lower() for c in (person_classes  or [])}
    wanted_office     = {c.lower() for c in (office_classes  or [])}

    for d in detections:
        name_lc = d.class_name.lower()
        if name_lc in wanted_violations:
            d.is_violation    = True
            d.detection_type  = "ppe"
        elif name_lc in wanted_persons:
            d.detection_type  = "person"
        elif name_lc in wanted_office:
            d.detection_type  = "office"
        else:
            d.detection_type  = "other"
    return detections


# ── Office / asset rules ──────────────────────────────────────────────────────

def flag_unattended_assets(
    detections: list[Detection],
    office_classes: list[str],
    person_classes: list[str],
    iou_threshold: float = 0.0,   # 0 = any spatial proximity in same quadrant
) -> list[Detection]:
    """Optionally mark office objects as 'unattended' when no person is near them.

    Uses a simple proximity heuristic: if the centre of the asset bbox falls
    within any person bbox (expanded by 50 %), the asset is considered attended.
    """
    wanted_office  = {c.lower() for c in office_classes}
    wanted_persons = {c.lower() for c in person_classes}

    person_boxes = [
        d for d in detections if d.class_name.lower() in wanted_persons
    ]

    for d in detections:
        if d.class_name.lower() not in wanted_office:
            continue
        cx = (d.x1 + d.x2) / 2
        cy = (d.y1 + d.y2) / 2
        attended = False
        for p in person_boxes:
            pw = p.x2 - p.x1
            ph = p.y2 - p.y1
            px1 = p.x1 - pw * 0.5
            py1 = p.y1 - ph * 0.5
            px2 = p.x2 + pw * 0.5
            py2 = p.y2 + ph * 0.5
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                attended = True
                break
        if not attended:
            d.is_violation   = True   # treated as a soft "alert"

    return detections


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise(
    detections: list[Detection],
    person_classes: list[str],
    office_classes: Optional[list[str]] = None,
) -> dict:
    persons   = {c.lower() for c in person_classes}
    office_set = {c.lower() for c in (office_classes or [])}

    ppe_violations: dict[str, int] = {}
    office_counts:  dict[str, int] = {}

    for d in detections:
        name_lc = d.class_name.lower()
        if d.is_violation and name_lc not in office_set:
            ppe_violations[d.class_name] = ppe_violations.get(d.class_name, 0) + 1
        if name_lc in office_set:
            office_counts[d.class_name] = office_counts.get(d.class_name, 0) + 1

    return {
        "n_detections":       len(detections),
        "n_persons":          sum(1 for d in detections if d.class_name.lower() in persons),
        "n_violations":       sum(ppe_violations.values()),
        "violations_by_type": ppe_violations,
        # office inventory
        "n_office_objects":   sum(office_counts.values()),
        "office_by_type":     office_counts,
    }
