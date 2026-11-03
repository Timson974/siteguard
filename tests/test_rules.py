from app.rules import Detection, apply_rules, flag_unattended_assets, summarise


def d(name, conf=0.9, x1=0, y1=0, x2=10, y2=10):
    return Detection(class_name=name, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


OFFICE = ["laptop", "keyboard", "mouse", "cup"]
PERSONS = ["Person", "person"]


# ── PPE rules ─────────────────────────────────────────────────────────────────

def test_violation_classes_are_marked():
    dets = apply_rules(
        [d("Person"), d("NO-Hardhat"), d("Hardhat"), d("NO-Safety Vest")],
        ["NO-Hardhat", "NO-Safety Vest"],
    )
    assert [x.is_violation for x in dets] == [False, True, False, True]


def test_matching_is_case_insensitive():
    dets = apply_rules([d("no-hardhat")], ["NO-Hardhat"])
    assert dets[0].is_violation


def test_summary_counts():
    dets = apply_rules(
        [d("Person"), d("Person"), d("NO-Hardhat"), d("NO-Hardhat"), d("NO-Safety Vest")],
        ["NO-Hardhat", "NO-Safety Vest"],
    )
    s = summarise(dets, ["Person"])
    assert s["n_persons"] == 2
    assert s["n_violations"] == 3
    assert s["violations_by_type"] == {"NO-Hardhat": 2, "NO-Safety Vest": 1}


def test_no_detections():
    assert summarise([], ["Person"])["n_violations"] == 0


# ── Office / asset rules ──────────────────────────────────────────────────────

def test_office_detection_type():
    dets = apply_rules(
        [d("Person"), d("laptop"), d("cup")],
        violation_classes=["NO-Hardhat"],
        person_classes=PERSONS,
        office_classes=OFFICE,
    )
    types = {x.class_name: x.detection_type for x in dets}
    assert types["Person"] == "person"
    assert types["laptop"] == "office"
    assert types["cup"] == "office"


def test_office_summary():
    dets = apply_rules(
        [d("Person"), d("laptop"), d("laptop"), d("cup"), d("NO-Hardhat")],
        violation_classes=["NO-Hardhat"],
        person_classes=PERSONS,
        office_classes=OFFICE,
    )
    s = summarise(dets, PERSONS, office_classes=OFFICE)
    assert s["n_office_objects"] == 3
    assert s["office_by_type"] == {"laptop": 2, "cup": 1}
    assert s["n_violations"] == 1


def test_unattended_flag():
    # Laptop far from person → should be flagged
    laptop = d("laptop", x1=200, y1=200, x2=250, y2=240)
    person = d("Person", x1=0,   y1=0,   x2=50,  y2=100)

    dets = [laptop, person]
    apply_rules(dets, [], person_classes=PERSONS, office_classes=OFFICE)
    flag_unattended_assets(dets, office_classes=OFFICE, person_classes=PERSONS)
    assert laptop.is_violation is True


def test_attended_not_flagged():
    # Laptop inside person bbox → should NOT be flagged
    laptop = d("laptop", x1=10, y1=10, x2=30, y2=30)
    person = d("Person", x1=0,  y1=0,  x2=50, y2=100)

    dets = [laptop, person]
    apply_rules(dets, [], person_classes=PERSONS, office_classes=OFFICE)
    flag_unattended_assets(dets, office_classes=OFFICE, person_classes=PERSONS)
    assert laptop.is_violation is False
