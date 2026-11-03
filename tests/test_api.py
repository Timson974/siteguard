"""API tests with a fake detector, so they run in CI without downloading YOLO."""
import io
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_siteguard.db"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rules import Detection  # noqa: E402


class FakeDetector:
    model_name = "fake.pt"
    is_ppe_model = True
    class_names = ["Person", "Hardhat", "NO-Hardhat"]

    def predict(self, image):
        dets = [Detection("Person", 0.9, 0, 0, 10, 10),
                Detection("NO-Hardhat", 0.8, 0, 0, 5, 5)]
        return dets, 12.3, image


@pytest.fixture()
def client():
    Base.metadata.drop_all(engine)
    with TestClient(create_app(detector=FakeDetector())) as c:
        yield c
    Base.metadata.drop_all(engine)


def jpeg_bytes():
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    return io.BytesIO(buf.tobytes())


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_detect_saves_and_stats(client):
    r = client.post("/detect", files={"file": ("x.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"camera_id": "cam_02"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_violations"] == 1 and body["n_persons"] == 1
    assert body["inference_id"] is not None

    s = client.get("/stats/summary").json()
    assert s["frames"] == 1 and s["violations"] == 1

    cams = client.get("/stats/by-camera").json()
    assert cams[0]["camera_id"] == "cam_02"

    days = client.get("/stats/by-day").json()
    assert days[0]["violation_type"] == "NO-Hardhat"

    assert len(client.get("/violations/recent").json()) == 1


def test_bad_image_rejected(client):
    r = client.post("/detect", files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")})
    assert r.status_code == 400
