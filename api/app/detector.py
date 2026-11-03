"""Wrapper around the YOLOv8 model (ultralytics).

Supports three detection modes (set via DETECTION_MODE env var):
  ppe    – only PPE-trained model
  office – only office-objects model (or COCO fallback which knows laptops, etc.)
  both   – run one model that covers both class sets (default)
"""
import base64
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .config import settings
from .rules import Detection

log = logging.getLogger("siteguard.detector")


class Detector:
    """Loads the right model(s) depending on DETECTION_MODE."""

    def __init__(
        self,
        model_path:        str | None = None,
        office_model_path: str | None = None,
    ):
        from ultralytics import YOLO  # imported lazily: heavy import

        mode = settings.detection_mode.lower()

        # ── primary / PPE model ───────────────────────────────────────────────
        path = model_path or settings.model_path
        if Path(path).exists():
            self.model_name   = Path(path).name
            self._model       = YOLO(path)
            self.is_ppe_model = True
        else:
            log.warning(
                "Model %s not found, using fallback %s (people only, no PPE classes)",
                path, settings.fallback_model,
            )
            self.model_name   = settings.fallback_model
            self._model       = YOLO(settings.fallback_model)
            self.is_ppe_model = False

        # ── office model (only loaded when mode != ppe) ───────────────────────
        self._office_model      = None
        self._office_model_name = None

        if mode in ("office", "both"):
            opath = office_model_path or settings.office_model_path
            if Path(opath).exists():
                self._office_model      = YOLO(opath)
                self._office_model_name = Path(opath).name
                log.info("Loaded office model: %s", self._office_model_name)
            else:
                # COCO knows: laptop, keyboard, mouse, cell phone, cup, tv
                log.info(
                    "Office model %s not found – using COCO fallback (%s) for office objects",
                    opath, settings.fallback_model,
                )
                self._office_model      = YOLO(settings.fallback_model)
                self._office_model_name = f"coco-{settings.fallback_model}"

        self.mode = mode

        # Merge class names from all loaded models
        names: list[str] = list(self._model.names.values())
        if self._office_model is not None:
            for n in self._office_model.names.values():
                if n not in names:
                    names.append(n)
        self.class_names = names

    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[list[Detection], float, np.ndarray]:
        """Run the model(s). Returns (detections, latency_ms, annotated_image)."""
        t0          = time.perf_counter()
        detections  = []
        annotated   = image_bgr.copy()

        # -- PPE model (mode ppe or both) --
        if self.mode in ("ppe", "both"):
            result = self._model.predict(
                image_bgr,
                conf=settings.conf_threshold,
                imgsz=settings.img_size,
                verbose=False,
            )[0]
            annotated = result.plot()
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(
                    class_name=self._model.names[int(box.cls[0])],
                    confidence=round(float(box.conf[0]), 4),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        # -- Office model (mode office or both) --
        if self.mode in ("office", "both") and self._office_model is not None:
            # Filter predictions to only the classes we care about
            office_set = {c.lower() for c in settings.office_classes}

            o_result = self._office_model.predict(
                image_bgr,
                conf=settings.conf_threshold,
                imgsz=settings.img_size,
                verbose=False,
            )[0]

            for box in o_result.boxes:
                name = self._office_model.names[int(box.cls[0])]
                if name.lower() not in office_set:
                    continue   # skip non-office classes from COCO fallback
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(Detection(
                    class_name=name,
                    confidence=round(float(box.conf[0]), 4),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    detection_type="office",
                ))
            # Merge annotations when both models are used
            if self.mode == "both":
                annotated = o_result.plot(img=annotated)

        latency_ms = (time.perf_counter() - t0) * 1000
        return detections, latency_ms, annotated


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


def encode_jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""
