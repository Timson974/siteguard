# SiteGuard — PPE & Office Object Detection

SiteGuard is a personal end-to-end computer vision project: images go in, safety violations and
asset detections come out, everything is stored in PostgreSQL and surfaced on an analytics dashboard.

> **Status:** in progress. Model metrics will be filled in after training.

---

## What it does

| Mode | Use case | What gets flagged |
|---|---|---|
| `ppe` | Construction sites | Workers without hard hats or safety vests |
| `office` | Office environments | Laptop, keyboard, mouse, phone, cup, monitor (inventory) |
| `both` *(default)* | Mixed / general | PPE violations **and** office assets in one pass |

Switch modes with the `DETECTION_MODE` environment variable — no code changes needed.

---

## Architecture

```
image / video frame
      │  POST /detect  (or /detect/office)
      ▼
┌─────────────┐    YOLOv8 (fine-tuned PPE)     ┌───────────────────┐
│  FastAPI    │  ──────────────────────────▶   │  rules.py         │
│  (api/)     │                                │  PPE → violation  │
│             │  ──────────────────────────▶   │  office → asset   │
└──────┬──────┘    YOLOv8 / COCO (office)      └───────────────────┘
       │  save inference + detections
       ▼
┌─────────────┐   SQL aggregates   ┌──────────────────────────────┐
│ PostgreSQL  │  ───────────────▶  │  Streamlit dashboard         │
└─────────────┘   /stats/*         │  violations + inventory view │
                                   └──────────────────────────────┘
```

| Layer | Technology | Location |
|---|---|---|
| Object detection | YOLOv8 (ultralytics) | `api/app/detector.py` |
| Business rules | Python dataclasses | `api/app/rules.py` |
| REST API | FastAPI + OpenCV | `api/` |
| Storage | PostgreSQL + SQLAlchemy | `api/app/db.py` |
| Analytics SQL | Plain SQL (Postgres + SQLite) | `api/app/stats.py`, `sql/` |
| Dashboard | Streamlit + Plotly | `dashboard/` |
| Training notebook | YOLOv8, Google Colab GPU | `notebooks/` |
| Packaging | Docker Compose | `docker-compose.yml` |
| CI | GitHub Actions: lint + tests + Docker build | `.github/workflows/ci.yml` |

---

## Quick start

**Requirements:** Docker Desktop only.

```bash
git clone https://github.com/Timson974/siteguard.git
cd siteguard
cp .env.example .env        # edit if needed — defaults work out of the box
docker compose up --build
```

- **API + interactive docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

Without trained weights the API falls back to the generic `yolov8n.pt` (detects people;
office-class COCO objects work immediately — laptop, keyboard, mouse, phone, cup, etc.).

### Add trained PPE weights

Train with `notebooks/train_yolov8_colab.ipynb` → copy `best.pt` to `models/` → restart:

```bash
docker compose restart api
```

### Simulate camera traffic

```bash
pip install requests
python scripts/simulate_cameras.py --images samples --n 200
```

### Try the dashboard without training

```bash
# Insert labelled synthetic rows (never used in real metrics)
curl -X POST "http://localhost:8000/demo/synthetic?n=500&days=14"
# Remove them again
curl -X DELETE "http://localhost:8000/demo/synthetic"
```

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model` | Loaded model, detection mode, and class list |
| `POST` | `/detect` | Image → PPE violations + office assets (saved to DB) |
| `POST` | `/detect/office` | Image → office objects only (inventory / asset tracking) |
| `GET` | `/stats/summary` | Totals: frames, persons, violations, latency |
| `GET` | `/stats/by-camera` | Violations per camera, normalised per frame |
| `GET` | `/stats/by-day` | Violations per day and type |
| `GET` | `/stats/by-hour` | Violations by hour of day |
| `GET` | `/stats/by-object` | Office asset counts (inventory view) |
| `GET` | `/violations/recent` | Latest violations (default: last 50) |

### `/detect` response (mode = both)

```json
{
  "inference_id": 42,
  "camera_id": "cam_01",
  "latency_ms": 87.3,
  "n_persons": 3,
  "n_violations": 1,
  "violations_by_type": { "NO-Hardhat": 1 },
  "n_office_objects": 2,
  "office_by_type": { "laptop": 1, "cup": 1 },
  "detections": [ ... ]
}
```

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env` and edit:

| Variable | Default | Description |
|---|---|---|
| `DETECTION_MODE` | `both` | `ppe` / `office` / `both` |
| `MODEL_PATH` | `models/best.pt` | PPE model weights |
| `OFFICE_MODEL_PATH` | `models/office_best.pt` | Office model weights (COCO fallback if missing) |
| `VIOLATION_CLASSES` | `NO-Hardhat,NO-Safety Vest` | PPE classes that count as violations |
| `OFFICE_CLASSES` | `laptop,keyboard,mouse,cell phone,cup,monitor,tv` | Asset classes to track |
| `CONF_THRESHOLD` | `0.35` | Minimum detection confidence |
| `FLAG_UNATTENDED` | `false` | Soft-alert when an asset has no person nearby |

---

## Dataset

*Construction Site Safety Image Dataset* — Roboflow Universe
(`snehilsanyal/construction-site-safety-image-dataset-roboflow`, Kaggle mirror):
~2,800 images, 10 classes including `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`.
See the dataset page for its licence.

Office objects use COCO pre-trained weights (no extra dataset needed).

---

## Model results

| Metric (PPE, test split) | Value |
|---|---|
| mAP@0.5 | *tbd — after training* |
| Precision | *tbd* |
| Recall | *tbd* |
| Recall — NO-Hardhat | *tbd* |
| CPU latency per image | *tbd* |

<!-- After training: add docs/results.png, docs/confusion_matrix.png -->

---

## Run tests locally

```bash
pip install -r api/requirements.txt pytest httpx
pytest -q
```

---

## Limitations & next steps

- Public dataset images differ from real site cameras (angle, lighting, distance); production use would need domain-specific fine-tuning.
- Small or partially occluded people are harder to detect.
- Planned: video stream support with cross-frame person tracking; alert webhook; optional AWS deployment (see `docs/aws_deploy.md`).

---

## Licence

Code: MIT. Dataset: see its own licence.
