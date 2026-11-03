"""Send images to the API as if they came from several site cameras.

Real mode (default): every image goes through the model via POST /detect.
    python scripts/simulate_cameras.py --images samples --n 200

Synthetic mode: asks the API to add clearly-labelled fake rows (source='synthetic')
so you can see the dashboard before the model is trained. They are hidden in the
dashboard unless you tick "Include synthetic demo data". Never report them as results.
    python scripts/simulate_cameras.py --synthetic --n 500
    python scripts/simulate_cameras.py --clear-synthetic

Only needs:  pip install requests
"""
import argparse
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

CAMERAS = ["cam_01", "cam_02", "cam_03", "cam_04"]


def random_time(days: int) -> datetime:
    """Random moment in the last `days` days, during a 07:00-19:00 shift."""
    now = datetime.now(timezone.utc)
    day = now - timedelta(days=random.randint(0, days - 1))
    return day.replace(hour=random.randint(7, 18), minute=random.randint(0, 59), second=0, microsecond=0)


def real_mode(api: str, folder: Path, n: int, days: int) -> None:
    images = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        sys.exit(f"No images in {folder}. Copy some test images there first.")
    total_v = 0
    for i in range(n):
        img = random.choice(images)
        with img.open("rb") as f:
            r = requests.post(f"{api}/detect", files={"file": (img.name, f, "image/jpeg")},
                              data={"camera_id": random.choice(CAMERAS),
                                    "captured_at": random_time(days).isoformat()}, timeout=60)
        r.raise_for_status()
        res = r.json()
        total_v += res["n_violations"]
        print(f"[{i + 1}/{n}] {img.name[:40]:40s} {res['camera_id']}  "
              f"violations={res['n_violations']}  {res['latency_ms']} ms")
    print(f"Done: {n} frames, {total_v} violations.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--images", type=Path, default=Path("samples"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--clear-synthetic", action="store_true")
    a = p.parse_args()
    if a.clear_synthetic:
        print(requests.delete(f"{a.api}/demo/synthetic", timeout=30).json())
    elif a.synthetic:
        print(requests.post(f"{a.api}/demo/synthetic", params={"n": a.n, "days": a.days}, timeout=60).json())
    else:
        real_mode(a.api, a.images, a.n, a.days)
