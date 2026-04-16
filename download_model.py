"""
Download / set up the YOLO26 signature detection model.

YOLO26 is the latest Ultralytics architecture with no-NMS end-to-end
inference and MuSGD optimizer — up to 43 % faster CPU inference than
previous YOLO generations.

Workflow
--------
1. Run this script to download the pretrained YOLO26n base weights.
2. Fine-tune on a signature dataset (see instructions below).
3. Place the fine-tuned weights at  models/signature_yolo.pt  and restart
   the server.

If you skip fine-tuning the server will still run — detection is skipped
and the full image is passed to the preprocessing pipeline.

Run:
    python download_model.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MODEL_DIR  = Path("models")
MODEL_FILE = MODEL_DIR / "signature_yolo.pt"

# Base pretrained YOLO26 weight (nano = fastest on CPU)
# Change to yolo26s / yolo26m for higher accuracy at the cost of speed.
YOLO26_BASE = "yolo26n.pt"


def download_base_weights() -> Path | None:
    """Download the pretrained YOLO26n base weights via ultralytics."""
    try:
        from ultralytics import YOLO

        print(f"Downloading pretrained {YOLO26_BASE} via ultralytics …")
        model = YOLO(YOLO26_BASE)   # auto-downloads to ultralytics cache
        # Resolve where ultralytics cached the file
        import inspect, pathlib
        cached = pathlib.Path(inspect.getfile(model.__class__)).parent.parent / YOLO26_BASE
        if not cached.exists():
            # ultralytics may store it in the current dir on first download
            cached = Path(YOLO26_BASE)
        return cached if cached.exists() else None
    except Exception as e:
        print(f"  ultralytics download failed: {e}")
        return None


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_FILE.exists():
        print(f"Model already present at '{MODEL_FILE}'.")
        print("Delete it and re-run this script to refresh.")
        return

    cached = download_base_weights()

    if cached and cached.exists():
        shutil.copy(cached, MODEL_FILE)
        size_mb = MODEL_FILE.stat().st_size / 1024 / 1024
        print(f"\n✓  YOLO26n base weights saved to '{MODEL_FILE}' ({size_mb:.1f} MB)")
        print(
            "\nNOTE: This is the generic COCO-pretrained model.\n"
            "For best results, fine-tune on a signature dataset:\n\n"
            "  from ultralytics import YOLO\n"
            "  model = YOLO('yolo26n.pt')\n"
            "  model.train(data='signature.yaml', epochs=50, imgsz=640)\n"
            "  # then copy runs/detect/train/weights/best.pt → models/signature_yolo.pt\n\n"
            "Public signature datasets:\n"
            "  https://universe.roboflow.com/search?q=signature\n"
        )
    else:
        print(
            "\n✗  Automatic download failed.\n"
            f"  Manually download '{YOLO26_BASE}' from:\n"
            "  https://github.com/ultralytics/assets/releases\n"
            f"  and copy it to:  {MODEL_FILE.resolve()}\n\n"
            "  The server will run without YOLO detection (full-image fallback)\n"
            "  until the model file is present."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
