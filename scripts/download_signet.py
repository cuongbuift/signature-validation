"""
Download the pretrained SigNet model from the sigver project.
Usage: python scripts/download_signet.py
"""
import io
import zipfile
from pathlib import Path

GDRIVE_ID = "1l8NFdxSvQSLb2QTv71E6bKcTgvShKPpx"
DEST = Path("models/signet.pth")


def download_from_gdrive(file_id: str, dest: Path):
    import urllib.request

    print(f"Downloading SigNet model to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use gdown if available, otherwise fall back to requests
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        # The zip contains signet.pth inside
        zip_path = dest.parent / "signet_models.zip"
        # The file is directly a .pth (not a zip)
        gdown.download(id=file_id, output=str(dest), quiet=False)
        print(f"Model saved to {dest}")
    except ImportError:
        print("gdown not found. Install with: pip install gdown")
        print("Then re-run this script.")
        raise


if __name__ == "__main__":
    if DEST.exists():
        print(f"Model already exists at {DEST}")
    else:
        download_from_gdrive(GDRIVE_ID, DEST)
