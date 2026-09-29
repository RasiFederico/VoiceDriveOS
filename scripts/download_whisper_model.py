"""Download one faster-whisper model for later offline use."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a Whisper model once so VoiceDriveOS can run offline"
    )
    parser.add_argument(
        "--model",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        default="base",
        help="model to download (default: base; tiny is available for lower-end PCs)",
    )
    args = parser.parse_args()

    try:
        from faster_whisper.utils import download_model
    except ImportError:
        parser.error("faster-whisper is not installed; run `python -m pip install -r requirements.txt`")

    cache_dir = os.environ.get("VDOS_AUDIO_WHISPER_CACHE_DIR")
    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        download_root = str(cache_path)
    else:
        download_root = None

    print(f"Downloading/validating Whisper model '{args.model}'. This needs internet access once.")
    try:
        model_path = download_model(
            args.model,
            cache_dir=download_root,
            local_files_only=False,
        )
    except Exception as exc:
        print(f"Could not download the model: {exc}")
        return 1

    print(
        f"Model '{args.model}' is cached at {model_path}. VoiceDriveOS will use the local cache "
        "and will not download weights during startup."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
