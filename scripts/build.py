"""Build the single-file VoiceDriveOS executable with PyInstaller."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    model_sources = {
        "laya-multilingual": ROOT / "models" / "laya-multilingual",
        "whisper-base": ROOT / "models" / "whisper-base",
    }
    missing_models = [name for name, path in model_sources.items() if not path.is_dir()]
    if missing_models:
        print(
            "Local model assets are missing: " + ", ".join(missing_models)
            + ". Run `python scripts\\download_models.py` first.",
            file=sys.stderr,
        )
        return 1

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print(
            "PyInstaller is not installed. Run `python -m pip install -r requirements-build.txt` first.",
            file=sys.stderr,
        )
        return 1

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        str(ROOT / "build.spec"),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        return completed.returncode

    executable_name = "VoiceDriveOS.exe" if sys.platform == "win32" else "VoiceDriveOS"
    executable = ROOT / "dist" / executable_name
    if not executable.is_file():
        print(f"Build finished without the expected executable: {executable}", file=sys.stderr)
        return 1

    model_destination = ROOT / "dist" / "models"
    model_destination.mkdir(parents=True, exist_ok=True)
    for name, source in model_sources.items():
        shutil.copytree(source, model_destination / name, dirs_exist_ok=True)

    print(f"Build complete: {executable}")
    print(f"Offline models copied beside the executable: {model_destination}")
    print("The application can run offline with the Base and Laya model folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
