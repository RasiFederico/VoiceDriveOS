"""Download VoiceDriveOS' local Laya and faster-whisper Base model assets."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def main() -> int:
    downloads = (
        ("convaiinnovations/laya-multilingual", MODELS / "laya-multilingual"),
        ("Systran/faster-whisper-base", MODELS / "whisper-base"),
    )
    for repo_id, local_dir in downloads:
        print(f"Downloading {repo_id} to {local_dir} (internet required)...", flush=True)
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        except Exception as exc:
            print(f"Download failed for {repo_id}: {exc}")
            return 1
        print(f"Ready: {local_dir}", flush=True)
    print("Both models are ready. The build script will copy them beside VoiceDriveOS.exe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
