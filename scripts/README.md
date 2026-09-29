# Scripts

## Prepare the offline speech model

```powershell
python scripts\download_whisper_model.py --model base
```

The setup step needs internet once. Runtime loading is local-only by default.
Set `VDOS_AUDIO_WHISPER_CACHE_DIR` before downloading and when running the app
to use a separate cache directory. The `tiny` model is available for slower
computers.

## Build the executable

```powershell
python -m pip install -r requirements-build.txt
python scripts\build.py
```

The build produces an executable under `dist/` for the current operating
system. Whisper weights stay outside the executable; RapidOCR and its local
models are bundled. On Windows, use `VoiceDriveOS.exe --download-model base`
to prepare the speech model, `--check-models` to load the models without
starting the microphone, and `--check-vision` to verify screenshot OCR.
