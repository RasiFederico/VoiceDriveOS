# Local model setup

VoiceDriveOS uses local Laya multilingual for target decisions and
faster-whisper Base for speech recognition. Download both once while connected
to the internet:

```powershell
python scripts\download_models.py
```

The script saves the models in `models\laya-multilingual` and
`models\whisper-base`. `scripts\build.py` copies these folders beside the
executable under `dist\models`; VoiceDriveOS then uses them locally without a
Hugging Face cache or model downloads at startup. RapidOCR and its CPU inference
models are included in the application build. Windows UI Automation is
provided by `pywinauto`. The icon detector file in `weights\icon_detect_v3` is
not used by the current OCR/UI Automation parser.
