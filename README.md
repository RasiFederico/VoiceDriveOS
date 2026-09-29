# VoiceDriveOS

VoiceDriveOS is a local voice controller for Windows applications and browsers. It listens for a spoken command, transcribes it offline, uses local Laya for screen-target selection, and executes a bounded mouse, keyboard, text, navigation, or application action. It does not use a wake word or a graphical settings interface.

## What it can do

- Read accessible controls in the foreground Windows app through UI Automation; use local OCR to fill in text targets the app does not expose.
- Left, double, right, and middle click; drag between two recognized labels; move the pointer to a label; or click spoken screen coordinates.
- Type dictated text exactly as spoken after the command prefix.
- Press named keys and combinations with two or more keys, such as `Ctrl+Shift+S`.
- Use shortcuts such as save, copy, paste, undo, select all, new tab, close tab, refresh, print, and switch window.
- Scroll, open applications or web addresses, close the active window, navigate back/forward, and wait.

Screen recognition combines Windows UI Automation and RapidOCR. UI Automation provides control roles, names, bounds, and item types where an app exposes them; OCR supplies labels for custom-drawn or inaccessible content. Unique exact labels are matched directly, explicit ordinal requests such as “the first folder” use accessible item types and screen order when available, and Laya multilingual handles semantic paraphrases or ambiguous candidates. Unlabeled icons still need a separate local icon detector or spoken coordinates.

## Try the Windows executable

The executable in `dist\VoiceDriveOS.exe` must be rebuilt after source updates. The build folder also contains `dist\models\whisper-base` and `dist\models\laya-multilingual`; keep these folders beside the EXE. RapidOCR and its CPU models are in the EXE, while UI Automation is provided by Windows.

```powershell
cd C:\Users\feder\Documents\VoiceDriveOS
dist\VoiceDriveOS.exe --check-models
dist\VoiceDriveOS.exe --list-audio-devices
```

If needed, set the microphone index printed by the last command, then start listening:

```powershell
$env:VDOS_AUDIO_INPUT_DEVICE = "1"
dist\VoiceDriveOS.exe
```

`Ctrl+C` stops the assistant. To check screen capture and OCR without opening the microphone or performing an action, run:

```powershell
dist\VoiceDriveOS.exe --check-vision
```

This captures one current screen frame and prints how many UIA/OCR targets were found. It does not save the screenshot.

### Whisper model choices

Base is the default balance between recognition quality and size. Tiny is about 75 MB and starts faster on slower computers; Base is about 145 MB. To use Tiny, download it and select it in the same PowerShell session:

```powershell
dist\VoiceDriveOS.exe --download-model tiny
$env:VDOS_AUDIO_WHISPER_MODEL = "tiny"
dist\VoiceDriveOS.exe
```

The supplied Base and Laya model folders let this build run offline. Internet is needed only to download updated assets or an alternate Whisper model. Listening, model inference, screen parsing, and actions run locally without a hosted inference API.

### Local Laya decision model

The build uses `convaiinnovations/laya-multilingual` through the local Python `laya` package. It chooses among screen candidates; it is not a generative assistant and cannot invent new action types. A unique, exact label match is handled directly; Laya handles semantic paraphrases and ambiguous candidates. Its weights are Apache-2.0. Set `VDOS_INTELLIGENCE_MODEL=mock` to use the simple deterministic matcher instead, or set `VDOS_INTELLIGENCE_LAYA_MODEL_PATH` to a local checkpoint directory.

The screen parser prefers accessible controls and falls back to OCR. Laya's choice confidence is not fully calibrated for VoiceDriveOS, so validate it against real Italian commands and keep the conservative no-action behaviour for uncertain targets.

## Voice commands

Speak one command at a time. Examples in Italian and English:

| Command | Examples |
|---|---|
| Click a visible label | `clicca su Impostazioni`, `click the Save button` |
| Double/right/middle click a label | `doppio clic su File`, `right click the document` |
| Click screen coordinates | `clicca alle coordinate 800 420`, `click at x 800 y 420` |
| Drag between labels | `trascina il file sul cestino`, `drag the tab to the left` |
| Move the pointer | `sposta il puntatore su Impostazioni`, `move the mouse to Search` |
| Type exact text | `scrivi Ciao, come stai?`, `type hello world` |
| Press a key or combination | `premi invio`, `premi Ctrl Maiusc S`, `press F11` |
| Common shortcuts | `salva`, `copia`, `incolla`, `undo`, `select all`, `new tab` |
| Scroll | `scorri giù`, `scroll down by 5` |
| Open an app or web address | `apri Blocco note`, `open chrome`, `open https://example.com` |
| Window/browser navigation | `chiudi questa finestra`, `indietro`, `go forward` |
| Wait | `aspetta 2 secondi`, `wait 2 seconds` |

For key combinations, VoiceDriveOS accepts spoken key names or plus-separated names (for example, `Ctrl+Alt+Canc`). The command must identify a supported key; unknown key phrases are ignored.

## Development

Python 3.11 or newer is supported. From PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_models.py
python -m app.main --check-models
python -m app.main
```

Run the automated tests with:

```powershell
python -m pytest tests -q
```

Build a Windows executable with:

```powershell
python -m pip install -r requirements-build.txt
python scripts\build.py
```

The download step saves Laya Multilingual and Whisper Base under `models\`; the build step copies them beside the EXE under `dist\models`. RapidOCR's CPU engine and OCR models are bundled in the executable. Build on Windows for a Windows executable.

## Configuration

Settings can be supplied as environment variables prefixed with `VDOS_`:

| Variable | Default | Purpose |
|---|---:|---|
| `VDOS_AUDIO_WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, or `large-v3` |
| `VDOS_AUDIO_INPUT_DEVICE` | system default | Microphone index from `--list-audio-devices` |
| `VDOS_AUDIO_WHISPER_LANGUAGE` | auto | Optional speech language, for example `it` |
| `VDOS_VISION_ENABLED` | `true` | Enable screenshot based commands |
| `VDOS_VISION_CAPTURE_SCALE` | `0.75` | Image scale passed to screen OCR |
| `VDOS_VISION_OCR_LANGUAGE` | `it` | OCR language; RapidOCR supports language-specific models |
| `VDOS_INTELLIGENCE_MODEL` | `laya` | `laya` or `mock` local decision core |
| `VDOS_INTELLIGENCE_LAYA_MODEL_PATH` | unset | Load a Laya checkpoint from a local directory |
| `VDOS_INTELLIGENCE_CONFIDENCE_THRESHOLD` | `0.6` | Minimum match confidence for visual actions |

## Processing flow

```text
Microphone → voice activity detection → local Whisper transcription
    → bounded Italian/English command parsing
    → (foreground UI Automation + OCR for screen commands)
    → local Laya candidate selection
    → validated action → mouse/keyboard/clipboard executor
```

The screenshot is captured only when a command needs screen context. The decision layer can only select among detected screen elements; it cannot generate code or run shell commands. Wake word activation and a graphical application interface are intentionally left for a later phase.
