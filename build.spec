# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller single-file build; model folders are copied beside the EXE by scripts/build.py."""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs

faster_whisper_datas, faster_whisper_binaries, faster_whisper_hidden = collect_all("faster_whisper")
ctranslate2_datas, ctranslate2_binaries, ctranslate2_hidden = collect_all("ctranslate2")
sounddevice_datas = collect_data_files("_sounddevice_data")
sounddevice_binaries = collect_dynamic_libs("_sounddevice_data")
rapidocr_datas, rapidocr_binaries, rapidocr_hidden = collect_all("rapidocr")
onnxruntime_datas, onnxruntime_binaries, onnxruntime_hidden = collect_all("onnxruntime")
shapely_datas, shapely_binaries, shapely_hidden = collect_all("shapely")
pywinauto_datas, pywinauto_binaries, pywinauto_hidden = collect_all("pywinauto")
laya_datas, laya_binaries, laya_hidden = collect_all("laya")
torch_datas, torch_binaries, torch_hidden = collect_all("torch")
transformers_datas, transformers_binaries, transformers_hidden = collect_all("transformers")

block_cipher = None

a = Analysis(
    ["app/main.py"],
    pathex=["."],
    binaries=(
        faster_whisper_binaries + ctranslate2_binaries + sounddevice_binaries
        + rapidocr_binaries + onnxruntime_binaries + shapely_binaries
        + pywinauto_binaries + laya_binaries + torch_binaries
        + transformers_binaries
    ),
    datas=(
        faster_whisper_datas + ctranslate2_datas + sounddevice_datas
        + rapidocr_datas + onnxruntime_datas + shapely_datas
        + pywinauto_datas + laya_datas + torch_datas + transformers_datas
    ),
    hiddenimports=[
        "sounddevice",
        "faster_whisper",
        "ctranslate2",
        "pyautogui",
        "pyperclip",
        "mss",
        "PIL",
        "numpy",
        "rapidocr",
        "cv2",
        "onnxruntime",
        "pywinauto",
        "laya",
        "torch",
        "transformers",
    ] + faster_whisper_hidden + ctranslate2_hidden + rapidocr_hidden + onnxruntime_hidden + shapely_hidden + pywinauto_hidden + laya_hidden + torch_hidden + transformers_hidden + ["_sounddevice_data"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VoiceDriveOS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
