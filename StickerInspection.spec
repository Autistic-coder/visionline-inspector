# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_submodules
from pathlib import Path
import importlib.util


datas = []
binaries = []
hiddenimports = []

for package in (
    "ultralytics",
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "PIL",
    "yaml",
    "PySide6",
):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    except Exception:
        try:
            hiddenimports += collect_submodules(package)
        except Exception:
            pass

try:
    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec and cv2_spec.origin:
        cv2_dir = Path(cv2_spec.origin).parent
        for dll in cv2_dir.glob("opencv_videoio_ffmpeg*.dll"):
            binaries.append((str(dll), "cv2"))
except Exception:
    pass


a = Analysis(
    ["desktop_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StickerInspection",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StickerInspection",
)
