from __future__ import annotations

import json
import os
import shutil
import stat
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = Path.home() / "Desktop"
RUNTIME_DIR = DESKTOP_DIR / "sticker app"
DIST_DIR = PROJECT_DIR / "dist" / "StickerInspection"
REPORT_PATH = PROJECT_DIR / "build_report.txt"
CAMERA_CONFIG_NAME = "camera_config.ini"
DEFAULT_CAMERA_CONFIG_TEXT = """# VisionLine Inspector camera config
# Set [app] mode=live to open the RTSP cameras automatically on startup.
# Keep mode=video_test to use the existing selected-video workflow.
#
# RTSP example:
# rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101
#
# If the RTSP password has special characters, URL encode them:
# @ should be written as %40
# # should be written as %23
# space should be written as %20

[app]
mode=video_test

[left_camera]
enabled=true
name=LEFT SIDE
source=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101

[right_camera]
enabled=true
name=RIGHT SIDE
source=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101

[stream]
use_ffmpeg=true
reconnect_seconds=2
max_no_frame_seconds=2
prefer_latest_frame=true
"""


def unique_existing(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    found: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file():
            found.append(path)
    return found


def find_model() -> Path:
    config = load_source_config()
    candidates: list[Path] = []
    configured = str(config.get("MODEL_PATH", "")).strip()
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else PROJECT_DIR / configured_path)
    candidates.extend(
        [
            PROJECT_DIR / "models" / "best.pt",
            PROJECT_DIR / "models" / "best_sticker_yolov8.pt",
            PROJECT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]
    )
    candidates.extend(sorted(PROJECT_DIR.glob("runs/*/weights/best.pt")))
    candidates.extend(sorted(PROJECT_DIR.glob("runs/detect/train*/weights/best.pt")))
    candidates.extend(sorted(PROJECT_DIR.rglob("best.pt")))
    candidates.extend(sorted(PROJECT_DIR.rglob("last.pt")))
    found = unique_existing(candidates)
    if not found:
        raise FileNotFoundError("No trained YOLO model file found.")
    return found[0]


def find_logo(config: dict) -> Path | None:
    candidates: list[Path] = []
    configured = str(config.get("LOGO_PATH", "")).strip()
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else PROJECT_DIR / configured_path)

    for root in (PROJECT_DIR / "assets", PROJECT_DIR, DESKTOP_DIR):
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidates.append(root / f"visionline logo{suffix}")
            candidates.append(root / f"VisionLine logo{suffix}")
            candidates.append(root / f"VISIONLINE LOGO{suffix}")
            candidates.append(root / f"line_logo{suffix}")

    found = unique_existing(candidates)
    return found[0] if found else None


def load_source_config() -> dict:
    config_path = PROJECT_DIR / "config_app.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def write_runtime_readme(model_path: Path, config_path: Path, logo_path: Path | None) -> None:
    readme = f"""Sticker Inspection Runtime
==========================

How to run:
1. Double-click StickerInspection.exe
2. Or double-click run_sticker_app.bat to keep a terminal open if the app exits with an error.

Runtime folders:
- Model: models\\best.pt
- Config: config\\app_config.json
- Camera config: camera_config.ini
- Logo: assets\\{logo_path.name if logo_path else 'logo.png (missing, app will show text placeholder)'}
- Logs: logs\\exe_runtime.log
- Live camera logs: logs\\live_camera.log
- Output: output\\

Replacing the model:
1. Close the app.
2. Replace models\\best.pt with your new trained YOLO best.pt.
3. Start the app again.

Changing the logo:
1. Put the new logo image in assets\\.
2. Edit config\\app_config.json and set LOGO_PATH to the relative logo path, for example assets/logo.png.

Checking GPU:
- The top-right app status and the status panel show GPU READY / CUDA when CUDA is available.
- If CUDA is not available, the app falls back to CPU.

Detection status:
- Normal no-detection/waiting is MISSING in grey.
- PLA CARD OK is green and is latched for the current car once the configured rule is satisfied.
- PLA CARD NG is red only after a checked inspection cycle completes without the required sticker detection.
- One-sticker-enough mode passes when either side sees a valid sticker during the car cycle.
- Require-both-stickers mode passes only after both sides have been seen during the car cycle.
- The default confidence threshold is read from config\\app_config.json as CONF_THRESHOLD.

ROI settings:
- Search ROI is used internally for inference/cropping and is larger so moving stickers are not cut off.
- Expected zone is smaller and is used for inspection decisions.
- ROI boxes are hidden by default.
- To show ROI boxes for debugging, edit config\\app_config.json and set "show_roi_box": true and/or "show_expected_zone_box": true.
- To resize ROI, adjust search_roi_margin_ratio, expected_zone_margin_ratio, min/max search ROI, or min/max expected-zone values in config\\app_config.json, then rebuild/regenerate config if needed.

Troubleshooting:
- If the app does not open, run run_sticker_app.bat and check the terminal message.
- Check logs\\exe_runtime.log for model, device, video open, frame read, and display errors.
- For live cameras, edit camera_config.ini beside StickerInspection.exe and set [app] mode=live.
- Use rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101 as the template for a local RTSP stream.
- If an RTSP password contains @, #, or a space, URL encode it as %40, %23, or %20.
- Check logs\\live_camera.log for camera connect, disconnect, and reconnect events.
- Make sure models\\best.pt exists.
- Make sure config\\app_config.json exists.
- If a video reaches the end, the panel will show VIDEO ENDED.

If a video panel is black:
1. Run run_sticker_app.bat.
2. Check logs\\exe_runtime.log.
3. Check whether logs\\debug_left_first_frame.jpg or logs\\debug_right_first_frame.jpg was created.
4. If the debug image was created, video reading works and the issue is display rendering.
5. If the debug image was not created, confirm the selected video path exists and OpenCV can open the file.
6. Try an MP4/H264 or AVI test video.
7. If VideoCapture fails inside the EXE, check that OpenCV FFmpeg DLLs are present under _internal\\cv2.

Build source paths used:
- Source model: {model_path}
- Source config: {config_path}
- Source logo: {logo_path if logo_path else 'not found'}
"""
    (RUNTIME_DIR / "README.txt").write_text(readme, encoding="utf-8")


def write_run_bat() -> None:
    text = """@echo off
cd /d "%~dp0"
StickerInspection.exe
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" (
    echo.
    echo StickerInspection.exe exited with code %EXITCODE%.
    echo Check logs\\exe_runtime.log for details.
)
echo.
pause
"""
    (RUNTIME_DIR / "run_sticker_app.bat").write_text(text, encoding="utf-8")


def ensure_runtime_camera_config() -> str:
    runtime_camera_config = RUNTIME_DIR / CAMERA_CONFIG_NAME
    source_camera_config = PROJECT_DIR / CAMERA_CONFIG_NAME
    if runtime_camera_config.exists():
        return f"Camera config preserved: {runtime_camera_config}"
    if source_camera_config.exists():
        shutil.copy2(source_camera_config, runtime_camera_config)
        return f"Camera config copied from: {source_camera_config}"
    runtime_camera_config.write_text(DEFAULT_CAMERA_CONFIG_TEXT, encoding="utf-8")
    return f"Camera config template created: {runtime_camera_config}"


def clean_known_runtime_items() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("StickerInspection.exe", "_internal", "models", "config", "assets", "README.txt", "run_sticker_app.bat"):
        target = RUNTIME_DIR / name
        if target.is_dir():
            remove_tree(target)
        elif target.exists():
            target.unlink()


def remove_tree(target: Path) -> None:
    def handle_remove_error(function, path, _exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            function(path)
        except Exception:
            pass

    for _ in range(6):
        try:
            shutil.rmtree(target, onerror=handle_remove_error)
            return
        except OSError:
            time.sleep(1.0)
            if not target.exists():
                return

    fallback = target.with_name(f"{target.name}_old_{int(time.time())}")
    target.rename(fallback)


def main() -> None:
    if not DIST_DIR.exists():
        raise FileNotFoundError(f"PyInstaller dist folder missing: {DIST_DIR}")

    source_config_path = PROJECT_DIR / "config_app.json"
    source_config = load_source_config()
    model_path = find_model()
    logo_path = find_logo(source_config)

    clean_known_runtime_items()
    shutil.copytree(DIST_DIR, RUNTIME_DIR, dirs_exist_ok=True)

    models_dir = RUNTIME_DIR / "models"
    config_dir = RUNTIME_DIR / "config"
    assets_dir = RUNTIME_DIR / "assets"
    logs_dir = RUNTIME_DIR / "logs"
    output_dir = RUNTIME_DIR / "output"
    for folder in (models_dir, config_dir, assets_dir, logs_dir, output_dir):
        folder.mkdir(parents=True, exist_ok=True)

    runtime_model = models_dir / "best.pt"
    shutil.copy2(model_path, runtime_model)

    runtime_logo_relative = ""
    runtime_logo = None
    if logo_path:
        suffix = logo_path.suffix.lower() if logo_path.suffix else ".png"
        runtime_logo = assets_dir / f"logo{suffix}"
        shutil.copy2(logo_path, runtime_logo)
        runtime_logo_relative = f"assets/{runtime_logo.name}"

    runtime_config = dict(source_config)
    runtime_config["MODEL_PATH"] = "models/best.pt"
    runtime_config["LOGO_PATH"] = runtime_logo_relative
    runtime_config["OUTPUT_DIR"] = "output"
    runtime_config["LOG_DIR"] = "logs"
    runtime_config_path = config_dir / "app_config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")

    write_runtime_readme(model_path, source_config_path, runtime_logo)
    write_run_bat()
    camera_config_report = ensure_runtime_camera_config()

    report_lines = [
        "Sticker Inspection build report",
        "==============================",
        f"Final runtime folder: {RUNTIME_DIR}",
        f"Final EXE path: {RUNTIME_DIR / 'StickerInspection.exe'}",
        f"Model copied from: {model_path}",
        f"Model copied to: {runtime_model}",
        f"Config copied from: {source_config_path}",
        f"Config copied to: {runtime_config_path}",
        f"Logo copied from: {logo_path if logo_path else 'missing'}",
        f"Logo copied to: {runtime_logo if runtime_logo else 'missing'}",
        camera_config_report,
        "Created folders: models, config, assets, logs, output",
        "Created files: README.txt, run_sticker_app.bat",
    ]
    report = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    (RUNTIME_DIR / "BUILD_REPORT.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
