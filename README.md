# VisionLine Inspector

YOLOv8-powered desktop app for production line visual inspection.

VisionLine Inspector is a PySide6 desktop application for running YOLOv8 visual inspection on a left/right production-line camera setup. It supports local video testing and RTSP/live camera mode while keeping model files, datasets, logs, and private configuration outside the public repository.

## Key Features

- PySide6 desktop UI with two inspection feeds.
- YOLOv8 object detection through Ultralytics.
- Local video test mode for left, right, or both feeds.
- RTSP/live camera mode with reconnect handling.
- ROI/search-zone and expected-zone support.
- Inspection-cycle state machine for PLA card OK/NG decisions.
- Optional debug overlays, cycle logs, and NG/low-confidence evidence capture.
- GPU/CUDA support when available, with CPU fallback.

## Tech Stack

- Python
- PySide6
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyTorch / TorchVision
- PyYAML

## Folder Structure

```text
.
|-- desktop_app.py                 # Main PySide6 inspection app
|-- run_desktop_app.bat            # Windows launcher
|-- run_app.bat                    # Compatibility launcher
|-- config_app.example.json        # Public-safe app config template
|-- camera_config.example.ini      # Public-safe camera config template
|-- requirements.txt               # Runtime/training dependencies
|-- retrain_model.py               # Dataset merge/retraining helper
|-- train_sticker_yolov8.py        # Basic training helper
|-- test_sticker_model.py          # Model test helper
|-- prepare_sticker_dataset.py     # Dataset preparation helper
|-- generate_roi_from_annotations.py
`-- package_runtime.py             # Local packaging helper
```

The following are intentionally excluded from GitHub: datasets, videos, images, labels, logs, generated runs, private camera configs, local configs, build folders, dist folders, and executables.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Copy the public template and fill in your own local values:

```powershell
Copy-Item config_app.example.json config_app.json
```

Edit `config_app.json` and set:

- `MODEL_PATH` to your local YOLO model, for example `models/best.pt`.
- ROI and expected-zone values for your camera geometry.
- `LOGO_PATH` only if you want a local logo asset.
- inspection thresholds and debug settings for your environment.

For live cameras, copy the camera template:

```powershell
Copy-Item camera_config.example.ini camera_config.ini
```

Then edit `camera_config.ini` with your own camera usernames, passwords, IP addresses, and RTSP paths. Do not commit real camera details.

## Running The Desktop App

```powershell
.\run_desktop_app.bat
```

The compatibility launcher also works:

```powershell
.\run_app.bat
```

## Local Video Test Mode

Keep `[app] mode=video_test` in `camera_config.ini`, or leave the camera config absent. Start the app, select a left video and/or right video, then press `Start`. If only one side is selected, only that side runs.

## RTSP / Live Camera Mode

Copy `camera_config.example.ini` to `camera_config.ini`, set `[app] mode=live`, and provide your own local RTSP camera URLs:

```ini
source=rtsp://USERNAME:PASSWORD@CAMERA_IP:554/Streaming/Channels/101
```

If a password contains special characters, URL encode it before placing it in the local-only config file.

## Model Weights

This repository includes the available YOLO model weights for convenience. You can also provide your own trained YOLO model and point `MODEL_PATH` in `config_app.json` to that file.

The app searches common local model locations such as:

```text
models/best.pt
models/best_sticker_yolov8.pt
runs/detect/train/weights/best.pt
```

If you replace or retrain the model, keep any private datasets and generated training runs out of Git.

## Retraining

If you use `retrain_model.py`, keep datasets and trained outputs local. A typical flow is:

```powershell
python prepare_sticker_dataset.py
python retrain_model.py
python test_sticker_model.py
```

Training data, labels, generated runs, and prediction images are excluded from GitHub. Review and adapt dataset paths for your own workstation before training.

## Debugging And Logs

The app can write runtime logs, inspection-cycle JSONL logs, NG evidence snapshots, and low-confidence evidence snapshots. These outputs are useful for local debugging but are intentionally ignored by Git.

## Public Safety

This repository excludes:

- `config_app.json`
- `camera_config.ini`
- `.env` files
- credentials and tokens
- RTSP URLs with real credentials
- private IP addresses
- datasets, videos, images, and labels
- logs and debug snapshots
- `build/`, `dist/`, and `.exe` files

Before making a fork or release public, confirm that only template configs and source files are committed.
