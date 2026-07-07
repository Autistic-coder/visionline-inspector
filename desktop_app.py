from __future__ import annotations

import contextlib
import configparser
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_VERBOSE", "False")

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

try:
    import yaml
except Exception:  # pragma: no cover - optional fallback
    yaml = None


SOURCE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else SOURCE_DIR
MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config" / "app_config.json" if getattr(sys, "frozen", False) else BASE_DIR / "config_app.json"
CAMERA_CONFIG_PATH = BASE_DIR / "camera_config.ini"
LOG_PATH = BASE_DIR / "logs" / "exe_runtime.log" if getattr(sys, "frozen", False) else BASE_DIR / "app_runtime.log"
APP_LOG_PATH = BASE_DIR / "logs" / "app.log"
LIVE_LOG_PATH = BASE_DIR / "logs" / "live_camera.log"
INSPECTION_CYCLE_LOG_DIR = BASE_DIR / "logs" / "inspection_cycles"
NG_CASE_LOG_DIR = BASE_DIR / "logs" / "ng_cases"
LOW_CONFIDENCE_LOG_DIR = BASE_DIR / "logs" / "low_confidence_cases"

DEFAULT_CONFIG = {
    "MODEL_PATH": "",
    "CONF_THRESHOLD": 0.30,
    "IMG_SIZE": 416,
    "REQUIRE_BOTH_STICKERS": False,
    "SMOOTHING_WINDOW": 12,
    "MIN_PRESENT_FRAMES": 3,
    "USE_GPU": True,
    "USE_FP16": True,
    "FRAME_SKIP": 2,
    "MAX_DISPLAY_WIDTH": 960,
    "MAX_DISPLAY_HEIGHT": 540,
    "LOGO_PATH": "",
    "roi_enabled": True,
    "roi_mode": "crop",
    "roi_source": "annotations",
    "use_search_roi": True,
    "show_roi_box": False,
    "show_debug_roi_box": False,
    "show_expected_zone_box": False,
    "loop_video": False,
    "roi_margin_ratio": 0.20,
    "min_roi_width_ratio": 0.08,
    "min_roi_height_ratio": 0.08,
    "max_roi_width_ratio": 0.35,
    "max_roi_height_ratio": 0.35,
    "search_roi_margin_ratio": 0.40,
    "expected_zone_margin_ratio": 0.20,
    "max_search_roi_width_ratio": 0.70,
    "max_search_roi_height_ratio": 0.60,
    "min_search_roi_width_ratio": 0.25,
    "min_search_roi_height_ratio": 0.20,
    "left_roi": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "right_roi": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "left_search_roi": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "right_search_roi": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "left_expected_zone": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "right_expected_zone": {"x_norm": 0.0, "y_norm": 0.0, "w_norm": 1.0, "h_norm": 1.0},
    "inspection_state_enabled": True,
    "use_confirmed_fail_only": True,
    "show_missing_instead_of_fail": True,
    "missing_status_color": "grey",
    "present_status_color": "green",
    "fail_status_color": "red",
    "fail_requires_active_inspection": True,
    "fail_after_expected_zone_missed": True,
    "missing_grace_frames": 30,
    "fail_confirmation_frames": 20,
    "ng_confirmation_frames": 20,
    "cycle_end_empty_frames": 25,
    "fail_after_missing_frames": 45,
    "pass_hold_frames": 15,
    "no_car_reset_frames": 30,
    "empty_roi_detection_enabled": True,
    "empty_roi_texture_threshold": 10.0,
    "BOX_HOLD_FRAMES": 10,
    "BOX_HOLD_MS": 400,
    "ENABLE_BOX_HOLD": True,
    "SMOOTH_BOXES": True,
    "DEBUG_MODE": False,
}

PRESENT_WORDS = (
    "sticker_present",
    "present",
    "sticker",
    "label",
    "ok",
    "good",
    "pass",
)
MISSING_WORDS = (
    "sticker_missing",
    "missing",
    "not_present",
    "notpresent",
    "absent",
    "defect",
    "ng",
    "fail",
)

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

logger = logging.getLogger("sticker_app")
logger.setLevel(logging.INFO)
if not logger.handlers:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=512_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    app_handler = RotatingFileHandler(APP_LOG_PATH, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    app_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(app_handler)

live_logger = logging.getLogger("sticker_app.live_camera")
live_logger.setLevel(logging.INFO)
if not live_logger.handlers:
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    live_handler = RotatingFileHandler(LIVE_LOG_PATH, maxBytes=512_000, backupCount=3, encoding="utf-8")
    live_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    live_logger.addHandler(live_handler)


FINAL_WAITING = "WAITING"
FINAL_MISSING = "MISSING"
FINAL_OK = "PLA_CARD_OK"
FINAL_NG = "PLA_CARD_NG"


class AppState(Enum):
    IDLE = "IDLE"
    LOADING_MODEL = "LOADING_MODEL"
    RUNNING_VIDEO = "RUNNING_VIDEO"
    RUNNING_CAMERA = "RUNNING_CAMERA"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_bool_config(config: dict[str, Any], key: str, default: bool) -> None:
    value = config.get(key, default)
    if isinstance(value, bool):
        config[key] = value
        return
    logger.warning("Invalid config value for %s=%r; using default %s", key, value, default)
    config[key] = default


def safe_non_negative_int_config(config: dict[str, Any], key: str, default: int) -> None:
    value = config.get(key, default)
    try:
        parsed = int(value)
        if parsed >= 0:
            config[key] = parsed
            return
    except (TypeError, ValueError):
        pass
    logger.warning("Invalid config value for %s=%r; using default %s", key, value, default)
    config[key] = default


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("ENABLE_BOX_HOLD", "SMOOTH_BOXES", "DEBUG_MODE"):
        safe_bool_config(config, key, bool(DEFAULT_CONFIG[key]))
    for key in ("BOX_HOLD_FRAMES", "BOX_HOLD_MS"):
        safe_non_negative_int_config(config, key, int(DEFAULT_CONFIG[key]))
    return config


@contextlib.contextmanager
def quiet_console() -> Any:
    devnull = open(os.devnull, "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
    finally:
        devnull.close()


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config = {**DEFAULT_CONFIG, **loaded}
        except Exception:
            config = DEFAULT_CONFIG.copy()
    else:
        config = DEFAULT_CONFIG.copy()

    legacy_map = {
        "model_path": "MODEL_PATH",
        "confidence": "CONF_THRESHOLD",
        "imgsz": "IMG_SIZE",
        "frame_skip": "FRAME_SKIP",
        "display_width": "MAX_DISPLAY_WIDTH",
        "display_height": "MAX_DISPLAY_HEIGHT",
    }
    for old_key, new_key in legacy_map.items():
        if old_key in config and (new_key not in config or config[new_key] == DEFAULT_CONFIG[new_key]):
            config[new_key] = config[old_key]
    if "conf_threshold" in config:
        config["CONF_THRESHOLD"] = float(config["conf_threshold"])
    return validate_config(config)


def save_config(config: dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


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


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "enabled"):
        return True
    if text in ("0", "false", "no", "off", "disabled"):
        return False
    return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def mask_rtsp_url(url: str) -> str:
    text = str(url).strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
        if not parts.username or parts.password is None:
            return text
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        username = parts.username or ""
        netloc = f"{username}:****@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        if "@" not in text:
            return text
        prefix, suffix = text.rsplit("@", 1)
        if ":" not in prefix:
            return text
        user_prefix = prefix.split(":", 1)[0]
        return f"{user_prefix}:****@{suffix}"


def ensure_camera_config_exists() -> None:
    if CAMERA_CONFIG_PATH.exists():
        return
    try:
        CAMERA_CONFIG_PATH.write_text(DEFAULT_CAMERA_CONFIG_TEXT, encoding="utf-8")
        live_logger.info("Created default camera config template: %s", CAMERA_CONFIG_PATH)
        logger.info("Created default camera config template: %s", CAMERA_CONFIG_PATH)
    except Exception:
        live_logger.exception("Failed to create default camera config template")


def load_camera_config() -> dict[str, Any]:
    ensure_camera_config_exists()
    parser = configparser.ConfigParser()
    try:
        parser.read(CAMERA_CONFIG_PATH, encoding="utf-8-sig")
    except Exception:
        live_logger.exception("Invalid camera_config.ini; falling back to video_test mode")
        return {
            "mode": "video_test",
            "cameras": {"left": {}, "right": {}},
            "stream": {"use_ffmpeg": True, "reconnect_seconds": 2.0, "max_no_frame_seconds": 2.0, "prefer_latest_frame": True},
        }

    mode = parser.get("app", "mode", fallback="video_test").strip().lower()
    if mode not in ("live", "video_test"):
        live_logger.warning("Unknown camera app mode %r; using video_test", mode)
        mode = "video_test"

    cameras: dict[str, dict[str, Any]] = {}
    for side, section in (("left", "left_camera"), ("right", "right_camera")):
        enabled = parser.get(section, "enabled", fallback="false")
        source = parser.get(section, "source", fallback="").strip()
        name = parser.get(section, "name", fallback=f"{side.upper()} SIDE").strip()
        cameras[side] = {
            "enabled": parse_bool(enabled, False),
            "name": name,
            "source": source,
            "masked_source": mask_rtsp_url(source),
        }

    stream = {
        "use_ffmpeg": parse_bool(parser.get("stream", "use_ffmpeg", fallback="true"), True),
        "reconnect_seconds": max(0.5, safe_float(parser.get("stream", "reconnect_seconds", fallback="2"), 2.0)),
        "max_no_frame_seconds": max(0.5, safe_float(parser.get("stream", "max_no_frame_seconds", fallback="2"), 2.0)),
        "prefer_latest_frame": parse_bool(parser.get("stream", "prefer_latest_frame", fallback="true"), True),
    }
    live_logger.info("Selected mode: %s", mode)
    for side in ("left", "right"):
        live_logger.info("%s camera source loaded: enabled=%s source=%s", side, cameras[side]["enabled"], cameras[side]["masked_source"])
    return {"mode": mode, "cameras": cameras, "stream": stream}


def config_model_path_value(model_path: Path) -> str:
    try:
        return str(model_path.resolve().relative_to(BASE_DIR.resolve()))
    except ValueError:
        return str(model_path)


def model_candidates(config_path: str = "") -> list[Path]:
    candidates: list[Path] = []
    if config_path:
        configured = Path(config_path)
        candidates.append(configured if configured.is_absolute() else BASE_DIR / configured)

    candidates.extend(
        [
            MODEL_DIR / "best.pt",
            MODEL_DIR / "best_sticker_yolov8.pt",
            BASE_DIR / "runs" / "detect" / "train" / "weights" / "best.pt",
            BASE_DIR / "best.pt",
            BASE_DIR / "best_sticker_yolov8.pt",
        ]
    )
    candidates.extend(sorted(BASE_DIR.glob("runs/detect/train*/weights/best.pt")))
    candidates.extend(sorted(BASE_DIR.rglob("best.pt")))
    candidates.extend(sorted(BASE_DIR.rglob("*.pt")))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_model_path(config_path: str = "") -> Path | None:
    for candidate in model_candidates(config_path):
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pt":
            return candidate
    return None


def data_yaml_candidates() -> list[Path]:
    return [
        BASE_DIR / "data.yaml",
        BASE_DIR / "dataset.yaml",
        BASE_DIR / "sticker project yolo dataset" / "data.yaml",
        Path.home() / "Desktop" / "sticker project yolo dataset" / "data.yaml",
        Path.home() / "Desktop" / "sticker project annotated" / "data.yaml",
    ]


def names_from_data_yaml() -> dict[int, str]:
    if yaml is None:
        return {}
    for path in data_yaml_candidates():
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names", {})
            if isinstance(names, list):
                return {idx: str(name) for idx, name in enumerate(names)}
            if isinstance(names, dict):
                return {int(idx): str(name) for idx, name in names.items()}
        except Exception:
            continue
    return {}


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def class_status(class_name: str, class_count: int) -> str:
    normalized = normalize_name(class_name)
    if any(word in normalized for word in MISSING_WORDS):
        return "MISSING"
    if any(word in normalized for word in PRESENT_WORDS):
        return "PRESENT"
    if class_count <= 1:
        return "PRESENT"
    return "LOW CONFIDENCE"


def get_final_decision(
    left_result: bool | None,
    right_result: bool | None,
    require_both: bool = False,
) -> str:
    active_results = [value for value in (left_result, right_result) if value is not None]
    if not active_results:
        return FINAL_WAITING
    if require_both:
        if left_result is None or right_result is None:
            return FINAL_WAITING
        return FINAL_OK if left_result and right_result else FINAL_MISSING
    return FINAL_OK if any(active_results) else FINAL_MISSING


def smoothed_present(history: deque[bool], min_present_frames: int) -> bool:
    if not history:
        return False
    required = min(max(1, min_present_frames), len(history))
    return sum(1 for item in history if item) >= required


def get_final_decision_from_states(side_states: dict[str, str | None], require_both: bool = False) -> str:
    active_states = [state for state in side_states.values() if state]
    if not active_states:
        return FINAL_WAITING
    present_states = {"PASS", "STICKER_PRESENT", FINAL_OK}
    fail_states = {"FAIL", "CONFIRMED_FAIL", FINAL_NG}
    if require_both:
        if len(active_states) < 2:
            if active_states[0] in present_states:
                return FINAL_MISSING
            if active_states[0] in fail_states:
                return FINAL_NG
            return FINAL_MISSING
        if all(state in present_states for state in active_states):
            return FINAL_OK
        if any(state in fail_states for state in active_states):
            return FINAL_NG
        if any(state == "MISSING" for state in active_states):
            return FINAL_MISSING
        return FINAL_MISSING
    if any(state in present_states for state in active_states):
        return FINAL_OK
    if any(state in fail_states for state in active_states):
        return FINAL_NG
    if any(state == "MISSING" for state in active_states):
        return FINAL_MISSING
    return FINAL_MISSING


def color_for_class(class_id: int) -> tuple[int, int, int]:
    hue = int((class_id * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def status_color(status: str) -> tuple[int, int, int]:
    if status in ("PRESENT", "STICKER_PRESENT", "PASS", FINAL_OK):
        return 40, 190, 80
    if status == "MISSING" or status == FINAL_MISSING:
        return 160, 160, 160
    if status in ("FAIL", "CONFIRMED_FAIL", FINAL_NG):
        return 45, 55, 230
    if status == "LOW CONFIDENCE":
        return 0, 170, 255
    return 160, 160, 160


def final_decision_label(decision: str) -> str:
    if decision in (FINAL_OK, "PASS"):
        return "PLA CARD OK"
    if decision in (FINAL_NG, "FAIL", "CONFIRMED_FAIL"):
        return "PLA CARD NG"
    if decision == FINAL_WAITING:
        return "WAITING"
    return "MISSING"


def format_path(path: str) -> str:
    if not path:
        return "-"
    return str(Path(path))


def display_status(status: str) -> str:
    if status == "CONFIRMED_FAIL":
        return "NG"
    if status in ("PASS", "STICKER_PRESENT", FINAL_OK):
        return "PLA CARD PRESENT"
    if status in ("FAIL", FINAL_NG):
        return "NG"
    return status


def find_logo_path(config: dict[str, Any]) -> Path | None:
    configured = str(config.get("LOGO_PATH", "")).strip()
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else BASE_DIR / configured_path)

    desktop = Path.home() / "Desktop"
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidates.append(desktop / f"visionline logo{suffix}")
        candidates.append(desktop / f"VisionLine logo{suffix}")
        candidates.append(desktop / f"VISIONLINE LOGO{suffix}")
        candidates.append(desktop / f"line logo{suffix}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    logger.warning("Logo image not found; using text placeholder")
    return None


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    above_threshold: bool


@dataclass
class FrameAnalysis:
    status: str
    detected_class: str
    highest_confidence: float
    total_detections: int
    detections: list[Detection]
    sticker_present: bool = False


def detection_to_dict(det: Detection | None) -> dict[str, Any] | None:
    if det is None:
        return None
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "box": list(det.box),
        "above_threshold": det.above_threshold,
    }


def save_frame_snapshot(path: Path, frame: np.ndarray | None) -> bool:
    if frame is None or frame.size == 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]))


def save_cycle_evidence(
    folder: Path,
    cycle_log: dict[str, Any],
    frames: dict[str, np.ndarray],
    best_frames: dict[str, np.ndarray | None],
) -> None:
    cycle_id = int(cycle_log.get("cycle_id", 0))
    stamp = str(cycle_log.get("end_time", utc_timestamp())).replace(":", "").replace("-", "")
    case_dir = folder / f"cycle_{cycle_id}_{stamp}"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "cycle_log.json").write_text(json.dumps(cycle_log, indent=2, default=str), encoding="utf-8")
    for side in ("left", "right"):
        current = frames.get(side)
        best = best_frames.get(side)
        if current is not None:
            save_frame_snapshot(case_dir / f"{side}_snapshot.jpg", current)
        if best is not None:
            save_frame_snapshot(case_dir / f"{side}_best_frame.jpg", best)


def inspection_mode_label(require_both: bool) -> str:
    return "require-both-stickers" if require_both else "one-sticker-enough"


def new_cycle_state() -> dict[str, Any]:
    return {
        "current_car_active": False,
        "inspection_cycle_id": 0,
        "cycle_start_time": "",
        "cycle_start_frame": 0,
        "left_seen_this_car": False,
        "right_seen_this_car": False,
        "left_current_visible": False,
        "right_current_visible": False,
        "left_best_confidence": 0.0,
        "right_best_confidence": 0.0,
        "left_best_raw_confidence": 0.0,
        "right_best_raw_confidence": 0.0,
        "car_has_completed_zone": False,
        "expected_zone_checked": False,
        "checked_frames": 0,
        "empty_frames": 0,
        "final_result_for_current_car": FINAL_WAITING,
        "decision_reason": "waiting for inspection cycle",
        "result_latched_until_reset_or_next_car": False,
        "last_result_reported": FINAL_WAITING,
        "last_ok_latch_frame": 0,
        "last_ng_frame": 0,
        "last_completed_cycle_log": None,
        "last_completed_cycle_saved": 0,
    }


def cycle_success_met(cycle: dict[str, Any], require_both: bool) -> bool:
    left_seen = bool(cycle.get("left_seen_this_car", False))
    right_seen = bool(cycle.get("right_seen_this_car", False))
    return (left_seen and right_seen) if require_both else (left_seen or right_seen)


def set_cycle_result(cycle: dict[str, Any], result: str, reason: str, frame_number: int) -> None:
    previous = str(cycle.get("final_result_for_current_car", FINAL_WAITING))
    if previous == FINAL_OK and result == FINAL_NG:
        logger.info(
            "Prevented PLA CARD OK downgrade at frame %s: %s",
            frame_number,
            reason,
        )
        return
    if previous != result:
        logger.info("Final decision changed: %s -> %s (%s)", previous, result, reason)
    cycle["final_result_for_current_car"] = result
    cycle["decision_reason"] = reason
    cycle["result_latched_until_reset_or_next_car"] = result in (FINAL_OK, FINAL_NG)


def start_inspection_cycle(cycle: dict[str, Any], frame_number: int, reason: str, require_both: bool) -> None:
    previous_result = str(cycle.get("final_result_for_current_car", FINAL_WAITING))
    cycle["inspection_cycle_id"] = int(cycle.get("inspection_cycle_id", 0)) + 1
    cycle["cycle_start_time"] = utc_timestamp()
    cycle["cycle_start_frame"] = frame_number
    cycle["current_car_active"] = True
    cycle["left_seen_this_car"] = False
    cycle["right_seen_this_car"] = False
    cycle["left_current_visible"] = False
    cycle["right_current_visible"] = False
    cycle["left_best_confidence"] = 0.0
    cycle["right_best_confidence"] = 0.0
    cycle["left_best_raw_confidence"] = 0.0
    cycle["right_best_raw_confidence"] = 0.0
    cycle["car_has_completed_zone"] = False
    cycle["expected_zone_checked"] = False
    cycle["checked_frames"] = 0
    cycle["empty_frames"] = 0
    cycle["last_ok_latch_frame"] = 0
    cycle["last_ng_frame"] = 0
    cycle["result_latched_until_reset_or_next_car"] = False
    cycle["final_result_for_current_car"] = FINAL_MISSING
    cycle["decision_reason"] = f"cycle started: {reason}"
    logger.info(
        "Inspection cycle %s started at frame %s: reason=%s mode=%s previous_result=%s",
        cycle["inspection_cycle_id"],
        frame_number,
        reason,
        inspection_mode_label(require_both),
        previous_result,
    )
    if previous_result != FINAL_MISSING:
        logger.info("Final decision changed: %s -> %s (new inspection cycle)", previous_result, FINAL_MISSING)


def build_cycle_log(
    cycle: dict[str, Any],
    frame_number: int,
    final_decision: str,
    decision_reason: str,
    require_both: bool,
) -> dict[str, Any]:
    left_seen = bool(cycle.get("left_seen_this_car", False))
    right_seen = bool(cycle.get("right_seen_this_car", False))
    return {
        "cycle_id": int(cycle.get("inspection_cycle_id", 0)),
        "start_time": str(cycle.get("cycle_start_time", "")),
        "end_time": utc_timestamp(),
        "start_frame": int(cycle.get("cycle_start_frame", 0)),
        "end_frame": frame_number,
        "left_sticker_seen": left_seen,
        "right_sticker_seen": right_seen,
        "best_left_confidence": float(cycle.get("left_best_confidence", 0.0)),
        "best_right_confidence": float(cycle.get("right_best_confidence", 0.0)),
        "best_left_raw_confidence": float(cycle.get("left_best_raw_confidence", 0.0)),
        "best_right_raw_confidence": float(cycle.get("right_best_raw_confidence", 0.0)),
        "checked_frame_count": int(cycle.get("checked_frames", 0)),
        "expected_zone_checked": bool(cycle.get("expected_zone_checked", False)),
        "inspection_mode": inspection_mode_label(require_both),
        "final_decision": final_decision,
        "decision_reason": decision_reason,
    }


def record_completed_cycle(cycle: dict[str, Any], cycle_log: dict[str, Any]) -> None:
    cycle["last_completed_cycle_log"] = cycle_log
    cycle["last_completed_cycle_saved"] = 0
    append_jsonl(INSPECTION_CYCLE_LOG_DIR / "inspection_cycles.jsonl", cycle_log)
    logger.info("Inspection cycle log saved: %s", cycle_log)


def finish_inspection_cycle(
    cycle: dict[str, Any],
    frame_number: int,
    require_both: bool,
    config: dict[str, Any],
) -> None:
    cycle["current_car_active"] = False
    cycle["car_has_completed_zone"] = True
    cycle_id = int(cycle.get("inspection_cycle_id", 0))
    left_seen = bool(cycle.get("left_seen_this_car", False))
    right_seen = bool(cycle.get("right_seen_this_car", False))
    mode = inspection_mode_label(require_both)
    logger.info(
        "Inspection cycle %s ended at frame %s: left_seen_this_car=%s right_seen_this_car=%s mode=%s checked_frames=%s final_before_end=%s",
        cycle_id,
        frame_number,
        left_seen,
        right_seen,
        mode,
        int(cycle.get("checked_frames", 0)),
        cycle.get("final_result_for_current_car", FINAL_MISSING),
    )

    if cycle_success_met(cycle, require_both):
        reason = "successful car left inspection zone"
        cycle_log = build_cycle_log(cycle, frame_number, FINAL_OK, reason, require_both)
        record_completed_cycle(cycle, cycle_log)
        previous = str(cycle.get("final_result_for_current_car", FINAL_MISSING))
        cycle["left_seen_this_car"] = False
        cycle["right_seen_this_car"] = False
        cycle["left_current_visible"] = False
        cycle["right_current_visible"] = False
        cycle["expected_zone_checked"] = False
        cycle["checked_frames"] = 0
        cycle["empty_frames"] = 0
        cycle["result_latched_until_reset_or_next_car"] = False
        cycle["final_result_for_current_car"] = FINAL_MISSING
        logger.info("Final decision changed: %s -> %s (successful car left inspection zone)", previous, FINAL_MISSING)
        return

    minimum_checked_frames = int(config.get("ng_confirmation_frames", config.get("fail_confirmation_frames", 20)))
    if bool(cycle.get("expected_zone_checked", False)) and int(cycle.get("checked_frames", 0)) >= minimum_checked_frames:
        reason = (
            "current cycle completed, no valid left or right detection observed in one-sticker-enough mode."
            if not require_both
            else "current cycle completed, both required stickers were not observed."
        )
        set_cycle_result(cycle, FINAL_NG, f"PLA CARD NG confirmed: {reason}", frame_number)
        cycle["last_ng_frame"] = frame_number
        cycle_log = build_cycle_log(cycle, frame_number, FINAL_NG, reason, require_both)
        record_completed_cycle(cycle, cycle_log)
        logger.info(
            "PLA CARD NG confirmed: %s cycle=%s left_seen_this_car=%s right_seen_this_car=%s",
            reason,
            cycle_id,
            left_seen,
            right_seen,
        )
    else:
        previous = str(cycle.get("final_result_for_current_car", FINAL_MISSING))
        cycle["result_latched_until_reset_or_next_car"] = False
        cycle["final_result_for_current_car"] = FINAL_MISSING
        cycle["decision_reason"] = "inspection ended without confirmed NG"
        cycle_log = build_cycle_log(
            cycle,
            frame_number,
            FINAL_MISSING,
            "inspection ended without enough checked frames for NG",
            require_both,
        )
        record_completed_cycle(cycle, cycle_log)
        logger.info(
            "Inspection cycle %s ended without NG: checked_frames=%s required=%s expected_zone_checked=%s",
            cycle_id,
            int(cycle.get("checked_frames", 0)),
            minimum_checked_frames,
            bool(cycle.get("expected_zone_checked", False)),
        )
        if previous != FINAL_MISSING:
            logger.info("Final decision changed: %s -> %s (no confirmed failure)", previous, FINAL_MISSING)


def update_inspection_cycle(
    cycle: dict[str, Any],
    active_sides: list[str],
    current_visible: dict[str, bool],
    roi_content: dict[str, bool],
    expected_zone_content: dict[str, bool],
    frame_number: int,
    require_both: bool,
    config: dict[str, Any],
    accepted_confidence: dict[str, float] | None = None,
    raw_confidence: dict[str, float] | None = None,
) -> str:
    if not bool(config.get("inspection_state_enabled", True)):
        return FINAL_OK if any(current_visible.get(side, False) for side in active_sides) else FINAL_MISSING

    content_active = any(
        bool(current_visible.get(side, False))
        or bool(roi_content.get(side, False))
        or bool(expected_zone_content.get(side, False))
        for side in active_sides
    )
    visible_sides = [side for side in active_sides if bool(current_visible.get(side, False))]
    if content_active and not bool(cycle.get("current_car_active", False)):
        reason = "valid sticker candidate appeared" if visible_sides else "inspection ROI content appeared"
        start_inspection_cycle(cycle, frame_number, reason, require_both)

    if not bool(cycle.get("current_car_active", False)):
        if str(cycle.get("final_result_for_current_car", FINAL_WAITING)) == FINAL_WAITING and active_sides:
            set_cycle_result(cycle, FINAL_MISSING, "waiting between cars", frame_number)
        return str(cycle.get("final_result_for_current_car", FINAL_MISSING))

    accepted_confidence = accepted_confidence or {}
    raw_confidence = raw_confidence or {}
    for side in ("left", "right"):
        cycle[f"{side}_best_confidence"] = max(
            float(cycle.get(f"{side}_best_confidence", 0.0)),
            float(accepted_confidence.get(side, 0.0) or 0.0),
        )
        cycle[f"{side}_best_raw_confidence"] = max(
            float(cycle.get(f"{side}_best_raw_confidence", 0.0)),
            float(raw_confidence.get(side, 0.0) or 0.0),
        )

    cycle["left_current_visible"] = bool(current_visible.get("left", False))
    cycle["right_current_visible"] = bool(current_visible.get("right", False))

    for side in ("left", "right"):
        if side not in active_sides or not bool(current_visible.get(side, False)):
            continue
        seen_key = f"{side}_seen_this_car"
        if not bool(cycle.get(seen_key, False)):
            cycle[seen_key] = True
            logger.info(
                "Valid %s-side detection observed for inspection cycle %s at frame %s",
                side,
                cycle.get("inspection_cycle_id", 0),
                frame_number,
            )

    if any(bool(expected_zone_content.get(side, False)) for side in active_sides):
        cycle["expected_zone_checked"] = True
        cycle["checked_frames"] = int(cycle.get("checked_frames", 0)) + 1

    if cycle_success_met(cycle, require_both):
        if str(cycle.get("final_result_for_current_car", FINAL_MISSING)) != FINAL_OK:
            detected_side = "both sides" if require_both else ("left side" if cycle.get("left_seen_this_car") else "right side")
            set_cycle_result(
                cycle,
                FINAL_OK,
                f"PLA CARD OK latched: valid {detected_side} detection at frame {frame_number}.",
                frame_number,
            )
            cycle["last_ok_latch_frame"] = frame_number
            logger.info(
                "PLA CARD OK latched: cycle=%s left_seen_this_car=%s right_seen_this_car=%s mode=%s frame=%s",
                cycle.get("inspection_cycle_id", 0),
                bool(cycle.get("left_seen_this_car", False)),
                bool(cycle.get("right_seen_this_car", False)),
                inspection_mode_label(require_both),
                frame_number,
            )

    if content_active:
        cycle["empty_frames"] = 0
    else:
        cycle["empty_frames"] = int(cycle.get("empty_frames", 0)) + 1
        if int(cycle["empty_frames"]) >= int(config.get("cycle_end_empty_frames", 25)):
            finish_inspection_cycle(cycle, frame_number, require_both, config)

    return str(cycle.get("final_result_for_current_car", FINAL_MISSING))


class ModelManager:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.names: dict[int, str] = {}
        self.device: int | str = "cpu"
        self.device_label = "CPU"
        self.half = False
        self.error = ""
        self.lock = threading.Lock()
        self.load_model(config.get("MODEL_PATH", ""))

    def load_model(self, selected_path: str = "") -> None:
        self.error = ""
        self.model = None
        self.model_path = find_model_path(selected_path)
        if self.model_path is None:
            self.error = "No .pt model found in this project."
            return

        try:
            with quiet_console():
                self.model = YOLO(str(self.model_path))

            use_gpu = bool(self.config.get("USE_GPU", True))
            if use_gpu and torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                self.device = 0
                self.half = bool(self.config.get("USE_FP16", True))
                gpu_name = torch.cuda.get_device_name(0)
                self.device_label = f"CUDA GPU: {gpu_name}"
                with quiet_console():
                    self.model.to("cuda")
            else:
                self.device = "cpu"
                self.half = False
                self.device_label = "CPU: CUDA not available"

            model_names = getattr(self.model, "names", {}) or {}
            if isinstance(model_names, list):
                self.names = {idx: str(name) for idx, name in enumerate(model_names)}
            else:
                self.names = {int(idx): str(name) for idx, name in model_names.items()}

            yaml_names = names_from_data_yaml()
            if yaml_names:
                self.names.update(yaml_names)

            if not self.names:
                self.names = {idx: f"class_{idx}" for idx in range(10)}
            logger.info("Model class names used for status: %s", self.names)

            warmup_size = int(self.config.get("IMG_SIZE", 416))
            dummy = np.zeros((warmup_size, warmup_size, 3), dtype=np.uint8)
            with torch.inference_mode(), quiet_console():
                self.model.predict(
                    dummy,
                    imgsz=warmup_size,
                    conf=0.25,
                    device=self.device,
                    half=self.half,
                    max_det=1,
                    verbose=False,
                )
            logger.info("Model loaded: %s", self.model_path)
            logger.info("Device selected: %s half=%s", self.device_label, self.half)
        except Exception as exc:
            self.error = f"Model load failed: {exc}"
            self.model = None
            logger.exception("Model load failed")

    @property
    def ready(self) -> bool:
        return self.model is not None and not self.error

    def predict_batch(self, frames: list[np.ndarray], imgsz: int, threshold: float) -> list[Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        if not frames:
            return []
        predict_conf = max(0.05, min(0.25, threshold * 0.5))
        with self.lock:
            with torch.inference_mode(), quiet_console():
                return self.model.predict(
                    frames,
                    imgsz=imgsz,
                    conf=predict_conf,
                    device=self.device,
                    half=self.half,
                    max_det=10,
                    verbose=False,
                )


def analyze_result(result: Any, names: dict[int, str], threshold: float) -> FrameAnalysis:
    high: list[Detection] = []
    low: list[Detection] = []
    boxes = getattr(result, "boxes", None)

    if boxes is not None:
        for box in boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                box=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                above_threshold=confidence >= threshold,
            )
            if confidence >= threshold:
                high.append(detection)
            else:
                low.append(detection)

    class_count = len(names)
    if high:
        statuses = [class_status(det.class_name, class_count) for det in high]
        best = max(high, key=lambda det: det.confidence)
        if "MISSING" in statuses:
            status = "MISSING"
        elif "PRESENT" in statuses:
            status = "PRESENT"
        else:
            status = "LOW CONFIDENCE"
        return FrameAnalysis(status, best.class_name, best.confidence, len(high), high, status == "PRESENT")

    if low:
        best_low = max(low, key=lambda det: det.confidence)
        return FrameAnalysis("LOW CONFIDENCE", best_low.class_name, best_low.confidence, 0, [best_low], False)

    return FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)


def analysis_from_detections(detections: list[Detection], names: dict[int, str], threshold: float) -> FrameAnalysis:
    high = [det for det in detections if det.confidence >= threshold]
    low = [det for det in detections if det.confidence < threshold]
    class_count = len(names)
    if high:
        statuses = [class_status(det.class_name, class_count) for det in high]
        best = max(high, key=lambda det: det.confidence)
        if "MISSING" in statuses:
            status = "MISSING"
        elif "PRESENT" in statuses:
            status = "PRESENT"
        else:
            status = "LOW CONFIDENCE"
        return FrameAnalysis(status, best.class_name, best.confidence, len(high), high, status == "PRESENT")
    if low:
        best_low = max(low, key=lambda det: det.confidence)
        return FrameAnalysis("LOW CONFIDENCE", best_low.class_name, best_low.confidence, 0, [best_low], False)
    return FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)


def translate_analysis(analysis: FrameAnalysis, offset_x: int, offset_y: int) -> FrameAnalysis:
    translated: list[Detection] = []
    for det in analysis.detections:
        x1, y1, x2, y2 = det.box
        translated.append(
            Detection(
                det.class_id,
                det.class_name,
                det.confidence,
                (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                det.above_threshold,
            )
        )
    return FrameAnalysis(
        analysis.status,
        analysis.detected_class,
        analysis.highest_confidence,
        analysis.total_detections,
        translated,
        analysis.sticker_present,
    )


def rect_norm_for_side(config: dict[str, Any], side: str, suffix: str, fallback_suffix: str = "roi") -> dict[str, float]:
    roi = config.get(f"{side}_{suffix}") or config.get(f"{side}_{fallback_suffix}") or {}
    if not isinstance(roi, dict):
        roi = {}
    return {
        "x_norm": float(roi.get("x_norm", 0.0)),
        "y_norm": float(roi.get("y_norm", 0.0)),
        "w_norm": float(roi.get("w_norm", 1.0)),
        "h_norm": float(roi.get("h_norm", 1.0)),
    }


def norm_rect_to_pixels(rect: dict[str, float], frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    x = int(round(rect["x_norm"] * width))
    y = int(round(rect["y_norm"] * height))
    w = int(round(rect["w_norm"] * width))
    h = int(round(rect["h_norm"] * height))
    x = max(0, min(x, width - 2))
    y = max(0, min(y, height - 2))
    w = max(2, min(w, width - x))
    h = max(2, min(h, height - y))
    return x, y, w, h


def roi_norm_for_side(config: dict[str, Any], side: str) -> dict[str, float]:
    if bool(config.get("use_search_roi", True)):
        return rect_norm_for_side(config, side, "search_roi")
    return rect_norm_for_side(config, side, "roi")


def expected_zone_norm_for_side(config: dict[str, Any], side: str) -> dict[str, float]:
    return rect_norm_for_side(config, side, "expected_zone")


def roi_to_pixels(config: dict[str, Any], side: str, frame: np.ndarray) -> tuple[int, int, int, int]:
    return norm_rect_to_pixels(roi_norm_for_side(config, side), frame)


def expected_zone_to_pixels(config: dict[str, Any], side: str, frame: np.ndarray) -> tuple[int, int, int, int]:
    return norm_rect_to_pixels(expected_zone_norm_for_side(config, side), frame)


def detection_center(det: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = det.box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def detection_touches_boundary(analysis: FrameAnalysis, frame: np.ndarray, margin_px: int = 3) -> bool:
    height, width = frame.shape[:2]
    for det in analysis.detections:
        x1, y1, x2, y2 = det.box
        if x1 <= margin_px or y1 <= margin_px or x2 >= width - margin_px or y2 >= height - margin_px:
            return True
    return False


def analysis_has_present_in_zone(
    analysis: FrameAnalysis,
    zone_rect: tuple[int, int, int, int],
    class_count: int,
) -> bool:
    x, y, w, h = zone_rect
    x2 = x + w
    y2 = y + h
    for det in analysis.detections:
        if not det.above_threshold:
            continue
        if class_status(det.class_name, class_count) != "PRESENT":
            continue
        cx, cy = detection_center(det)
        if x <= cx <= x2 and y <= cy <= y2:
            return True
    return False


def valid_present_detections(analysis: FrameAnalysis, class_count: int) -> list[Detection]:
    return [
        det
        for det in analysis.detections
        if det.above_threshold and class_status(det.class_name, class_count) == "PRESENT"
    ]


def best_present_detection(analysis: FrameAnalysis, class_count: int) -> Detection | None:
    detections = valid_present_detections(analysis, class_count)
    if not detections:
        return None
    return max(detections, key=lambda det: det.confidence)


def smooth_box(
    previous_box: tuple[int, int, int, int] | None,
    current_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if previous_box is None:
        return current_box
    return tuple(
        int(round(0.7 * old + 0.3 * new))
        for old, new in zip(previous_box, current_box)
    )


def new_overlay_memory() -> dict[str, Any]:
    return {
        "last_valid_box": None,
        "last_valid_confidence": 0.0,
        "last_valid_class": "No detection",
        "last_valid_class_id": 0,
        "last_valid_frame": 0,
        "last_valid_timestamp": 0.0,
        "last_valid_detection_source": "",
        "display_box": None,
        "last_reason": "no raw detection",
        "last_logged_reason": "",
    }


def rejection_reason(analysis: FrameAnalysis, best_det: Detection | None, roi_active: bool, expected_zone_active: bool) -> str:
    if not roi_active:
        return "ROI inactive"
    if not analysis.detections:
        return "no raw detection"
    if best_det is None and analysis.highest_confidence > 0.0 and analysis.highest_confidence < 1.0:
        if analysis.status == "LOW CONFIDENCE" or all(not det.above_threshold for det in analysis.detections):
            return "below confidence threshold"
        if not expected_zone_active:
            return "outside expected zone"
        return "raw detection class rejected"
    if best_det is None:
        return "no accepted detection"
    if not expected_zone_active:
        return "expected zone inactive"
    return "accepted detection"


def held_overlay_detection(
    side: str,
    memory: dict[str, Any],
    current_frame: int,
    cycle_active: bool,
    config: dict[str, Any],
) -> Detection | None:
    if not bool(config.get("ENABLE_BOX_HOLD", True)):
        memory["last_reason"] = "box hold disabled"
        return None
    last_box = memory.get("last_valid_box")
    last_frame = int(memory.get("last_valid_frame", 0) or 0)
    last_time = float(memory.get("last_valid_timestamp", 0.0) or 0.0)
    if last_box is None:
        memory["last_reason"] = "no raw detection"
        memory["display_box"] = None
        return None
    if not cycle_active:
        memory["last_reason"] = "cycle inactive"
        memory["display_box"] = None
        return None
    hold_frames = int(config.get("BOX_HOLD_FRAMES", 10))
    hold_ms = int(config.get("BOX_HOLD_MS", 400))
    frame_age = max(0, current_frame - last_frame)
    age_ms = max(0.0, (time.perf_counter() - last_time) * 1000.0)
    if frame_age <= hold_frames and age_ms <= hold_ms:
        memory["last_reason"] = "using held previous box"
        return Detection(
            int(memory.get("last_valid_class_id", 0) or 0),
            str(memory.get("last_valid_class", "held detection")),
            float(memory.get("last_valid_confidence", 0.0) or 0.0),
            tuple(int(v) for v in last_box),
            True,
        )
    memory["last_reason"] = "hold expired"
    memory["display_box"] = None
    logger.info(
        "%s overlay box not drawn: hold expired frame_age=%s hold_frames=%s age_ms=%.1f hold_ms=%s",
        side,
        frame_age,
        hold_frames,
        age_ms,
        hold_ms,
    )
    return None


def update_overlay_memory(
    side: str,
    memory: dict[str, Any],
    accepted_det: Detection | None,
    analysis: FrameAnalysis,
    frame_number: int,
    fresh_inference: bool,
    cycle_active: bool,
    roi_active: bool,
    expected_zone_active: bool,
    config: dict[str, Any],
) -> tuple[Detection | None, str]:
    if accepted_det is not None:
        display_box = accepted_det.box
        if bool(config.get("SMOOTH_BOXES", True)):
            display_box = smooth_box(memory.get("display_box"), accepted_det.box)
        memory.update(
            {
                "last_valid_box": display_box,
                "last_valid_confidence": accepted_det.confidence,
                "last_valid_class": accepted_det.class_name,
                "last_valid_class_id": accepted_det.class_id,
                "last_valid_frame": frame_number,
                "last_valid_timestamp": time.perf_counter(),
                "last_valid_detection_source": "fresh_yolo",
                "display_box": display_box,
                "last_reason": "fresh accepted detection",
            }
        )
        return Detection(
            accepted_det.class_id,
            accepted_det.class_name,
            accepted_det.confidence,
            tuple(int(v) for v in display_box),
            True,
        ), "fresh accepted detection"

    if not fresh_inference:
        held = held_overlay_detection(side, memory, frame_number, cycle_active, config)
        if held is None and memory.get("last_valid_box") is None:
            memory["last_reason"] = "no fresh inference"
        return held, str(memory.get("last_reason", "no fresh inference"))

    reason = rejection_reason(analysis, accepted_det, roi_active, expected_zone_active)
    memory["last_reason"] = reason
    if reason != memory.get("last_logged_reason"):
        logger.info(
            "%s overlay box not drawn: %s raw_status=%s raw_conf=%.3f roi_active=%s expected_zone_active=%s cycle_active=%s",
            side,
            reason,
            analysis.status,
            analysis.highest_confidence,
            roi_active,
            expected_zone_active,
            cycle_active,
        )
        memory["last_logged_reason"] = reason
    return None, reason


def filter_analysis_to_roi(
    analysis: FrameAnalysis,
    roi_rect: tuple[int, int, int, int],
    names: dict[int, str],
    threshold: float,
) -> FrameAnalysis:
    x, y, w, h = roi_rect
    x2 = x + w
    y2 = y + h
    inside: list[Detection] = []
    ignored = 0
    for det in analysis.detections:
        cx, cy = detection_center(det)
        if x <= cx <= x2 and y <= cy <= y2:
            inside.append(det)
        else:
            ignored += 1
    if ignored:
        logger.info("Ignored %s detection(s) outside ROI", ignored)
    return analysis_from_detections(inside, names, threshold)


def roi_has_content(frame: np.ndarray, roi_rect: tuple[int, int, int, int], config: dict[str, Any]) -> bool:
    if not bool(config.get("empty_roi_detection_enabled", True)):
        return True
    x, y, w, h = roi_rect
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    texture = float(gray.std())
    return texture >= float(config.get("empty_roi_texture_threshold", 10.0))


def prepare_inference_frame(
    side: str,
    frame: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    roi_rect = roi_to_pixels(config, side, frame)
    if not bool(config.get("roi_enabled", True)) or str(config.get("roi_mode", "crop")).lower() != "crop":
        return frame, (0, 0), roi_rect
    x, y, w, h = roi_rect
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        logger.warning("Invalid ROI crop for %s; falling back to full frame", side)
        return frame, (0, 0), roi_rect
    return crop, (x, y), roi_rect


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    y = max(y, th + 8)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_analysis(
    frame: np.ndarray,
    analysis: FrameAnalysis,
    threshold: float,
    side_label: str,
    side_state: str,
    final_decision: str,
    fps: float,
    roi_rect: tuple[int, int, int, int] | None = None,
    show_roi_box: bool = True,
    expected_zone_rect: tuple[int, int, int, int] | None = None,
    show_expected_zone_box: bool = False,
    overlay_detections: list[Detection] | None = None,
    debug_lines: list[str] | None = None,
) -> np.ndarray:
    output = frame
    if show_roi_box and roi_rect is not None:
        x, y, w, h = roi_rect
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 180, 0), 2)
        draw_label(output, "Search ROI", x, max(22, y), (255, 180, 0))
    if show_expected_zone_box and expected_zone_rect is not None:
        x, y, w, h = expected_zone_rect
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 160, 255), 2)
        draw_label(output, "Expected Zone", x, max(22, y), (0, 160, 255))

    detections_to_draw = overlay_detections if overlay_detections is not None else analysis.detections
    for det in detections_to_draw:
        color = color_for_class(det.class_id) if det.above_threshold else status_color("LOW CONFIDENCE")
        x1, y1, x2, y2 = det.box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        draw_label(output, f"{det.class_name} {det.confidence:.2f}", max(0, x1), max(22, y1), color)

    if side_state in ("PASS", "STICKER_PRESENT", FINAL_OK):
        side_text = "PLA Card Present"
    elif side_state in ("FAIL", "CONFIRMED_FAIL"):
        side_text = "NG"
    elif side_state == "WAITING":
        side_text = "Waiting"
    else:
        side_text = "Missing"
    overlay_text = f"{side_label.upper()} | {side_text}"
    decision_color = status_color(final_decision)
    if final_decision == FINAL_WAITING:
        decision_color = (180, 180, 180)

    cv2.rectangle(output, (12, 12), (min(output.shape[1] - 12, 610), 92), (26, 27, 30), -1)
    cv2.putText(output, overlay_text, (26, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color(side_state), 2, cv2.LINE_AA)
    cv2.putText(output, f"{final_decision_label(final_decision)} | FPS {fps:.1f}", (26, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.72, decision_color, 2, cv2.LINE_AA)
    if debug_lines:
        line_height = 22
        width = min(output.shape[1] - 12, 760)
        height = 18 + line_height * len(debug_lines)
        y0 = 104
        cv2.rectangle(output, (12, y0), (width, y0 + height), (20, 20, 22), -1)
        for idx, line in enumerate(debug_lines):
            cv2.putText(
                output,
                line[:95],
                (24, y0 + 24 + idx * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 235, 240),
                1,
                cv2.LINE_AA,
            )
    return output


def resize_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1))
    scale = min(scale, 1.0) if w <= max_width and h <= max_height else scale
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def cv_to_qimage(frame: np.ndarray) -> QImage:
    h, w, channels = frame.shape
    bytes_per_line = channels * w
    if hasattr(QImage.Format, "Format_BGR888"):
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def video_diagnostics(path: str) -> tuple[dict[str, Any], np.ndarray | None]:
    resolved = str(Path(path).expanduser().resolve())
    file_path = Path(resolved)
    info: dict[str, Any] = {
        "path": resolved,
        "exists": file_path.exists(),
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "opened": False,
        "frame_count": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "first_frame_read": False,
        "error": "",
    }
    if not info["exists"]:
        info["error"] = "VIDEO FILE NOT FOUND"
        return info, None

    cap = cv2.VideoCapture(resolved)
    try:
        info["opened"] = bool(cap.isOpened())
        if not info["opened"]:
            info["error"] = "VIDEO OPEN FAILED"
            return info, None
        info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        ok, frame = cap.read()
        info["first_frame_read"] = bool(ok and frame is not None and frame.size > 0)
        if not info["first_frame_read"]:
            info["error"] = "NO FRAME READ"
            return info, None
        return info, frame
    except Exception as exc:
        info["error"] = str(exc)
        logger.exception("Video diagnostics failed for %s", resolved)
        return info, None
    finally:
        cap.release()


def write_debug_first_frame(side: str, frame: np.ndarray) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    debug_path = LOG_PATH.parent / f"debug_{side}_first_frame.jpg"
    ok = cv2.imwrite(str(debug_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        logger.info("Saved first %s debug frame: %s", side, debug_path)
    else:
        logger.warning("Failed to save first %s debug frame: %s", side, debug_path)


class LatestFrameCapture:
    def __init__(self, side: str, path: str, loop_video: bool = False) -> None:
        self.side = side
        self.path = path
        self.loop_video = loop_video
        self.cap = cv2.VideoCapture(path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_value = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps_value if fps_value and 0 < fps_value <= 240 else 30.0
        self.frame_number = 0
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ended = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.last_read_error = ""

    @property
    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def start(self) -> None:
        self.thread.start()
        logger.info(
            "Video source opened: side=%s opened=%s path=%s total_frames=%s source_fps=%.2f loop=%s",
            self.side,
            self.cap.isOpened(),
            self.path,
            self.total_frames,
            self.source_fps,
            self.loop_video,
        )

    def _reader(self) -> None:
        frame_interval = 1.0 / self.source_fps
        logger.info("Playback loop started: side=%s thread_alive=%s", self.side, self.thread.is_alive())
        last_progress_log = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                loop_start = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok:
                    current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    self.last_read_error = f"cap.read() returned false at frame {current_pos}"
                    logger.info(
                        "Read failure: side=%s frame=%s total=%s loop=%s",
                        self.side,
                        current_pos,
                        self.total_frames,
                        self.loop_video,
                    )
                    if self.loop_video and self.total_frames > 0 and not self.stop_event.is_set():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.ended = True
                    break
                with self.lock:
                    self.frame = frame
                    self.frame_number += 1
                now = time.perf_counter()
                if now - last_progress_log >= 3.0:
                    logger.info(
                        "Playback progress: side=%s frame_number=%s worker_thread_alive=%s",
                        self.side,
                        self.frame_number,
                        self.thread.is_alive(),
                    )
                    last_progress_log = now
                sleep_time = frame_interval - (time.perf_counter() - loop_start)
                if sleep_time > 0:
                    time.sleep(min(sleep_time, frame_interval))
        except Exception:
            logger.exception("Playback loop crashed: side=%s", self.side)
        finally:
            try:
                self.cap.release()
            except Exception:
                logger.exception("Failed to release video capture: side=%s", self.side)
            with self.lock:
                self.frame = None
            logger.info(
                "Playback loop stopped: side=%s frame_number=%s ended=%s stop_requested=%s last_read_error=%s",
                self.side,
                self.frame_number,
                self.ended,
                self.stop_event.is_set(),
                self.last_read_error,
            )

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self.lock:
            return self.frame_number, self.frame

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.5)
        if self.cap.isOpened():
            self.cap.release()
        with self.lock:
            self.frame = None
        logger.info("Playback resources cleaned: side=%s thread_alive=%s", self.side, self.thread.is_alive())


class RTSPCameraWorker:
    def __init__(
        self,
        side: str,
        source: str,
        name: str,
        use_ffmpeg: bool = True,
        reconnect_seconds: float = 2.0,
        max_no_frame_seconds: float = 2.0,
    ) -> None:
        self.side = side
        self.path = source
        self.name = name or side.upper()
        self.masked_source = mask_rtsp_url(source)
        self.use_ffmpeg = use_ffmpeg
        self.reconnect_seconds = max(0.5, float(reconnect_seconds))
        self.max_no_frame_seconds = max(0.5, float(max_no_frame_seconds))
        self.total_frames = 0
        self.frame_number = 0
        self.frame: np.ndarray | None = None
        self.connected = False
        self.ended = False
        self.fps = 0.0
        self.last_frame_time = 0.0
        self.error = "Not connected"
        self.last_status = ""
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.cap: cv2.VideoCapture | None = None
        self.thread = threading.Thread(target=self._reader, name=f"{side}-rtsp-camera", daemon=True)

    @property
    def is_opened(self) -> bool:
        return True

    def start(self) -> None:
        self.thread.start()
        live_logger.info("%s camera worker started: %s", self.side, self.masked_source)

    def _set_status(self, connected: bool, error: str = "") -> None:
        with self.lock:
            previous = self.connected
            self.connected = connected
            self.error = error
            status = "connected" if connected else f"disconnected: {error or 'no frame'}"
            self.last_status = status
        if connected and not previous:
            live_logger.info("%s camera connected: %s", self.side, self.masked_source)
        elif not connected and previous:
            live_logger.warning("%s camera disconnected: %s", self.side, error or "no frame")

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_FFMPEG if self.use_ffmpeg else 0
        params: list[int] = []
        timeout_ms = int(max(500, self.max_no_frame_seconds * 1000))
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC), timeout_ms])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([int(cv2.CAP_PROP_READ_TIMEOUT_MSEC), timeout_ms])
        if self.use_ffmpeg:
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                f"rtsp_transport;tcp|stimeout;{timeout_ms * 1000}|timeout;{timeout_ms * 1000}",
            )
            try:
                cap = cv2.VideoCapture(self.path, backend, params) if params else cv2.VideoCapture(self.path, backend)
            except Exception:
                cap = cv2.VideoCapture(self.path, backend)
        else:
            try:
                cap = cv2.VideoCapture(self.path, backend, params) if params else cv2.VideoCapture(self.path)
            except Exception:
                cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _release_capture(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                live_logger.exception("%s camera release failed", self.side)
        self.cap = None

    def _reader(self) -> None:
        live_logger.info("%s camera read loop starting", self.side)
        fps_count = 0
        fps_clock = time.perf_counter()
        while not self.stop_event.is_set():
            try:
                live_logger.info("%s camera reconnect/open attempt: %s", self.side, self.masked_source)
                self.cap = self._open_capture()
                if self.stop_event.is_set():
                    break
                if self.cap is None or not self.cap.isOpened():
                    self._set_status(False, "RTSP open failed")
                    self._release_capture()
                    time.sleep(self.reconnect_seconds)
                    continue

                self._set_status(True, "")
                last_frame_clock = time.perf_counter()
                fps_count = 0
                fps_clock = time.perf_counter()

                while not self.stop_event.is_set() and self.cap is not None:
                    ok, frame = self.cap.read()
                    now = time.perf_counter()
                    if ok and frame is not None and frame.size > 0:
                        with self.lock:
                            self.frame = frame
                            self.frame_number += 1
                            self.last_frame_time = time.time()
                        last_frame_clock = now
                        fps_count += 1
                        if now - fps_clock >= 1.0:
                            with self.lock:
                                self.fps = fps_count / max(now - fps_clock, 0.001)
                            fps_count = 0
                            fps_clock = now
                        if not self.connected:
                            self._set_status(True, "")
                        continue

                    if now - last_frame_clock >= self.max_no_frame_seconds:
                        self._set_status(False, "No frames received")
                        break
                    time.sleep(0.01)
            except Exception as exc:
                self._set_status(False, str(exc))
                live_logger.exception("%s camera read loop error", self.side)
            finally:
                self._release_capture()

            if not self.stop_event.is_set():
                live_logger.info("%s camera retrying in %.1f seconds", self.side, self.reconnect_seconds)
                time.sleep(self.reconnect_seconds)

        self._release_capture()
        self._set_status(False, "Stopped")
        live_logger.info("%s camera worker stopped at frame %s", self.side, self.frame_number)

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self.lock:
            return self.frame_number, self.frame

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "connected": self.connected,
                "fps": self.fps,
                "last_frame_time": self.last_frame_time,
                "error": self.error,
                "status": self.last_status,
            }

    def stop(self) -> None:
        self.stop_event.set()
        self._release_capture()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        with self.lock:
            self.frame = None
        live_logger.info("%s camera resources cleaned thread_alive=%s", self.side, self.thread.is_alive())


class DualVideoWorker(QThread):
    frame_ready = Signal(str, QImage, dict)
    status_ready = Signal(str, dict)
    finished_ready = Signal(dict)

    def __init__(
        self,
        video_paths: dict[str, str],
        model_manager: ModelManager,
        confidence: float,
        imgsz: int,
        frame_skip: int,
        display_width: int,
        display_height: int,
        require_both: bool,
        smoothing_window: int,
        min_present_frames: int,
        config: dict[str, Any],
        source_mode: str = "video_test",
        camera_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.video_paths = {side: path for side, path in video_paths.items() if path}
        self.source_mode = source_mode
        self.camera_options = dict(camera_options or {})
        self.model_manager = model_manager
        self.confidence = confidence
        self.imgsz = imgsz
        self.frame_skip = max(1, frame_skip)
        self.display_width = display_width
        self.display_height = display_height
        self.require_both = require_both
        self.smoothing_window = max(1, smoothing_window)
        self.min_present_frames = max(1, min_present_frames)
        self.config = dict(config)
        self._stop_event = threading.Event()
        logger.info(
            "Confidence threshold used: %.3f",
            self.confidence,
        )
        live_logger.info("Detection worker source mode: %s", self.source_mode)
        logger.info(
            "ROI mode=%s enabled=%s source=%s left_search=%s right_search=%s left_expected=%s right_expected=%s",
            self.config.get("roi_mode", "crop"),
            self.config.get("roi_enabled", True),
            self.config.get("roi_source", "config"),
            self.config.get("left_search_roi", self.config.get("left_roi")),
            self.config.get("right_search_roi", self.config.get("right_roi")),
            self.config.get("left_expected_zone", self.config.get("left_roi")),
            self.config.get("right_expected_zone", self.config.get("right_roi")),
        )

    def update_side_state(
        self,
        side: str,
        sticker_present: bool,
        best_present_conf: float,
        detected_class: str,
        box_drawn: bool,
        present_vote: bool,
        roi_content: bool,
        expected_zone_content: bool,
        frame_number: int,
        state_memory: dict[str, dict[str, Any]],
    ) -> str:
        memory = state_memory[side]
        previous = str(memory["state"])
        if not bool(self.config.get("inspection_state_enabled", True)):
            state = "STICKER_PRESENT" if sticker_present else "MISSING"
            memory["state"] = state
            return state

        threshold_crossed_before = float(memory.get("best_conf_this_car", 0.0)) >= self.confidence
        inspection_content_active = bool(roi_content or expected_zone_content or sticker_present or present_vote)

        if inspection_content_active and bool(memory.get("ready_for_new_cycle", False)):
            logger.info(
                "%s new inspection window started; resetting side memory from previous car",
                side,
            )
            memory["sticker_seen_this_car"] = False
            memory["best_conf_this_car"] = 0.0
            memory["last_detected_class"] = "No detection"
            memory["seen_hold_logged"] = False
            memory["confirmed_missing_counter"] = 0
            memory["missing_count"] = 0
            memory["pass_hold"] = 0
            memory["ready_for_new_cycle"] = False
            threshold_crossed_before = False

        if sticker_present:
            memory["state"] = "STICKER_PRESENT"
            memory["pass_hold"] = int(self.config.get("pass_hold_frames", 15))
            memory["missing_count"] = 0
            memory["no_car_count"] = 0
            memory["current_car_active"] = True
            memory["sticker_seen_this_car"] = True
            memory["seen_hold_logged"] = False
            memory["expected_zone_checked"] = True
            memory["confirmed_missing_counter"] = 0
            memory["last_sticker_seen_frame"] = frame_number
            memory["best_conf_this_car"] = max(float(memory.get("best_conf_this_car", 0.0)), best_present_conf)
            memory["last_detected_class"] = detected_class
            logger.info(
                "%s PLA card detection latched from drawn box at frame %s: class=%s conf=%.3f threshold=%.3f",
                side,
                frame_number,
                detected_class,
                best_present_conf,
                self.confidence,
            )
        elif not inspection_content_active:
            memory["no_car_count"] = int(memory["no_car_count"]) + 1
            memory["missing_count"] = 0
            memory["confirmed_missing_counter"] = 0
            memory["expected_zone_checked"] = False
            side_was_seen = bool(memory.get("sticker_seen_this_car", False)) or threshold_crossed_before
            if int(memory["no_car_count"]) >= int(self.config.get("no_car_reset_frames", 30)):
                memory["current_car_active"] = False
                memory["seen_hold_logged"] = False
                memory["ready_for_new_cycle"] = True
                memory["state"] = "STICKER_PRESENT" if side_was_seen else ("MISSING" if bool(self.config.get("show_missing_instead_of_fail", True)) else "WAITING")
            elif int(memory["pass_hold"]) > 0 or side_was_seen:
                memory["pass_hold"] = max(0, int(memory["pass_hold"]) - 1)
                memory["state"] = "STICKER_PRESENT"
            else:
                memory["state"] = "WAITING" if bool(memory.get("current_car_active", False)) else "MISSING"
        else:
            memory["no_car_count"] = 0
            memory["current_car_active"] = True
            if int(memory["pass_hold"]) > 0:
                memory["pass_hold"] = int(memory["pass_hold"]) - 1
                memory["state"] = "STICKER_PRESENT"
                if not sticker_present:
                    logger.info("%s sticker temporarily not drawn after frame %s; holding present briefly", side, memory.get("last_sticker_seen_frame", 0))
            elif bool(memory.get("sticker_seen_this_car", False)) or threshold_crossed_before:
                memory["state"] = "STICKER_PRESENT"
                memory["confirmed_missing_counter"] = 0
                if not bool(memory.get("seen_hold_logged", False)):
                    logger.info("%s sticker already seen for current car; holding present instead of MISSING", side)
                    memory["seen_hold_logged"] = True
            else:
                if expected_zone_content:
                    memory["expected_zone_checked"] = True
                memory["missing_count"] = int(memory["missing_count"]) + 1
                memory["state"] = "WAITING"

        state = str(memory["state"])
        if state != previous:
            logger.info("%s state changed: %s -> %s", side, previous, state)
            if state == "MISSING":
                logger.info("%s changed to MISSING only after inspection window ended with no valid drawn detection", side)
        return state

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        captures: dict[str, Any] = {}
        processed_numbers = {"left": 0, "right": 0}
        last_analysis: dict[str, FrameAnalysis | None] = {"left": None, "right": None}
        last_raw_analysis: dict[str, FrameAnalysis | None] = {"left": None, "right": None}
        last_inference_ms = {"left": 0.0, "right": 0.0}
        last_health_emit = {"left": 0.0, "right": 0.0}
        overlay_memory = {"left": new_overlay_memory(), "right": new_overlay_memory()}
        overlay_detections: dict[str, Detection | None] = {"left": None, "right": None}
        overlay_reasons = {"left": "no raw detection", "right": "no raw detection"}
        best_evidence_frames: dict[str, np.ndarray | None] = {"left": None, "right": None}
        best_low_conf_frames: dict[str, np.ndarray | None] = {"left": None, "right": None}
        evidence_cycle_id = 0
        app_fps = 0.0
        app_fps_count = 0
        app_fps_clock = time.perf_counter()
        present_history = {
            "left": deque(maxlen=self.smoothing_window),
            "right": deque(maxlen=self.smoothing_window),
        }
        smoothed = {"left": None, "right": None}
        side_states: dict[str, str | None] = {"left": None, "right": None}
        roi_rects: dict[str, tuple[int, int, int, int] | None] = {"left": None, "right": None}
        expected_zone_rects: dict[str, tuple[int, int, int, int] | None] = {"left": None, "right": None}
        roi_content = {"left": False, "right": False}
        expected_zone_content = {"left": False, "right": False}
        sticker_drawn_present = {"left": False, "right": False}
        best_present_conf = {"left": 0.0, "right": 0.0}
        best_present_class = {"left": "No detection", "right": "No detection"}
        left_sticker_seen = False
        right_sticker_seen = False
        left_best_conf = 0.0
        right_best_conf = 0.0
        state_memory = {
            "left": {
                "state": "WAITING",
                "missing_count": 0,
                "pass_hold": 0,
                "no_car_count": 0,
                "current_car_active": False,
                "sticker_seen_this_car": False,
                "best_conf_this_car": 0.0,
                "last_detected_class": "No detection",
                "ready_for_new_cycle": False,
                "seen_hold_logged": False,
                "expected_zone_checked": False,
                "confirmed_missing_counter": 0,
                "last_sticker_seen_frame": 0,
            },
            "right": {
                "state": "WAITING",
                "missing_count": 0,
                "pass_hold": 0,
                "no_car_count": 0,
                "current_car_active": False,
                "sticker_seen_this_car": False,
                "best_conf_this_car": 0.0,
                "last_detected_class": "No detection",
                "ready_for_new_cycle": False,
                "seen_hold_logged": False,
                "expected_zone_checked": False,
                "confirmed_missing_counter": 0,
                "last_sticker_seen_frame": 0,
            },
        }
        ended: set[str] = set()
        cycle_state = new_cycle_state()
        inference_count = 0
        inference_clock = time.perf_counter()

        for side, path in self.video_paths.items():
            if self.source_mode == "live":
                camera_info = dict(self.camera_options.get("cameras", {}).get(side, {}))
                stream_info = dict(self.camera_options.get("stream", {}))
                source = RTSPCameraWorker(
                    side=side,
                    source=path,
                    name=str(camera_info.get("name", side.upper())),
                    use_ffmpeg=bool(stream_info.get("use_ffmpeg", True)),
                    reconnect_seconds=float(stream_info.get("reconnect_seconds", 2.0)),
                    max_no_frame_seconds=float(stream_info.get("max_no_frame_seconds", 2.0)),
                )
                captures[side] = source
                self.status_ready.emit(
                    side,
                    {
                        "status": "WAITING",
                        "camera_health": True,
                        "camera_status": "CONNECTING",
                        "camera_fps": 0.0,
                        "error": "",
                    },
                )
                source.start()
                continue

            source = LatestFrameCapture(side, path, bool(self.config.get("loop_video", False)))
            if not source.is_opened:
                self.status_ready.emit(side, {"status": "NO VIDEO", "error": "Invalid video file"})
                source.stop()
                continue
            captures[side] = source
            source.start()

        if not captures:
            self.finished_ready.emit({"status": "NO VIDEO"})
            return

        try:
            while captures and not self._stop_event.is_set():
                frames: dict[str, np.ndarray] = {}
                infer_sides: list[str] = []
                infer_frames: list[np.ndarray] = []
                infer_offsets: dict[str, tuple[int, int]] = {}
                fresh_inference = {"left": False, "right": False}
                fresh_best_det: dict[str, Detection | None] = {"left": None, "right": None}

                for side, source in list(captures.items()):
                    if self.source_mode == "live":
                        now = time.perf_counter()
                        if now - last_health_emit[side] >= 1.0:
                            health = source.health()
                            camera_status = "CONNECTED" if health.get("connected") else "DISCONNECTED"
                            self.status_ready.emit(
                                side,
                                {
                                    "status": side_states[side] or "WAITING",
                                    "camera_health": True,
                                    "camera_status": camera_status,
                                    "camera_fps": float(health.get("fps", 0.0) or 0.0),
                                    "error": "" if health.get("connected") else str(health.get("error", "")),
                                },
                            )
                            last_health_emit[side] = now

                    frame_number, frame = source.latest()
                    if self.source_mode != "live" and source.ended and (frame is None or processed_numbers[side] >= frame_number):
                        ended.add(side)
                        source.stop()
                        captures.pop(side, None)
                        if processed_numbers[side] == 0:
                            self.status_ready.emit(side, {"status": "NO FRAME READ", "error": "NO FRAME READ"})
                            logger.warning("NO FRAME READ from %s video: %s", side, source.path)
                        else:
                            self.status_ready.emit(side, {"status": "VIDEO ENDED"})
                        continue
                    if frame is None or frame_number <= processed_numbers[side]:
                        continue

                    frames[side] = frame.copy()
                    processed_numbers[side] = frame_number
                    roi_rects[side] = roi_to_pixels(self.config, side, frame)
                    expected_zone_rects[side] = expected_zone_to_pixels(self.config, side, frame)
                    roi_content[side] = roi_has_content(frame, roi_rects[side], self.config)
                    expected_zone_content[side] = roi_has_content(frame, expected_zone_rects[side], self.config)
                    should_infer = (
                        last_analysis[side] is None
                        or ((frame_number - 1) % self.frame_skip == 0)
                    )
                    if should_infer:
                        inference_frame, offset, roi_rect = prepare_inference_frame(side, frame, self.config)
                        roi_rects[side] = roi_rect
                        infer_sides.append(side)
                        infer_frames.append(inference_frame)
                        infer_offsets[side] = offset

                if infer_frames:
                    infer_start = time.perf_counter()
                    try:
                        results = self.model_manager.predict_batch(infer_frames, self.imgsz, self.confidence)
                        elapsed_ms = (time.perf_counter() - infer_start) * 1000.0
                        per_frame_ms = elapsed_ms / max(1, len(results))
                        for side, result in zip(infer_sides, results):
                            fresh_inference[side] = True
                            last_inference_ms[side] = per_frame_ms
                            raw_analysis = analyze_result(
                                result,
                                self.model_manager.names,
                                self.confidence,
                            )
                            last_raw_analysis[side] = raw_analysis
                            analysis = raw_analysis
                            offset_x, offset_y = infer_offsets.get(side, (0, 0))
                            if bool(self.config.get("roi_enabled", True)) and str(self.config.get("roi_mode", "crop")).lower() == "crop":
                                if detection_touches_boundary(analysis, infer_frames[infer_sides.index(side)]):
                                    logger.warning("%s detection touched search ROI crop boundary; falling back to full-frame inference", side)
                                    full_result = self.model_manager.predict_batch([frames[side]], self.imgsz, self.confidence)[0]
                                    raw_analysis = analyze_result(full_result, self.model_manager.names, self.confidence)
                                    last_raw_analysis[side] = raw_analysis
                                    analysis = raw_analysis
                                    logger.info("%s full-frame fallback used after crop boundary touch", side)
                                else:
                                    analysis = translate_analysis(analysis, offset_x, offset_y)
                            elif bool(self.config.get("roi_enabled", True)):
                                analysis = filter_analysis_to_roi(
                                    analysis,
                                    roi_rects[side] or roi_to_pixels(self.config, side, frames[side]),
                                    self.model_manager.names,
                                    self.confidence,
                                )
                            last_analysis[side] = analysis
                            best_det = best_present_detection(analysis, len(self.model_manager.names))
                            fresh_best_det[side] = best_det
                            sticker_drawn_present[side] = best_det is not None
                            best_present_conf[side] = float(best_det.confidence) if best_det is not None else 0.0
                            best_present_class[side] = best_det.class_name if best_det is not None else analysis.detected_class
                            if best_det is not None:
                                best_evidence_frames[side] = frames[side].copy()
                            elif 0.0 < analysis.highest_confidence < self.confidence and analysis.highest_confidence >= self.confidence * 0.75:
                                best_low_conf_frames[side] = frames[side].copy()
                            present_history[side].append(sticker_drawn_present[side])
                            smoothed[side] = smoothed_present(
                                present_history[side],
                                self.min_present_frames,
                            )
                        inference_count += len(results)
                        if time.perf_counter() - inference_clock >= 3.0:
                            total_elapsed = time.perf_counter() - inference_clock
                            logger.info("Inference FPS: %.1f", inference_count / max(total_elapsed, 0.001))
                            inference_count = 0
                            inference_clock = time.perf_counter()
                    except Exception as exc:
                        for side in infer_sides:
                            self.status_ready.emit(side, {"status": "NO VIDEO", "error": str(exc)})
                        logger.exception("Inference error")
                        break

                for side in frames:
                    previous_side_status = str(state_memory[side].get("state", "WAITING"))
                    side_states[side] = self.update_side_state(
                        side,
                        bool(sticker_drawn_present[side]),
                        float(best_present_conf[side]),
                        str(best_present_class[side]),
                        bool(sticker_drawn_present[side]),
                        bool(smoothed[side]),
                        bool(roi_content[side]),
                        bool(expected_zone_content[side]),
                        processed_numbers[side],
                        state_memory,
                    )
                    status_updated = previous_side_status != str(side_states[side])
                    if bool(self.config.get("DEBUG_MODE", False)):
                        raw = last_raw_analysis[side] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)
                        logger.info(
                            "%s debug frame=%s raw_status=%s raw_confidence=%.3f accepted=%s accepted_class=%s accepted_confidence=%.3f overlay_reason=%s status_updated=%s current_side_status=%s sticker_seen=%s best_conf=%.3f threshold=%.3f roi_active=%s expected_zone_active=%s",
                            side,
                            processed_numbers[side],
                            raw.status,
                            raw.highest_confidence,
                            sticker_drawn_present[side],
                            best_present_class[side],
                            best_present_conf[side],
                            overlay_reasons[side],
                            status_updated,
                            side_states[side],
                            bool(state_memory[side].get("sticker_seen_this_car", False)),
                            float(state_memory[side].get("best_conf_this_car", 0.0)),
                            self.confidence,
                            bool(roi_content[side]),
                            bool(expected_zone_content[side]),
                        )

                if frames:
                    left_sticker_seen = bool(state_memory["left"].get("sticker_seen_this_car", False))
                    right_sticker_seen = bool(state_memory["right"].get("sticker_seen_this_car", False))
                    left_best_conf = float(state_memory["left"].get("best_conf_this_car", 0.0))
                    right_best_conf = float(state_memory["right"].get("best_conf_this_car", 0.0))
                    active_sides = [side for side in ("left", "right") if side in self.video_paths]
                    current_visible = {
                        "left": bool(sticker_drawn_present["left"] or left_sticker_seen or left_best_conf >= self.confidence),
                        "right": bool(sticker_drawn_present["right"] or right_sticker_seen or right_best_conf >= self.confidence),
                    }
                    cycle_frame = max(processed_numbers[side] for side in frames)
                    final_decision = update_inspection_cycle(
                        cycle_state,
                        active_sides,
                        current_visible,
                        roi_content,
                        expected_zone_content,
                        cycle_frame,
                        self.require_both,
                        self.config,
                        best_present_conf,
                        {
                            "left": float((last_raw_analysis["left"] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)).highest_confidence),
                            "right": float((last_raw_analysis["right"] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)).highest_confidence),
                        },
                    )
                    current_cycle_id = int(cycle_state.get("inspection_cycle_id", 0))
                    if current_cycle_id and current_cycle_id != evidence_cycle_id:
                        evidence_cycle_id = current_cycle_id
                        best_evidence_frames = {"left": None, "right": None}
                        best_low_conf_frames = {"left": None, "right": None}
                        for evidence_side in frames:
                            evidence_analysis = last_analysis[evidence_side] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)
                            if fresh_best_det[evidence_side] is not None:
                                best_evidence_frames[evidence_side] = frames[evidence_side].copy()
                            elif (
                                0.0 < evidence_analysis.highest_confidence < self.confidence
                                and evidence_analysis.highest_confidence >= self.confidence * 0.75
                            ):
                                best_low_conf_frames[evidence_side] = frames[evidence_side].copy()
                    completed_log = cycle_state.get("last_completed_cycle_log")
                    completed_id = int(completed_log.get("cycle_id", 0)) if isinstance(completed_log, dict) else 0
                    if completed_id and int(cycle_state.get("last_completed_cycle_saved", 0)) != completed_id:
                        near_threshold = any(
                            self.confidence * 0.75 <= float(completed_log.get(key, 0.0)) < self.confidence
                            for key in ("best_left_raw_confidence", "best_right_raw_confidence")
                        )
                        if completed_log.get("final_decision") == FINAL_NG:
                            save_cycle_evidence(NG_CASE_LOG_DIR, completed_log, frames, best_evidence_frames)
                        if near_threshold:
                            evidence_frames = {}
                            for side in ("left", "right"):
                                evidence_frames[side] = (
                                    best_low_conf_frames.get(side)
                                    if best_low_conf_frames.get(side) is not None
                                    else best_evidence_frames.get(side)
                                )
                            save_cycle_evidence(LOW_CONFIDENCE_LOG_DIR, completed_log, frames, evidence_frames)
                        cycle_state["last_completed_cycle_saved"] = completed_id
                else:
                    final_decision = str(cycle_state.get("final_result_for_current_car", FINAL_WAITING))

                if frames:
                    app_fps_count += 1
                    elapsed = time.perf_counter() - app_fps_clock
                    if elapsed >= 1.0:
                        app_fps = app_fps_count / elapsed
                        app_fps_count = 0
                        app_fps_clock = time.perf_counter()

                for side, frame in frames.items():
                    analysis = last_analysis[side] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)
                    raw_analysis = last_raw_analysis[side] or FrameAnalysis("MISSING", "No detection", 0.0, 0, [], False)
                    side_state = side_states[side] or "WAITING"
                    camera_health = captures[side].health() if self.source_mode == "live" and side in captures else {}
                    display_fps = float(camera_health.get("fps", app_fps) or app_fps)
                    cycle_active = bool(cycle_state.get("current_car_active", False))
                    overlay_det, overlay_reason = update_overlay_memory(
                        side,
                        overlay_memory[side],
                        fresh_best_det[side],
                        analysis,
                        processed_numbers[side],
                        bool(fresh_inference[side]),
                        cycle_active,
                        bool(roi_content[side]),
                        bool(expected_zone_content[side]),
                        self.config,
                    )
                    overlay_detections[side] = overlay_det
                    overlay_reasons[side] = overlay_reason
                    if overlay_det is None and overlay_reason != overlay_memory[side].get("last_logged_reason"):
                        logger.info("%s overlay box not drawn: %s", side, overlay_reason)
                        overlay_memory[side]["last_logged_reason"] = overlay_reason
                    debug_lines = None
                    if bool(self.config.get("DEBUG_MODE", False)):
                        accepted = fresh_best_det[side] is not None
                        debug_lines = [
                            f"Raw YOLO: {raw_analysis.status} conf={raw_analysis.highest_confidence:.3f}",
                            f"Accepted: {accepted} reason={overlay_reason}",
                            f"ROI active={bool(roi_content[side])} expected zone active={bool(expected_zone_content[side])}",
                            f"Cycle active={cycle_active} state={side_state} seen={bool(state_memory[side].get('sticker_seen_this_car', False))}",
                            f"Final={final_decision} inf={last_inference_ms[side]:.1f}ms fps={display_fps:.1f}",
                        ]
                    display_overlay = [overlay_det] if overlay_det is not None else ([] if not bool(self.config.get("DEBUG_MODE", False)) else analysis.detections)

                    drawn = draw_analysis(
                        frame,
                        analysis,
                        self.confidence,
                        "Left" if side == "left" else "Right",
                        side_state,
                        final_decision,
                        app_fps,
                        roi_rects[side],
                        bool(self.config.get("show_roi_box", False)) and bool(self.config.get("roi_enabled", True)),
                        expected_zone_rects[side],
                        bool(self.config.get("show_expected_zone_box", False)) and bool(self.config.get("roi_enabled", True)),
                        display_overlay,
                        debug_lines,
                    )
                    display = resize_for_display(drawn, self.display_width, self.display_height)
                    qimage = cv_to_qimage(display)

                    metadata = {
                        "status": side_state,
                        "raw_status": analysis.status,
                        "frame": processed_numbers[side],
                        "total_frames": getattr(captures[side], "total_frames", 0) if side in captures else 0,
                        "fps": display_fps,
                        "camera_status": "CONNECTED" if camera_health.get("connected", self.source_mode != "live") else "DISCONNECTED",
                        "camera_fps": display_fps,
                        "source_mode": self.source_mode,
                        "inference_ms": last_inference_ms[side],
                        "detected_class": analysis.detected_class,
                        "confidence": analysis.highest_confidence,
                        "box_count": analysis.total_detections,
                        "video_path": self.video_paths.get(side, ""),
                        "final_decision": final_decision,
                    }
                    self.frame_ready.emit(side, qimage, metadata)
                    logger.debug("Display update queued: side=%s frame=%s", side, processed_numbers[side])

                if not frames:
                    time.sleep(0.003)
        except Exception:
            logger.exception("Worker loop crashed")
            for side in captures:
                self.status_ready.emit(side, {"status": "ERROR", "error": "Worker loop crashed"})
        finally:
            for source in captures.values():
                source.stop()
            final_status = "STOPPED" if self._stop_event.is_set() else "ENDED"
            self.finished_ready.emit(
                {
                    "status": final_status,
                    "left_frame": processed_numbers["left"],
                    "right_frame": processed_numbers["right"],
                    "ended": sorted(ended),
                }
            )


class VideoPanel(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("videoPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        self.video_label = QLabel("NO VIDEO")
        self.video_label.setObjectName("videoPreview")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(620, 380)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setScaledContents(False)

        self.status_label = QLabel("Status: NO VIDEO")
        self.status_label.setObjectName("statusText")

        layout.addWidget(title_label)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.status_label)


class InspectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.app_state = AppState.LOADING_MODEL
        self.action_locks: set[str] = set()
        self.control_buttons: dict[str, list[QPushButton]] = {}
        self.is_closing = False
        self.config = load_config()
        self.camera_config = load_camera_config()
        self.model_manager = ModelManager(self.config)
        self.video_paths = {"left": "", "right": ""}
        self.side_status = {"left": "NO VIDEO", "right": "NO VIDEO"}
        self.active_source_mode = "video_test"
        self.worker: DualVideoWorker | None = None
        self.current_final_decision = "WAITING"

        self.setWindowTitle("Sticker Inspection")
        self.resize(1720, 900)
        self.build_ui()
        self.apply_styles()
        self.refresh_static_status()

        if self.model_manager.error:
            self.set_app_state(AppState.ERROR, "model load failed")
            QMessageBox.warning(self, "Model Missing", self.model_manager.error)
        else:
            self.set_app_state(AppState.IDLE, "startup complete")
            if self.camera_config.get("mode") == "live":
                self.start_live_cameras(auto=True)

    def build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(22, 18, 22, 18)
        main.setSpacing(16)

        header = QGridLayout()
        header.setColumnStretch(0, 1)
        header.setColumnStretch(1, 3)
        header.setColumnStretch(2, 1)
        main.addLayout(header)

        self.logo_label = QLabel()
        self.logo_label.setObjectName("logo")
        logo_path = find_logo_path(self.config)
        if logo_path:
            pixmap = QPixmap(str(logo_path))
            self.logo_label.setPixmap(
                pixmap.scaled(260, 92, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            self.config["LOGO_PATH"] = str(logo_path)
        else:
            self.logo_label.setText("VISIONLINE")
        header.addWidget(self.logo_label, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        title = QLabel("PLA CARD INFORMATION INSPECTION")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(title, 0, 1)

        self.header_status = QLabel("MODEL READY")
        self.header_status.setObjectName("headerStatus")
        self.header_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.header_status, 0, 2)

        body = QHBoxLayout()
        body.setSpacing(18)
        main.addLayout(body, 1)

        video_area = QGridLayout()
        video_area.setSpacing(16)
        body.addLayout(video_area, 1)

        self.panels = {
            "left": VideoPanel("Camera: LEFT SIDE"),
            "right": VideoPanel("Camera: RIGHT SIDE"),
        }
        video_area.addWidget(self.panels["left"], 0, 0)
        video_area.addWidget(self.panels["right"], 0, 1)

        side_column = QVBoxLayout()
        side_column.setSpacing(14)
        body.addLayout(side_column)

        status_panel = QFrame()
        status_panel.setObjectName("sidePanel")
        status_panel.setMinimumWidth(330)
        status_panel.setMaximumWidth(360)
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(14, 14, 14, 14)
        status_layout.setSpacing(8)
        side_column.addWidget(status_panel)

        status_layout.addWidget(self.section_label("Inspection Status"))
        self.final_result = QLabel("WAITING")
        self.final_result.setObjectName("finalResult")
        status_layout.addWidget(self.final_result)

        self.status_labels: dict[str, QLabel] = {}
        for key in (
            "Left",
            "Right",
            "Mode",
            "Device",
            "Left Cam",
            "Right Cam",
            "Left FPS",
            "Right FPS",
            "Model",
        ):
            label = QLabel("-")
            label.setWordWrap(True)
            self.status_labels[key] = label
            status_layout.addWidget(self.form_row(key, label))

        controls = QFrame()
        controls.setObjectName("sidePanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(9)
        side_column.addWidget(controls)

        controls_layout.addWidget(self.section_label("Controls"))
        self.add_button(controls_layout, "Select Left Video", lambda: self.select_video("left"), "select_video")
        self.add_button(controls_layout, "Select Right Video", lambda: self.select_video("right"), "select_video")
        self.add_button(controls_layout, "Start", self.start_both, "start_video")
        self.add_button(controls_layout, "Stop", self.stop_both, "stop")
        self.add_button(controls_layout, "Start Live Cameras", self.start_live_cameras, "start_camera")
        self.add_button(controls_layout, "Stop Live Cameras", self.stop_both, "stop")
        self.add_button(controls_layout, "Reset", self.clear_results, "reset")
        self.require_both_checkbox = QCheckBox("Require both stickers")
        self.require_both_checkbox.setChecked(bool(self.config.get("REQUIRE_BOTH_STICKERS", False)))
        self.require_both_checkbox.toggled.connect(self.on_require_both_changed)
        controls_layout.addWidget(self.require_both_checkbox)
        controls_layout.addStretch(1)
        side_column.addStretch(1)

    def section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        return label

    def form_row(self, name: str, widget: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(name)
        label.setObjectName("formLabel")
        layout.addWidget(label, 0)
        layout.addWidget(widget, 1)
        return row

    def add_button(self, layout: QVBoxLayout | QHBoxLayout, text: str, callback: Any, action_key: str) -> None:
        button = QPushButton(text)
        button.clicked.connect(callback)
        self.control_buttons.setdefault(action_key, []).append(button)
        layout.addWidget(button)

    def set_app_state(self, state: AppState, reason: str = "") -> None:
        previous = getattr(self, "app_state", AppState.IDLE)
        if previous != state:
            logger.info("App state transition: %s -> %s reason=%s", previous.value, state.value, reason)
        self.app_state = state
        self.update_control_buttons()

    def begin_action(self, action: str) -> bool:
        if action in self.action_locks:
            logger.info("Ignored repeated %s click; action already in progress", action)
            return False
        self.action_locks.add(action)
        logger.info("Button action started: %s state=%s", action, self.app_state.value)
        self.update_control_buttons()
        return True

    def end_action(self, action: str) -> None:
        self.action_locks.discard(action)
        logger.info("Button action finished: %s state=%s", action, self.app_state.value)
        self.update_control_buttons()

    def worker_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def update_control_buttons(self) -> None:
        if not hasattr(self, "control_buttons"):
            return
        busy = bool(self.action_locks) or self.app_state in (AppState.LOADING_MODEL, AppState.STOPPING)
        running = self.app_state in (AppState.RUNNING_VIDEO, AppState.RUNNING_CAMERA) or self.worker_running()
        enabled_by_action = {
            "select_video": not busy and not running,
            "start_video": not busy and not running and self.model_manager.ready,
            "start_camera": not busy and not running and self.model_manager.ready,
            "stop": not busy and running,
            "reset": not busy and not running,
        }
        if self.app_state == AppState.ERROR:
            enabled_by_action["select_video"] = not busy
            enabled_by_action["reset"] = not busy
        for action, buttons in self.control_buttons.items():
            enabled = enabled_by_action.get(action, not busy)
            for button in buttons:
                button.setEnabled(enabled)
        if hasattr(self, "require_both_checkbox"):
            self.require_both_checkbox.setEnabled(not busy and not running)

    def show_clean_error(self, title: str, message: str) -> None:
        logger.error("%s: %s", title, message)
        self.set_app_state(AppState.ERROR, title)
        QMessageBox.warning(self, title, message)

    def stop_worker_safely(self, reason: str, wait_ms: int = 3000) -> bool:
        worker = self.worker
        if worker is None:
            logger.info("No worker to stop: %s", reason)
            return True
        logger.info("Stopping worker: reason=%s running=%s", reason, worker.isRunning())
        try:
            worker.stop()
            if worker.isRunning() and not worker.wait(wait_ms):
                logger.warning("Worker did not stop within %sms: reason=%s", wait_ms, reason)
                return False
            logger.info("Worker stopped cleanly: reason=%s", reason)
            return True
        except Exception:
            logger.exception("Worker stop failed: reason=%s", reason)
            return False
        finally:
            if not worker.isRunning():
                self.worker = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleanup completed")
                except Exception:
                    logger.exception("CUDA cache cleanup failed")

    def apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f3f4f1;
                color: #1f252b;
                font-family: Bahnschrift, Aptos, "Segoe UI Variable", "Segoe UI", Arial;
                font-size: 14px;
            }
            QLabel#title {
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos Display, "Segoe UI", Arial;
                font-size: 36px;
                font-weight: 800;
                color: #101820;
                padding: 4px 2px 8px 2px;
                letter-spacing: 0;
            }
            QLabel#logo {
                color: #005bab;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos Display, "Segoe UI", Arial;
                font-size: 24px;
                font-weight: 800;
            }
            QLabel#headerStatus {
                color: #005bab;
                font-family: Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 800;
                font-size: 13px;
            }
            QFrame#videoPanel, QFrame#sidePanel {
                background: #ffffff;
                border: 1px solid #cfd4d8;
                border-radius: 0;
            }
            QLabel#panelTitle, QLabel#sectionLabel {
                color: #003f7d;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 800;
                font-size: 16px;
                background: #eef5fb;
                padding: 5px 7px;
                border-left: 4px solid #005bab;
            }
            QLabel#videoPreview {
                background: #0d0f12;
                border: 1px solid #7d9fbd;
                border-radius: 0;
                color: #808890;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos, "Segoe UI", Arial;
                font-size: 20px;
                font-weight: 800;
            }
            QLabel#statusText {
                color: #2d363d;
                font-family: Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 700;
                background: #eef1f2;
                padding: 4px 6px;
            }
            QLabel#formLabel {
                color: #46535d;
                min-width: 78px;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 800;
            }
            QLabel#finalResult {
                background: #8d99ae;
                border-radius: 0;
                padding: 12px;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos Display, "Segoe UI", Arial;
                font-size: 26px;
                font-weight: 900;
                qproperty-alignment: AlignCenter;
            }
            QPushButton {
                background: #005bab;
                color: #ffffff;
                border: 0;
                border-radius: 0;
                min-height: 36px;
                padding: 8px 10px;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 800;
            }
            QPushButton:hover { background: #0a73c9; }
            QPushButton:pressed { background: #003f7d; }
            QCheckBox {
                color: #1f252b;
                font-family: Bahnschrift SemiBold, Bahnschrift, Aptos, "Segoe UI", Arial;
                font-weight: 800;
                padding-top: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #7d9fbd;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #005bab;
                border: 1px solid #005bab;
            }
            """
        )

    def refresh_static_status(self) -> None:
        self.status_labels["Model"].setText(Path(self.model_manager.model_path or "").name or "-")
        self.status_labels["Device"].setText(self.model_manager.device_label)
        self.status_labels["Left"].setText(self.side_status["left"])
        self.status_labels["Right"].setText(self.side_status["right"])
        inspection_mode = "Both stickers required" if bool(self.config.get("REQUIRE_BOTH_STICKERS", False)) else "One sticker enough"
        camera_mode = "Live Camera Mode" if self.active_source_mode == "live" else "Video Test Mode"
        self.status_labels["Mode"].setText(f"{camera_mode} | {inspection_mode}")
        for side, key in (("left", "Left Cam"), ("right", "Right Cam")):
            if self.status_labels[key].text() in ("", "-"):
                camera_info = self.camera_config.get("cameras", {}).get(side, {})
                self.status_labels[key].setText("ENABLED" if camera_info.get("enabled") else "DISABLED")
        self.header_status.setText("GPU READY" if "CUDA" in self.model_manager.device_label else "CPU MODE")
        self.update_final_result()
        self.update_control_buttons()

    def on_require_both_changed(self, checked: bool) -> None:
        if self.app_state in (AppState.RUNNING_VIDEO, AppState.RUNNING_CAMERA, AppState.STOPPING):
            logger.info("Ignored require-both toggle while state=%s", self.app_state.value)
            self.require_both_checkbox.setChecked(bool(self.config.get("REQUIRE_BOTH_STICKERS", False)))
            return
        self.config["REQUIRE_BOTH_STICKERS"] = bool(checked)
        self.current_final_decision = "WAITING"
        self.refresh_static_status()

    def current_display_width(self) -> int:
        configured = int(self.config.get("MAX_DISPLAY_WIDTH", 960))
        widths = [self.panels[side].video_label.width() for side in ("left", "right")]
        visible_widths = [width for width in widths if width > 100]
        return min(configured, max(visible_widths) if visible_widths else configured)

    def current_display_height(self) -> int:
        configured = int(self.config.get("MAX_DISPLAY_HEIGHT", 540))
        heights = [self.panels[side].video_label.height() for side in ("left", "right")]
        visible_heights = [height for height in heights if height > 100]
        return min(configured, max(visible_heights) if visible_heights else configured)

    def set_panel_frame(self, side: str, frame: np.ndarray) -> bool:
        try:
            display = resize_for_display(frame, self.current_display_width(), self.current_display_height())
            image = cv_to_qimage(display)
            panel = self.panels[side]
            pixmap = QPixmap.fromImage(image)
            label_size = panel.video_label.size()
            if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
                pixmap = pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            panel.video_label.setPixmap(pixmap)
            return True
        except Exception:
            logger.exception("DISPLAY UPDATE FAILED for %s", side)
            self.side_status[side] = "DISPLAY UPDATE FAILED"
            self.panels[side].status_label.setText("Status: DISPLAY UPDATE FAILED")
            self.status_labels["Left" if side == "left" else "Right"].setText("DISPLAY UPDATE FAILED")
            return False

    def diagnose_and_preview_video(self, side: str) -> bool:
        path = self.video_paths.get(side, "")
        if not path:
            self.side_status[side] = "NO VIDEO"
            self.panels[side].status_label.setText("Status: NO VIDEO")
            return False

        info, frame = video_diagnostics(path)
        logger.info("Video diagnostics %s: %s", side, info)
        prefix = "Left" if side == "left" else "Right"

        if not info.get("opened"):
            self.side_status[side] = "VIDEO OPEN FAILED"
            self.panels[side].video_label.setText("VIDEO OPEN FAILED")
            self.panels[side].status_label.setText("Status: VIDEO OPEN FAILED")
            self.status_labels[prefix].setText("VIDEO OPEN FAILED")
            return False

        if frame is None or not info.get("first_frame_read"):
            self.side_status[side] = "NO FRAME READ"
            self.panels[side].video_label.setText("NO FRAME READ")
            self.panels[side].status_label.setText("Status: NO FRAME READ")
            self.status_labels[prefix].setText("NO FRAME READ")
            return False

        write_debug_first_frame(side, frame)
        preview = frame.copy()
        if bool(self.config.get("roi_enabled", True)) and bool(self.config.get("show_roi_box", False)):
            roi_rect = roi_to_pixels(self.config, side, preview)
            x, y, w, h = roi_rect
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 180, 0), 2)
            draw_label(preview, "Search ROI", x, max(22, y), (255, 180, 0))
        if bool(self.config.get("roi_enabled", True)) and bool(self.config.get("show_expected_zone_box", False)):
            zone_rect = expected_zone_to_pixels(self.config, side, preview)
            x, y, w, h = zone_rect
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 160, 255), 2)
            draw_label(preview, "Expected Zone", x, max(22, y), (0, 160, 255))
        if not self.set_panel_frame(side, preview):
            return False

        self.side_status[side] = "READY"
        self.panels[side].status_label.setText(
            f"Status: READY | {info.get('width')}x{info.get('height')} | {float(info.get('fps') or 0):.1f} FPS"
        )
        self.status_labels[prefix].setText("READY")
        return True

    def select_video(self, side: str) -> None:
        action = f"select_video_{side}"
        if not self.begin_action(action):
            return
        try:
            if self.worker_running():
                self.set_app_state(AppState.STOPPING, f"switching {side} video")
                if not self.stop_worker_safely(f"switching {side} video"):
                    self.show_clean_error("Stop Error", "Could not stop the current inspection before changing video.")
                    return
                self.set_app_state(AppState.IDLE, "old worker stopped before video selection")
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {side.title()} Video",
                str(BASE_DIR),
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)",
            )
            if not path:
                logger.info("Video selection cancelled: side=%s", side)
                return
            logger.info("Video selected: side=%s path=%s", side, path)
            self.video_paths[side] = str(Path(path).expanduser().resolve())
            self.side_status[side] = "CHECKING"
            self.panels[side].video_label.setText("CHECKING VIDEO")
            self.panels[side].video_label.setPixmap(QPixmap())
            self.panels[side].status_label.setText("Status: CHECKING VIDEO")
            self.diagnose_and_preview_video(side)
            self.set_app_state(AppState.IDLE, "video selected")
            self.refresh_static_status()
        except Exception as exc:
            logger.exception("Video selection failed: side=%s", side)
            self.show_clean_error("Video Error", str(exc))
        finally:
            self.end_action(action)

    def select_model(self) -> None:
        if not self.begin_action("select_model"):
            return
        try:
            if self.worker_running():
                self.set_app_state(AppState.STOPPING, "model switch")
                if not self.stop_worker_safely("model switch"):
                    self.show_clean_error("Stop Error", "Could not stop the current inspection before changing model.")
                    return
                self.set_app_state(AppState.IDLE, "old worker stopped before model selection")
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select YOLO Model",
                str(BASE_DIR),
                "YOLO Model (*.pt);;All Files (*)",
            )
            if not path:
                logger.info("Model selection cancelled")
                return
            self.set_app_state(AppState.LOADING_MODEL, "manual model selection")
            self.config["MODEL_PATH"] = path
            self.model_manager.load_model(path)
            self.refresh_static_status()
            if self.model_manager.error:
                self.show_clean_error("Model Error", self.model_manager.error)
            else:
                self.set_app_state(AppState.IDLE, "model loaded")
        except Exception as exc:
            logger.exception("Model selection failed")
            self.show_clean_error("Model Error", str(exc))
        finally:
            self.end_action("select_model")

    def create_worker(
        self,
        sources: dict[str, str],
        source_mode: str,
        camera_options: dict[str, Any] | None = None,
    ) -> DualVideoWorker:
        return DualVideoWorker(
            video_paths=sources,
            model_manager=self.model_manager,
            confidence=float(self.config.get("CONF_THRESHOLD", 0.35)),
            imgsz=int(self.config.get("IMG_SIZE", 416)),
            frame_skip=int(self.config.get("FRAME_SKIP", 2)),
            display_width=self.current_display_width(),
            display_height=self.current_display_height(),
            require_both=bool(self.config.get("REQUIRE_BOTH_STICKERS", False)),
            smoothing_window=int(self.config.get("SMOOTHING_WINDOW", 12)),
            min_present_frames=int(self.config.get("MIN_PRESENT_FRAMES", 3)),
            config=self.config,
            source_mode=source_mode,
            camera_options=camera_options,
        )

    def start_both(self) -> None:
        if not self.begin_action("start_video"):
            return
        try:
            if not self.model_manager.ready:
                QMessageBox.warning(self, "Model Error", self.model_manager.error or "Model is not loaded.")
                return
            if self.worker_running():
                logger.info("Ignored Start click; worker already running state=%s", self.app_state.value)
                return
            if not self.video_paths["left"] and not self.video_paths["right"]:
                self.side_status = {"left": "NO VIDEO", "right": "NO VIDEO"}
                self.refresh_static_status()
                return

            self.stop_worker_safely("pre-start cleanup")
            self.current_final_decision = "WAITING"
            valid_video_paths: dict[str, str] = {}
            for side in ("left", "right"):
                if self.video_paths[side]:
                    if self.diagnose_and_preview_video(side):
                        valid_video_paths[side] = self.video_paths[side]
                    else:
                        logger.warning("Skipping %s video because diagnostics failed", side)
                else:
                    self.side_status[side] = "NO VIDEO"

            if not valid_video_paths:
                self.current_final_decision = "WAITING"
                self.refresh_static_status()
                return

            self.active_source_mode = "video_test"
            for side in valid_video_paths:
                self.side_status[side] = "RUNNING"
                self.panels[side].status_label.setText("Status: RUNNING")

            self.worker = self.create_worker(valid_video_paths, "video_test")
            self.worker.frame_ready.connect(self.on_frame_ready)
            self.worker.status_ready.connect(self.on_status_ready)
            self.worker.finished_ready.connect(self.on_finished)
            self.worker.start()
            self.set_app_state(AppState.RUNNING_VIDEO, "video worker started")
            self.refresh_static_status()
            logger.info("Detection started")
        except Exception as exc:
            logger.exception("Video detection start failed")
            self.stop_worker_safely("start_video exception cleanup")
            self.show_clean_error("Start Error", str(exc))
        finally:
            self.end_action("start_video")

    def start_live_cameras(self, auto: bool = False) -> None:
        if not self.begin_action("start_camera"):
            return
        try:
            if not self.model_manager.ready:
                if not auto:
                    QMessageBox.warning(self, "Model Error", self.model_manager.error or "Model is not loaded.")
                return
            if self.worker_running():
                logger.info("Ignored live camera start; worker already running state=%s", self.app_state.value)
                return

            self.stop_worker_safely("pre-live-start cleanup")
            self.camera_config = load_camera_config()
            cameras = self.camera_config.get("cameras", {})
            live_sources: dict[str, str] = {}
            for side in ("left", "right"):
                camera_info = cameras.get(side, {})
                prefix = "Left" if side == "left" else "Right"
                if not camera_info.get("enabled", False):
                    self.status_labels[f"{prefix} Cam"].setText("DISABLED")
                    self.side_status[side] = "NO VIDEO"
                    self.panels[side].status_label.setText("Status: CAMERA DISABLED")
                    continue
                source = str(camera_info.get("source", "")).strip()
                if not source:
                    self.status_labels[f"{prefix} Cam"].setText("DISCONNECTED")
                    self.side_status[side] = "NO VIDEO"
                    self.panels[side].status_label.setText("Status: CAMERA SOURCE MISSING")
                    continue
                live_sources[side] = source
                self.status_labels[f"{prefix} Cam"].setText("CONNECTING")
                self.status_labels[f"{prefix} FPS"].setText("0.0")
                self.side_status[side] = "WAITING"
                self.panels[side].video_label.setText("CONNECTING CAMERA")
                self.panels[side].video_label.setPixmap(QPixmap())
                self.panels[side].status_label.setText("Status: CONNECTING CAMERA")

            if not live_sources:
                self.current_final_decision = "WAITING"
                self.active_source_mode = "video_test"
                self.refresh_static_status()
                live_logger.warning("Live camera mode requested but no enabled RTSP sources were configured")
                return

            self.active_source_mode = "live"
            self.current_final_decision = "WAITING"
            self.worker = self.create_worker(live_sources, "live", self.camera_config)
            self.worker.frame_ready.connect(self.on_frame_ready)
            self.worker.status_ready.connect(self.on_status_ready)
            self.worker.finished_ready.connect(self.on_finished)
            self.worker.start()
            self.set_app_state(AppState.RUNNING_CAMERA, "live worker started")
            self.refresh_static_status()
            live_logger.info("Live camera detection started")
            logger.info("Live camera detection started")
        except Exception as exc:
            logger.exception("Live camera start failed")
            self.stop_worker_safely("start_camera exception cleanup")
            self.show_clean_error("Camera Error", str(exc))
        finally:
            self.end_action("start_camera")

    def stop_both(self) -> None:
        if not self.begin_action("stop"):
            return
        previous_state = self.app_state
        try:
            self.set_app_state(AppState.STOPPING, "stop requested")
            stopped = self.stop_worker_safely("stop button")
            for side in ("left", "right"):
                prefix = "Left" if side == "left" else "Right"
                if self.video_paths[side] or previous_state == AppState.RUNNING_CAMERA or self.active_source_mode == "live":
                    self.side_status[side] = "STOPPED"
                    self.panels[side].status_label.setText("Status: STOPPED")
                    if previous_state == AppState.RUNNING_CAMERA or self.active_source_mode == "live":
                        self.status_labels[f"{prefix} Cam"].setText("STOPPED")
            self.set_app_state(AppState.IDLE if stopped else AppState.ERROR, "stop completed" if stopped else "worker stop timed out")
            logger.info("Detection stopped")
            live_logger.info("Detection stopped")
            self.refresh_static_status()
        except Exception:
            logger.exception("Stop failed")
            self.set_app_state(AppState.ERROR, "stop failed")
        finally:
            self.end_action("stop")

    def clear_results(self) -> None:
        if not self.begin_action("reset"):
            return
        try:
            if self.worker_running():
                logger.info("Ignored reset while worker is running")
                return
            self.current_final_decision = "WAITING"
            for side in ("left", "right"):
                self.side_status[side] = "NO VIDEO" if not self.video_paths[side] else "READY"
                self.panels[side].video_label.setPixmap(QPixmap())
                self.panels[side].video_label.setText("READY" if self.video_paths[side] else "NO VIDEO")
                self.panels[side].status_label.setText(f"Status: {self.side_status[side]}")
                prefix = "Left" if side == "left" else "Right"
                camera_info = self.camera_config.get("cameras", {}).get(side, {})
                self.status_labels[f"{prefix} Cam"].setText("ENABLED" if camera_info.get("enabled") else "DISABLED")
            for key in self.status_labels:
                if key not in ("Model", "Device", "Left", "Right", "Mode", "Left Cam", "Right Cam"):
                    self.status_labels[key].setText("-")
            self.set_app_state(AppState.IDLE, "reset completed")
            self.refresh_static_status()
        except Exception:
            logger.exception("Reset failed")
            self.set_app_state(AppState.ERROR, "reset failed")
        finally:
            self.end_action("reset")

    def on_frame_ready(self, side: str, image: QImage, metadata: dict) -> None:
        if self.app_state not in (AppState.RUNNING_VIDEO, AppState.RUNNING_CAMERA):
            logger.debug("Ignored stale frame signal: side=%s state=%s", side, self.app_state.value)
            return
        try:
            panel = self.panels[side]
            pixmap = QPixmap.fromImage(image)
            label_size = panel.video_label.size()
            if pixmap.width() > label_size.width() or pixmap.height() > label_size.height():
                pixmap = pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            panel.video_label.setPixmap(pixmap)
        except Exception:
            logger.exception("DISPLAY UPDATE FAILED for %s", side)
            self.side_status[side] = "DISPLAY UPDATE FAILED"
            self.panels[side].status_label.setText("Status: DISPLAY UPDATE FAILED")
            self.status_labels["Left" if side == "left" else "Right"].setText("DISPLAY UPDATE FAILED")
            return

        status = display_status(str(metadata["status"]))
        self.side_status[side] = status
        self.current_final_decision = metadata.get("final_decision", self.current_final_decision)
        camera_status = str(metadata.get("camera_status", "CONNECTED"))
        if metadata.get("source_mode") == "live":
            panel.status_label.setText(
                f"Status: {status} | Camera: {camera_status} | Frame {metadata['frame']} | FPS {metadata['camera_fps']:.1f}"
            )
        else:
            panel.status_label.setText(
                f"Status: {status} | Frame {metadata['frame']} | FPS {metadata['fps']:.1f}"
            )

        prefix = "Left" if side == "left" else "Right"
        fps_text = f"{metadata['fps']:.1f}"
        if metadata.get("source_mode") == "live":
            self.status_labels[f"{prefix} Cam"].setText(camera_status)
            self.status_labels[f"{prefix} FPS"].setText(f"{float(metadata.get('camera_fps', 0.0)):.1f}")
        elif self.video_paths["left"] and self.video_paths["right"]:
            self.status_labels["Left FPS"].setText(fps_text)
            self.status_labels["Right FPS"].setText(fps_text)
        else:
            self.status_labels[f"{prefix} FPS"].setText(fps_text)
        self.status_labels[prefix].setText(status)
        self.update_final_result()

    def on_status_ready(self, side: str, metadata: dict) -> None:
        if self.app_state == AppState.IDLE and self.worker is None:
            logger.debug("Ignored stale status signal: side=%s metadata=%s", side, metadata)
            return
        prefix = "Left" if side == "left" else "Right"
        if metadata.get("camera_health"):
            camera_status = str(metadata.get("camera_status", "DISCONNECTED"))
            camera_fps = float(metadata.get("camera_fps", 0.0) or 0.0)
            self.status_labels[f"{prefix} Cam"].setText(camera_status)
            self.status_labels[f"{prefix} FPS"].setText(f"{camera_fps:.1f}")
            current = self.side_status.get(side, "WAITING")
            error = str(metadata.get("error", "") or "")
            detail = f" | {error}" if error and camera_status == "DISCONNECTED" else ""
            self.panels[side].status_label.setText(
                f"Status: {current} | Camera: {camera_status} | FPS {camera_fps:.1f}{detail}"
            )
            if camera_status == "DISCONNECTED" and current in ("NO VIDEO", "WAITING", "RUNNING", "READY"):
                self.panels[side].video_label.setText("CAMERA DISCONNECTED")
            return

        status = display_status(str(metadata.get("status", "NO VIDEO")))
        if status == "ENDED":
            self.panels[side].status_label.setText("Status: ENDED")
        else:
            self.side_status[side] = status
            self.panels[side].status_label.setText(f"Status: {status}")
        error = metadata.get("error")
        if error:
            self.panels[side].video_label.setText(error)
        self.refresh_static_status()

    def on_finished(self, metadata: dict) -> None:
        logger.info("Worker finished signal received: metadata=%s state=%s", metadata, self.app_state.value)
        finished_mode = self.active_source_mode
        self.worker = None
        if metadata.get("status") == "ENDED":
            for side in ("left", "right"):
                if self.video_paths[side]:
                    self.panels[side].status_label.setText("Status: ENDED")
        elif finished_mode == "live":
            for side in ("left", "right"):
                self.status_labels[f"{'Left' if side == 'left' else 'Right'} Cam"].setText("STOPPED")
        if not self.is_closing and self.app_state in (AppState.RUNNING_VIDEO, AppState.RUNNING_CAMERA, AppState.STOPPING):
            self.set_app_state(AppState.IDLE, "worker finished")
        self.update_final_result()
        self.update_control_buttons()

    def update_final_result(self) -> None:
        result = self.current_final_decision
        if result in (FINAL_OK, "PASS"):
            color = "#078a4a"
            text_color = "#ffffff"
            label = "PLA CARD OK"
        elif result in (FINAL_NG, "FAIL", "CONFIRMED_FAIL"):
            color = "#c62828"
            text_color = "#ffffff"
            label = "PLA CARD NG"
        elif result == FINAL_MISSING:
            color = "#d7dde5"
            text_color = "#1f252b"
            label = "MISSING"
        else:
            color = "#8d99ae"
            text_color = "#101114"
            label = "WAITING" if result == FINAL_WAITING else final_decision_label(str(result))
        self.final_result.setText(label)
        self.final_result.setStyleSheet(f"background: {color}; color: {text_color};")

    def closeEvent(self, event: QCloseEvent) -> None:
        logger.info("Application close requested")
        self.is_closing = True
        try:
            self.set_app_state(AppState.STOPPING, "application close")
            self.stop_worker_safely("application close", wait_ms=4000)
            if self.model_manager.model_path:
                self.config["MODEL_PATH"] = config_model_path_value(self.model_manager.model_path)
            save_config(self.config)
            logger.info("Application cleanup complete")
        except Exception:
            logger.exception("Application close cleanup failed")
        finally:
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception:
                    pass
            event.accept()


def main() -> int:
    logger.info("App started")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    window = InspectionWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
