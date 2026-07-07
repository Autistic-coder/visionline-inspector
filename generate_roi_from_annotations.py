from __future__ import annotations

import json
import statistics
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config_app.json"
EXCLUDE_PARTS = {
    ".git",
    ".agents",
    "__pycache__",
    "build",
    "dist",
    "_internal",
    "Ultralytics",
    "sticker app",
    ".venv",
    "venv",
}
LEFT_WORDS = ("left", "lh", "driver")
RIGHT_WORDS = ("right", "rh", "passenger")
DEFAULT_ROI_SETTINGS = {
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
}


def candidate_roots() -> list[Path]:
    roots = [PROJECT_DIR]
    desktop = Path.home() / "Desktop"
    if desktop.exists():
        for child in desktop.iterdir():
            if not child.is_dir():
                continue
            name = child.name.lower()
            if "sticker" in name or "annotated" in name or "dataset" in name:
                if child.resolve() != (desktop / "sticker app").resolve():
                    roots.append(child)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen and root.exists():
            seen.add(key)
            unique.append(root)
    return unique


def excluded(path: Path) -> bool:
    return any(part in EXCLUDE_PARTS for part in path.parts)


def side_for_path(path: Path) -> str:
    text = str(path).lower()
    if any(word in text for word in LEFT_WORDS):
        return "left"
    if any(word in text for word in RIGHT_WORDS):
        return "right"
    return "shared"


def iter_label_files() -> list[Path]:
    files: list[Path] = []
    for root in candidate_roots():
        for path in root.rglob("*.txt"):
            if excluded(path):
                continue
            if "label" not in str(path.parent).lower():
                continue
            files.append(path)
    return files


def parse_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return boxes
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(1.0, cx + w / 2.0)
        y2 = min(1.0, cy + h / 2.0)
        boxes.append((x1, y1, x2, y2))
    return boxes


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def remove_center_outliers(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    if len(boxes) < 8:
        return boxes
    centers_x = [(x1 + x2) / 2.0 for x1, _y1, x2, _y2 in boxes]
    centers_y = [(y1 + y2) / 2.0 for _x1, y1, _x2, y2 in boxes]
    mx = statistics.median(centers_x)
    my = statistics.median(centers_y)
    distances = [abs(cx - mx) + abs(cy - my) for cx, cy in zip(centers_x, centers_y)]
    cutoff = percentile(distances, 0.90)
    return [box for box, dist in zip(boxes, distances) if dist <= cutoff]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def roi_from_boxes(
    boxes: list[tuple[float, float, float, float]],
    settings: dict,
    *,
    purpose: str,
) -> dict[str, float]:
    boxes = remove_center_outliers(boxes)
    if purpose == "search":
        margin_ratio = float(settings.get("search_roi_margin_ratio", DEFAULT_ROI_SETTINGS["search_roi_margin_ratio"]))
        min_w = float(settings.get("min_search_roi_width_ratio", DEFAULT_ROI_SETTINGS["min_search_roi_width_ratio"]))
        min_h = float(settings.get("min_search_roi_height_ratio", DEFAULT_ROI_SETTINGS["min_search_roi_height_ratio"]))
        max_w = float(settings.get("max_search_roi_width_ratio", DEFAULT_ROI_SETTINGS["max_search_roi_width_ratio"]))
        max_h = float(settings.get("max_search_roi_height_ratio", DEFAULT_ROI_SETTINGS["max_search_roi_height_ratio"]))
        low_pct = 0.02
        high_pct = 0.98
    else:
        margin_ratio = float(settings.get("expected_zone_margin_ratio", DEFAULT_ROI_SETTINGS["expected_zone_margin_ratio"]))
        min_w = float(settings.get("min_roi_width_ratio", DEFAULT_ROI_SETTINGS["min_roi_width_ratio"]))
        min_h = float(settings.get("min_roi_height_ratio", DEFAULT_ROI_SETTINGS["min_roi_height_ratio"]))
        max_w = float(settings.get("max_roi_width_ratio", DEFAULT_ROI_SETTINGS["max_roi_width_ratio"]))
        max_h = float(settings.get("max_roi_height_ratio", DEFAULT_ROI_SETTINGS["max_roi_height_ratio"]))
        low_pct = 0.10
        high_pct = 0.90

    centers_x = [(x1 + x2) / 2.0 for x1, _y1, x2, _y2 in boxes]
    centers_y = [(y1 + y2) / 2.0 for _x1, y1, _x2, y2 in boxes]
    center_x = statistics.median(centers_x)
    center_y = statistics.median(centers_y)

    x1 = percentile([box[0] for box in boxes], low_pct)
    y1 = percentile([box[1] for box in boxes], low_pct)
    x2 = percentile([box[2] for box in boxes], high_pct)
    y2 = percentile([box[3] for box in boxes], high_pct)
    w = max(0.02, x2 - x1)
    h = max(0.02, y2 - y1)
    margin_x = w * margin_ratio
    margin_y = h * margin_ratio
    x1 = max(0.0, x1 - margin_x)
    y1 = max(0.0, y1 - margin_y)
    x2 = min(1.0, x2 + margin_x)
    y2 = min(1.0, y2 + margin_y)
    w = clamp(x2 - x1, min_w, max_w)
    h = clamp(y2 - y1, min_h, max_h)
    x1 = clamp(center_x - w / 2.0, 0.0, 1.0 - w)
    y1 = clamp(center_y - h / 2.0, 0.0, 1.0 - h)
    return {
        "x_norm": round(x1, 6),
        "y_norm": round(y1, 6),
        "w_norm": round(w, 6),
        "h_norm": round(h, 6),
    }


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def main() -> None:
    grouped = {"left": [], "right": [], "shared": []}
    label_files = iter_label_files()
    for path in label_files:
        boxes = parse_boxes(path)
        if boxes:
            grouped[side_for_path(path)].extend(boxes)

    all_boxes = grouped["left"] + grouped["right"] + grouped["shared"]
    print(f"Label files scanned: {len(label_files)}")
    print(f"Annotation boxes used: {len(all_boxes)}")

    if not all_boxes:
        print("WARNING: no YOLO annotation boxes found. Existing ROI config was not changed.")
        return

    config = load_config()
    config.pop("conf_threshold", None)
    settings = {**DEFAULT_ROI_SETTINGS, **{key: config.get(key, value) for key, value in DEFAULT_ROI_SETTINGS.items()}}

    shared_search_roi = roi_from_boxes(all_boxes, settings, purpose="search")
    shared_expected_zone = roi_from_boxes(all_boxes, settings, purpose="expected")
    left_search_roi = roi_from_boxes(grouped["left"], settings, purpose="search") if len(grouped["left"]) >= 3 else shared_search_roi
    right_search_roi = roi_from_boxes(grouped["right"], settings, purpose="search") if len(grouped["right"]) >= 3 else shared_search_roi
    left_expected_zone = roi_from_boxes(grouped["left"], settings, purpose="expected") if len(grouped["left"]) >= 3 else shared_expected_zone
    right_expected_zone = roi_from_boxes(grouped["right"], settings, purpose="expected") if len(grouped["right"]) >= 3 else shared_expected_zone

    config.update(
        {
            "CONF_THRESHOLD": 0.30,
            "roi_enabled": True,
            "roi_mode": "crop",
            "roi_source": "annotations",
            "use_search_roi": True,
            "show_roi_box": False,
            "show_debug_roi_box": False,
            "show_expected_zone_box": False,
            "loop_video": bool(config.get("loop_video", False)),
            **settings,
            "left_roi": left_expected_zone,
            "right_roi": right_expected_zone,
            "left_search_roi": left_search_roi,
            "right_search_roi": right_search_roi,
            "left_expected_zone": left_expected_zone,
            "right_expected_zone": right_expected_zone,
            "inspection_state_enabled": True,
            "use_confirmed_fail_only": True,
            "show_missing_instead_of_fail": True,
            "missing_status_color": "grey",
            "present_status_color": "green",
            "fail_status_color": "red",
            "fail_requires_active_inspection": True,
            "fail_after_expected_zone_missed": True,
            "missing_grace_frames": 30,
            "fail_confirmation_frames": int(config.get("fail_confirmation_frames", 20)),
            "fail_after_missing_frames": int(config.get("fail_after_missing_frames", 45)),
            "pass_hold_frames": int(config.get("pass_hold_frames", 15)),
            "no_car_reset_frames": int(config.get("no_car_reset_frames", 30)),
            "empty_roi_detection_enabled": bool(config.get("empty_roi_detection_enabled", True)),
            "empty_roi_texture_threshold": float(config.get("empty_roi_texture_threshold", 10.0)),
            "roi_annotation_boxes_used": len(all_boxes),
            "roi_label_files_scanned": len(label_files),
        }
    )
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Separate left ROI: {'yes' if len(grouped['left']) >= 3 else 'no, using shared ROI'}")
    print(f"Separate right ROI: {'yes' if len(grouped['right']) >= 3 else 'no, using shared ROI'}")
    print(f"Left search ROI: {left_search_roi}")
    print(f"Right search ROI: {right_search_roi}")
    print(f"Left expected zone: {left_expected_zone}")
    print(f"Right expected zone: {right_expected_zone}")
    print(f"Updated: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
