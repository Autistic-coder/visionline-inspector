from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DESKTOP = Path.home() / "Desktop"
DEFAULT_MODEL_PATH = DESKTOP / "sticker project trained model" / "best_sticker_yolov8.pt"
DEFAULT_TEST_IMAGES_DIR = DESKTOP / "sticker project yolo dataset" / "images" / "test"
DEFAULT_OUTPUT_DIR = DESKTOP / "sticker project test predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the trained sticker model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf", type=float, default=0.35)
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"ERROR: {message}")
    return 1


def clear_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser()
    test_dir = args.test_images.expanduser()
    output_dir = args.output_dir.expanduser()

    if not model_path.exists():
        return fail(f"Model not found: {model_path}")
    if not test_dir.exists():
        return fail(f"Test image folder not found: {test_dir}")
    if not 0 <= args.conf <= 1:
        return fail("Confidence must be between 0 and 1")

    images = sorted(p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        return fail(f"No test images found in: {test_dir}")

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return fail(f"Test dependency missing: {exc}")

    clear_output(output_dir)
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(test_dir),
        conf=args.conf,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
    )

    detected = 0
    no_detection = 0
    pos_total = pos_detected = 0
    neg_total = neg_clean = 0
    for result in results:
        name = Path(result.path).name
        has_detection = result.boxes is not None and len(result.boxes) > 0
        detected += int(has_detection)
        no_detection += int(not has_detection)
        if name.startswith("pos_"):
            pos_total += 1
            pos_detected += int(has_detection)
        elif name.startswith(("neg_annotated_", "neg_left_", "neg_right_")):
            neg_total += 1
            neg_clean += int(not has_detection)

    print("\nTest summary")
    print("------------")
    print(f"Total test images: {len(images)}")
    print(f"Images with sticker detections: {detected}")
    print(f"Images with no detections: {no_detection}")
    print(f"Positive test images detected: {pos_detected}/{pos_total}")
    print(f"Negative test images with no detection: {neg_clean}/{neg_total}")
    print(f"Prediction images saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
