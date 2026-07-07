from __future__ import annotations

import shutil
import sys
from pathlib import Path


DESKTOP = Path.home() / "Desktop"
DATA_YAML = DESKTOP / "sticker project yolo dataset" / "data.yaml"
TRAINING_RUNS_DIR = DESKTOP / "sticker project training runs"
TRAINED_MODEL_DIR = DESKTOP / "sticker project trained model"
FINAL_MODEL_PATH = TRAINED_MODEL_DIR / "best_sticker_yolov8.pt"

MODEL_NAME = "yolov8s.pt"
EPOCHS = 70
IMAGE_SIZE = 640
BATCH_SIZE_CPU = 8
BATCH_SIZE_GPU = 4
RUN_NAME = "sticker_yolov8s_70_epochs"


def fail(message: str) -> int:
    print(f"ERROR: {message}")
    return 1


def validate_dataset() -> int:
    if not DATA_YAML.exists():
        return fail(f"Missing data.yaml: {DATA_YAML}")
    root = DATA_YAML.parent
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            return fail(f"Missing dataset split folders for {split}")
        image_count = sum(1 for p in image_dir.iterdir() if p.is_file())
        label_count = sum(1 for p in label_dir.iterdir() if p.is_file())
        if image_count == 0:
            return fail(f"images/{split} is empty")
        if image_count != label_count:
            return fail(f"{split}: {image_count} images but {label_count} labels")
    return 0


def main() -> int:
    status = validate_dataset()
    if status:
        return status

    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        return fail(f"Training dependencies are not ready: {exc}")

    device = 0 if torch.cuda.is_available() else "cpu"
    batch = BATCH_SIZE_GPU if torch.cuda.is_available() else BATCH_SIZE_CPU

    print(f"Dataset: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {'GPU 0' if device == 0 else 'CPU'}")
    print(f"Batch: {batch}")
    print(f"Runs folder: {TRAINING_RUNS_DIR}")

    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=device,
        project=str(TRAINING_RUNS_DIR),
        name=RUN_NAME,
        plots=True,
        val=True,
        workers=0,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        return fail(f"Training finished, but best.pt was not found: {best_path}")

    TRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, FINAL_MODEL_PATH)
    print("\nTraining complete.")
    print(f"Best model: {best_path}")
    print(f"Final copied model: {FINAL_MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
