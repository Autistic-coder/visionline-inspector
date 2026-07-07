from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DESKTOP_DIR = Path.home() / "Desktop"
OLD_DATA_YAML = DESKTOP_DIR / "sticker project yolo dataset" / "data.yaml"
NEW_ANNOTATED_ROOT = DESKTOP_DIR / "new annotated photos"
MERGED_DATASET_DIR = PROJECT_DIR / "dataset"
CONFIG_PATH = PROJECT_DIR / "config_app.json"
RUNS_DIR = PROJECT_DIR / "runs"
RUN_NAME_BASE = "sticker_retrain_70_epochs"
LOG_PATH = PROJECT_DIR / "retrain_70_epoch_log.txt"
EPOCHS = 70
IMG_SIZE = 640
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path | None
    prefix: str
    source: str
    original_split: str | None = None
    missing_label_created: bool = False


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log(message: str = "") -> None:
    print(message, flush=True)


def fail(message: str) -> int:
    log(f"ERROR: {message}")
    return 1


def normalize_path(path: str) -> Path:
    raw = str(path).strip().strip("'").strip('"')
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def parse_data_yaml(path: Path) -> tuple[Path, dict[str, str], dict[int, str]]:
    text = path.read_text(encoding="utf-8")
    data_path = path.parent
    splits: dict[str, str] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("path:"):
            data_path = normalize_path(line.split(":", 1)[1])
            in_names = False
            continue
        for split in SPLITS:
            if line.startswith(f"{split}:"):
                splits[split] = line.split(":", 1)[1].strip().strip("'").strip('"')
                in_names = False
                break
        else:
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names:
                match = re.match(r"(\d+)\s*:\s*(.+)", line)
                if match:
                    names[int(match.group(1))] = match.group(2).strip().strip("'").strip('"')
    if not names:
        names = {0: "sticker"}
    return data_path, splits, names


def read_label_lines(label_path: Path) -> list[str]:
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = label_path.read_text(errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def validate_label(label_path: Path, names: dict[int, str]) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(read_label_lines(label_path), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path} line {line_number}: expected 5 values, found {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            errors.append(f"{label_path} line {line_number}: class id is not numeric: {parts[0]}")
            continue
        if class_id not in names:
            errors.append(f"{label_path} line {line_number}: class id {class_id} is not in current data.yaml names")
            continue
        for value_name, raw_value in zip(("x_center", "y_center", "width", "height"), parts[1:], strict=True):
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"{label_path} line {line_number}: {value_name} is not numeric: {raw_value}")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"{label_path} line {line_number}: {value_name} must be between 0 and 1: {value}")
        try:
            if float(parts[3]) <= 0.0 or float(parts[4]) <= 0.0:
                errors.append(f"{label_path} line {line_number}: width and height must be greater than 0")
        except ValueError:
            pass
    return errors


def load_config_model_path() -> Path:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    configured = str(config.get("MODEL_PATH", "")).strip()
    if not configured:
        raise FileNotFoundError("MODEL_PATH is empty in config_app.json")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_DIR / path


def backup_path_for(model_path: Path) -> Path:
    return model_path.with_name("best_backup_before_retrain.pt")


def unique_backup_path(model_path: Path) -> Path:
    exact = backup_path_for(model_path)
    if not exact.exists():
        return exact
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return model_path.with_name(f"best_backup_before_retrain_{timestamp}.pt")


def find_new_side_dirs() -> dict[str, Path]:
    if not NEW_ANNOTATED_ROOT.exists():
        raise FileNotFoundError(f"New annotated folder missing: {NEW_ANNOTATED_ROOT}")
    side_dirs: dict[str, Path] = {}
    for child in NEW_ANNOTATED_ROOT.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "left" in name and "left" not in side_dirs:
            side_dirs["left"] = child
        elif "right" in name and "right" not in side_dirs:
            side_dirs["right"] = child
    missing = [side for side in ("left", "right") if side not in side_dirs]
    if missing:
        raise FileNotFoundError(f"Missing new side folder(s): {', '.join(missing)} under {NEW_ANNOTATED_ROOT}")
    return side_dirs


def image_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def scan_old_dataset(data_root: Path, splits: dict[str, str], names: dict[int, str]) -> tuple[dict[str, list[DatasetItem]], list[str]]:
    items: dict[str, list[DatasetItem]] = {split: [] for split in SPLITS}
    errors: list[str] = []
    for split in SPLITS:
        image_dir = data_root / splits.get(split, f"images/{split}")
        label_dir = data_root / "labels" / split
        if not image_dir.exists():
            errors.append(f"Old dataset missing image split: {image_dir}")
            continue
        if not label_dir.exists():
            errors.append(f"Old dataset missing label split: {label_dir}")
            continue
        for image_path in image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Old dataset image missing label: {image_path}")
                continue
            errors.extend(validate_label(label_path, names))
            items[split].append(DatasetItem(image_path, label_path, "old", "old_dataset", split))
    return items, errors


def scan_new_side(side: str, folder: Path, names: dict[int, str]) -> tuple[list[DatasetItem], list[str]]:
    data_dir = folder / "obj_train_data"
    scan_dir = data_dir if data_dir.exists() else folder
    items: list[DatasetItem] = []
    errors: list[str] = []
    for image_path in image_files(scan_dir):
        label_path = image_path.with_suffix(".txt")
        missing_created = False
        if not label_path.exists():
            label_path = None
            missing_created = True
        else:
            errors.extend(validate_label(label_path, names))
        items.append(
            DatasetItem(
                image_path=image_path,
                label_path=label_path,
                prefix=f"{side}_new",
                source=f"new_{side}",
                original_split=None,
                missing_label_created=missing_created,
            )
        )
    return items, errors


def split_new_items(items: list[DatasetItem], seed: int) -> tuple[list[DatasetItem], list[DatasetItem]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    train_count = max(1, int(len(shuffled) * 0.80))
    if train_count >= len(shuffled):
        train_count = len(shuffled) - 1
    return shuffled[:train_count], shuffled[train_count:]


def target_stem(item: DatasetItem, counter: int) -> str:
    digest = hashlib.sha1(str(item.image_path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{item.prefix}_{counter:06d}_{digest}"


def prepare_dataset_dirs() -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            folder = MERGED_DATASET_DIR / kind / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
    for filename in ("data.yaml", "dataset_manifest.csv", "dataset_summary.txt"):
        path = MERGED_DATASET_DIR / filename
        if path.exists():
            path.unlink()


def copy_item(item: DatasetItem, split: str, counter: int) -> dict[str, str]:
    stem = target_stem(item, counter)
    image_target = MERGED_DATASET_DIR / "images" / split / f"{stem}{item.image_path.suffix.lower()}"
    label_target = MERGED_DATASET_DIR / "labels" / split / f"{stem}.txt"
    shutil.copy2(item.image_path, image_target)
    if item.label_path is None:
        label_target.write_text("", encoding="utf-8")
    else:
        shutil.copy2(item.label_path, label_target)
    label_lines = read_label_lines(label_target)
    return {
        "split": split,
        "source": item.source,
        "source_split": item.original_split or "",
        "kind": "positive" if label_lines else "negative",
        "created_empty_label": "yes" if item.missing_label_created else "no",
        "target_image": str(image_target),
        "target_label": str(label_target),
        "source_image": str(item.image_path),
        "source_label": str(item.label_path or ""),
    }


def write_data_yaml(names: dict[int, str]) -> Path:
    names_lines = "".join(f"  {idx}: {name}\n" for idx, name in sorted(names.items()))
    yaml_path = MERGED_DATASET_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: '{str(MERGED_DATASET_DIR.resolve()).replace(chr(39), chr(39) + chr(39))}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_lines}",
        encoding="utf-8",
    )
    return yaml_path


def write_manifest(rows: list[dict[str, str]]) -> None:
    with (MERGED_DATASET_DIR / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "source",
                "source_split",
                "kind",
                "created_empty_label",
                "target_image",
                "target_label",
                "source_image",
                "source_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_split(split: str) -> tuple[int, int, int, int]:
    images = image_files(MERGED_DATASET_DIR / "images" / split)
    labels = sorted((MERGED_DATASET_DIR / "labels" / split).glob("*.txt"))
    positives = 0
    negatives = 0
    for label in labels:
        if read_label_lines(label):
            positives += 1
        else:
            negatives += 1
    return len(images), len(labels), positives, negatives


def unique_run_name() -> str:
    candidate = RUN_NAME_BASE
    if not (RUNS_DIR / candidate).exists():
        return candidate
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{RUN_NAME_BASE}_{timestamp}"


def parse_final_metrics(results_csv: Path) -> dict[str, str]:
    if not results_csv.exists():
        return {}
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        return {}
    last = rows[-1]
    return {
        "epochs_completed": str(len(rows)),
        "precision": last.get("metrics/precision(B)", ""),
        "recall": last.get("metrics/recall(B)", ""),
        "mAP50": last.get("metrics/mAP50(B)", ""),
        "mAP50-95": last.get("metrics/mAP50-95(B)", ""),
    }


def build_merged_dataset() -> tuple[Path, dict[str, Any]]:
    data_root, splits, names = parse_data_yaml(OLD_DATA_YAML)
    old_items, old_errors = scan_old_dataset(data_root, splits, names)
    side_dirs = find_new_side_dirs()
    new_left, left_errors = scan_new_side("left", side_dirs["left"], names)
    new_right, right_errors = scan_new_side("right", side_dirs["right"], names)
    errors = old_errors + left_errors + right_errors
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        if len(errors) > 30:
            preview += f"\n- ... {len(errors) - 30} more errors"
        raise ValueError(f"Dataset validation failed:\n{preview}")

    left_train, left_val = split_new_items(new_left, SEED + 10)
    right_train, right_val = split_new_items(new_right, SEED + 20)

    prepare_dataset_dirs()
    rows: list[dict[str, str]] = []
    counter = 1
    for split in SPLITS:
        for item in old_items[split]:
            rows.append(copy_item(item, split, counter))
            counter += 1
    for split, split_items in (("train", left_train + right_train), ("val", left_val + right_val)):
        for item in split_items:
            rows.append(copy_item(item, split, counter))
            counter += 1

    data_yaml = write_data_yaml(names)
    write_manifest(rows)

    summary: dict[str, Any] = {
        "old_data_yaml": str(OLD_DATA_YAML),
        "old_data_root": str(data_root),
        "current_classes": names,
        "new_left_dir": str(side_dirs["left"]),
        "new_right_dir": str(side_dirs["right"]),
        "new_left_total": len(new_left),
        "new_right_total": len(new_right),
        "new_left_train": len(left_train),
        "new_left_val": len(left_val),
        "new_right_train": len(right_train),
        "new_right_val": len(right_val),
        "new_missing_labels_created": sum(1 for item in new_left + new_right if item.missing_label_created),
    }
    for split in SPLITS:
        image_count, label_count, positives, negatives = count_split(split)
        if image_count != label_count:
            raise ValueError(f"Merged {split} split has {image_count} images but {label_count} labels")
        summary[f"{split}_images"] = image_count
        summary[f"{split}_labels"] = label_count
        summary[f"{split}_positive_labels"] = positives
        summary[f"{split}_empty_labels"] = negatives

    (MERGED_DATASET_DIR / "dataset_summary.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    return data_yaml, summary


def main() -> int:
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        try:
            log("Sticker / PLA card YOLO retraining")
            log("==================================")
            log(f"Project: {PROJECT_DIR}")
            log(f"Old data.yaml: {OLD_DATA_YAML}")
            log(f"New annotated root: {NEW_ANNOTATED_ROOT}")
            log(f"Training log: {LOG_PATH}")

            app_model_path = load_config_model_path()
            if not app_model_path.exists():
                return fail(f"Current app model does not exist: {app_model_path}")
            backup_path = unique_backup_path(app_model_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(app_model_path, backup_path)
            log(f"Backed up current app model: {backup_path}")

            data_yaml, summary = build_merged_dataset()
            log("\nMerged dataset summary")
            log("----------------------")
            for key, value in summary.items():
                log(f"{key}: {value}")
            log(f"Merged data.yaml: {data_yaml}")

            import torch
            from ultralytics import YOLO

            device = 0 if torch.cuda.is_available() else "cpu"
            batch = 4 if torch.cuda.is_available() else 8
            run_name = unique_run_name()
            log("\nTraining configuration")
            log("----------------------")
            log(f"YOLO version: ultralytics")
            log(f"Starting weights: {app_model_path}")
            log(f"Run name: {run_name}")
            log(f"Epochs: {EPOCHS}")
            log(f"Image size: {IMG_SIZE}")
            log(f"Batch: {batch}")
            log(f"Device: {'CUDA GPU 0 - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            log("")

            model = YOLO(str(app_model_path))
            results = model.train(
                data=str(data_yaml),
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=batch,
                device=device,
                project=str(RUNS_DIR),
                name=run_name,
                plots=True,
                val=True,
                workers=0,
                exist_ok=False,
                verbose=True,
            )

            save_dir = Path(results.save_dir)
            new_best = save_dir / "weights" / "best.pt"
            if not new_best.exists():
                return fail(f"Training completed but new best.pt was not found: {new_best}")

            shutil.copy2(new_best, app_model_path)
            metrics = parse_final_metrics(save_dir / "results.csv")
            log("\nTraining finished")
            log("-----------------")
            log(f"path of the new best.pt: {new_best}")
            log(f"path of the backup old model: {backup_path}")
            log(f"total epochs completed: {metrics.get('epochs_completed', 'unknown')}")
            log(f"final precision: {metrics.get('precision', 'unknown')}")
            log(f"final recall: {metrics.get('recall', 'unknown')}")
            log(f"final mAP50: {metrics.get('mAP50', 'unknown')}")
            log(f"final mAP50-95: {metrics.get('mAP50-95', 'unknown')}")
            log(f"app model path updated successfully: {app_model_path.exists() and app_model_path.stat().st_size == new_best.stat().st_size}")
            return 0
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    sys.exit(main())
