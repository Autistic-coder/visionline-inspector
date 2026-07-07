from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

DESKTOP = Path.home() / "Desktop"
DEFAULT_ANNOTATED_DIR = DESKTOP / "sticker project annotated photos"
DEFAULT_NEGATIVE_LEFT_DIR = DESKTOP / "sticker project unannotated photos" / "left side" / "ng"
DEFAULT_NEGATIVE_RIGHT_DIR = DESKTOP / "sticker project unannotated photos" / "right side" / "ng"
DEFAULT_OUTPUT_DIR = DESKTOP / "sticker project yolo dataset"


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    label_path: Path | None
    kind: str
    prefix: str


@dataclass
class SplitResult:
    train: list[DatasetItem]
    val: list[DatasetItem]
    test: list[DatasetItem]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the one-class sticker YOLOv8 dataset.")
    parser.add_argument("--annotated-dir", type=Path, default=DEFAULT_ANNOTATED_DIR)
    parser.add_argument("--negative-left-dir", type=Path, default=DEFAULT_NEGATIVE_LEFT_DIR)
    parser.add_argument("--negative-right-dir", type=Path, default=DEFAULT_NEGATIVE_RIGHT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_non_empty_label_lines(label_path: Path) -> list[str]:
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = label_path.read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


def validate_yolo_label(label_path: Path, lines: list[str]) -> list[str]:
    errors: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path} line {line_number}: expected 5 values, found {len(parts)}")
            continue
        if parts[0] != "0":
            errors.append(f"{label_path} line {line_number}: class must be 0 sticker, found {parts[0]}")
            continue
        for name, raw in zip(("x_center", "y_center", "width", "height"), parts[1:], strict=True):
            try:
                value = float(raw)
            except ValueError:
                errors.append(f"{label_path} line {line_number}: {name} is not numeric: {raw}")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"{label_path} line {line_number}: {name} must be between 0 and 1: {value}")
        try:
            if float(parts[3]) <= 0 or float(parts[4]) <= 0:
                errors.append(f"{label_path} line {line_number}: width and height must be greater than 0")
        except ValueError:
            pass
    return errors


def require_dir(path: Path, description: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing {description}: {path}")
    elif not path.is_dir():
        errors.append(f"{description} is not a folder: {path}")


def scan_annotated_folder(folder: Path) -> tuple[list[DatasetItem], list[DatasetItem], int, list[str]]:
    positives: list[DatasetItem] = []
    negatives: list[DatasetItem] = []
    errors: list[str] = []
    skipped = 0

    print(f"Scanning annotated folder: {folder}")
    for image_path in sorted(folder.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            if image_path.suffix.lower() != ".txt":
                skipped += 1
            continue

        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            errors.append(f"Image is missing matching YOLO label: {image_path}")
            continue

        lines = read_non_empty_label_lines(label_path)
        if not lines:
            negatives.append(DatasetItem(image_path, None, "negative", "neg_annotated"))
            continue

        label_errors = validate_yolo_label(label_path, lines)
        if label_errors:
            errors.extend(label_errors)
            continue
        positives.append(DatasetItem(image_path, label_path, "positive", "pos"))

    return positives, negatives, skipped, errors


def scan_negative_folder(folder: Path, prefix: str) -> tuple[list[DatasetItem], int]:
    negatives: list[DatasetItem] = []
    skipped = 0
    print(f"Scanning negative folder: {folder}")
    for image_path in sorted(folder.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped += 1
            continue
        negatives.append(DatasetItem(image_path, None, "negative", prefix))
    return negatives, skipped


def split_items(items: list[DatasetItem], seed: int) -> SplitResult:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total == 0:
        return SplitResult([], [], [])
    if total == 1:
        return SplitResult(shuffled, [], [])
    if total == 2:
        return SplitResult([shuffled[0]], [shuffled[1]], [])
    train_count = max(1, int(total * 0.70))
    val_count = max(1, int(total * 0.20))
    if total - train_count - val_count < 1:
        train_count = max(1, train_count - 1)
    return SplitResult(
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def safe_prepare_output(output_dir: Path) -> None:
    for split in SPLITS:
        for folder_type in ("images", "labels"):
            path = output_dir / folder_type / split
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    for file_name in ("data.yaml", "dataset_manifest.csv"):
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def target_stem(item: DatasetItem, counter: int) -> str:
    digest = hashlib.sha1(str(item.image_path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{item.prefix}_{counter:06d}_{digest}"


def copy_item(item: DatasetItem, output_dir: Path, split: str, counter: int) -> dict[str, str]:
    stem = target_stem(item, counter)
    target_image = output_dir / "images" / split / f"{stem}{item.image_path.suffix.lower()}"
    target_label = output_dir / "labels" / split / f"{stem}.txt"
    shutil.copy2(item.image_path, target_image)
    if item.label_path is None:
        target_label.write_text("", encoding="utf-8")
    else:
        shutil.copy2(item.label_path, target_label)
    return {
        "split": split,
        "kind": item.kind,
        "target_image": str(target_image),
        "target_label": str(target_label),
        "source_image": str(item.image_path),
        "source_label": str(item.label_path or ""),
    }


def write_data_yaml(output_dir: Path) -> None:
    dataset_path = str(output_dir.resolve()).replace("'", "''")
    (output_dir / "data.yaml").write_text(
        f"path: '{dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: sticker\n",
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, rows: list[dict[str, str]]) -> None:
    with (output_dir / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "kind", "target_image", "target_label", "source_image", "source_label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def count_split(output_dir: Path, split: str) -> tuple[int, int]:
    images = [p for p in (output_dir / "images" / split).iterdir() if p.is_file()]
    labels = [p for p in (output_dir / "labels" / split).iterdir() if p.is_file()]
    return len(images), len(labels)


def main() -> int:
    args = parse_args()
    annotated_dir = args.annotated_dir.expanduser()
    negative_left_dir = args.negative_left_dir.expanduser()
    negative_right_dir = args.negative_right_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    errors: list[str] = []
    require_dir(annotated_dir, "annotated image folder", errors)
    require_dir(negative_left_dir, "left-side ng folder", errors)
    require_dir(negative_right_dir, "right-side ng folder", errors)
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors))
        return 1

    positives, annotated_negatives, skipped_annotated, label_errors = scan_annotated_folder(annotated_dir)
    left_negatives, skipped_left = scan_negative_folder(negative_left_dir, "neg_left")
    right_negatives, skipped_right = scan_negative_folder(negative_right_dir, "neg_right")
    negatives = annotated_negatives + left_negatives + right_negatives
    errors.extend(label_errors)

    if not positives:
        errors.append("No sticker-present images found. Non-empty YOLO labels are required.")
    if not negatives:
        errors.append("No no-sticker/background images found.")
    if errors:
        print("\nDataset errors:")
        for error in errors:
            print(f"- {error}")
        return 1

    pos_split = split_items(positives, args.seed)
    neg_split = split_items(negatives, args.seed + 1)

    split_errors: list[str] = []
    for split in SPLITS:
        if len(getattr(pos_split, split)) + len(getattr(neg_split, split)) == 0:
            split_errors.append(f"{split} split is empty")
    if split_errors:
        print("\n".join(f"ERROR: {e}" for e in split_errors))
        return 1

    print(f"\nCreating dataset at: {output_dir}")
    safe_prepare_output(output_dir)
    manifest_rows: list[dict[str, str]] = []
    counter = 1
    for split in SPLITS:
        for item in getattr(pos_split, split) + getattr(neg_split, split):
            manifest_rows.append(copy_item(item, output_dir, split, counter))
            counter += 1

    write_data_yaml(output_dir)
    write_manifest(output_dir, manifest_rows)

    for split in SPLITS:
        image_count, label_count = count_split(output_dir, split)
        if image_count == 0 or image_count != label_count:
            print(f"ERROR: generated {split} has {image_count} images and {label_count} labels")
            return 1

    skipped = skipped_annotated + skipped_left + skipped_right
    print("\nDataset summary")
    print("---------------")
    print(f"Total sticker-present annotated images: {len(positives)}")
    print(f"Total no-sticker/background images: {len(negatives)}")
    print(f"  Empty labels in annotated folder: {len(annotated_negatives)}")
    print(f"  Left/right ng images: {len(left_negatives) + len(right_negatives)}")
    print(f"Train positive/negative: {len(pos_split.train)}/{len(neg_split.train)}")
    print(f"Val positive/negative: {len(pos_split.val)}/{len(neg_split.val)}")
    print(f"Test positive/negative: {len(pos_split.test)}/{len(neg_split.test)}")
    print(f"Skipped unrelated files: {skipped}")
    print("Errors: 0")
    print(f"\nDone: {output_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
