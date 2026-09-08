"""Preprocess, filter and re-split the Massachusetts road dataset.

All official train/val/test images are first merged into one pool. Large
near-white or near-black blank components touching the image boundary are
converted to black, and road labels inside those invalid regions are removed.
Images with more than ``max_blank_ratio`` invalid area are rejected. The
remaining full-size images are randomly split into raw_data train/test sets.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_SOURCE = Path(
    r"/mnt/sdb2/jiangjiazheng/RoadNet2/Massachusetts_Road_Dataset/tiff"
)
DEFAULT_TARGET = Path("raw_data")
OFFICIAL_SPLITS = ("train", "val", "test")


def collect_official_pairs(source_root):
    """Merge official splits and verify one-to-one image/label pairing."""
    pairs = []
    seen = set()
    for official_split in OFFICIAL_SPLITS:
        image_dir = source_root / official_split
        label_dir = source_root / f"{official_split}_labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing official {official_split} directories:\n"
                f"  {image_dir}\n  {label_dir}"
            )
        images = {path.stem: path for path in image_dir.glob("*.tiff")}
        labels = {path.stem: path for path in label_dir.glob("*.tif")}
        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        if missing_labels or missing_images:
            raise ValueError(
                f"Unpaired files in official {official_split}; "
                f"missing labels={missing_labels[:10]}, "
                f"missing images={missing_images[:10]}"
            )
        for source_id in sorted(images):
            if source_id in seen:
                raise ValueError(f"Duplicate Source_ID across official splits: {source_id}")
            seen.add(source_id)
            pairs.append(
                (source_id, official_split, images[source_id], labels[source_id])
            )
    if not pairs:
        raise RuntimeError(f"No source pairs found in {source_root}.")
    return pairs


def load_pair(image_path, label_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(label_path) as label:
        gray = np.asarray(label.convert("L"), dtype=np.uint8)
    if rgb.shape[:2] != gray.shape:
        raise ValueError(
            f"Image/label shape mismatch: {image_path.name}={rgb.shape[:2]}, "
            f"{label_path.name}={gray.shape}"
        )
    # Normalize the label to an explicit binary 0/255 mask.
    binary_label = np.zeros_like(gray, dtype=np.uint8)
    binary_label[gray > 127] = 255
    return rgb, binary_label


def large_border_components(candidate_mask, min_component_area):
    """Keep large candidate components that touch any image boundary."""
    candidate = candidate_mask.astype(np.uint8)
    component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    height, width = candidate.shape
    result = np.zeros_like(candidate, dtype=bool)
    for component_id in range(1, component_count):
        left, top, component_width, component_height, area = stats[component_id]
        touches_border = (
            left == 0
            or top == 0
            or left + component_width == width
            or top + component_height == height
        )
        if touches_border and area >= min_component_area:
            result |= component_map == component_id
    return result


def detect_large_blank_mask(
    rgb,
    white_threshold,
    black_threshold,
    min_component_area,
    morphology_kernel,
):
    """Detect large near-white/near-black no-data regions connected to borders."""
    near_white = np.all(rgb >= white_threshold, axis=2)
    near_black = np.all(rgb <= black_threshold, axis=2)
    candidates = near_white | near_black
    if morphology_kernel > 1:
        kernel = np.ones((morphology_kernel, morphology_kernel), dtype=np.uint8)
        candidates = cv2.morphologyEx(
            candidates.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
    return large_border_components(candidates, min_component_area)


def preprocess_pair(
    image_path,
    label_path,
    white_threshold,
    black_threshold,
    min_component_area,
    morphology_kernel,
):
    """Return cleaned arrays and source-level quality statistics."""
    image, label = load_pair(image_path, label_path)
    blank_mask = detect_large_blank_mask(
        image,
        white_threshold=white_threshold,
        black_threshold=black_threshold,
        min_component_area=min_component_area,
        morphology_kernel=morphology_kernel,
    )
    removed_road_pixels = int(np.count_nonzero((label == 255) & blank_mask))
    image[blank_mask] = 0
    label[blank_mask] = 0
    blank_ratio = float(blank_mask.mean())
    road_ratio = float(np.count_nonzero(label == 255) / label.size)
    return image, label, {
        "width": int(label.shape[1]),
        "height": int(label.shape[0]),
        "blank_ratio": blank_ratio,
        "road_ratio": road_ratio,
        "removed_road_pixels": removed_road_pixels,
    }


def inspect_task(task):
    (
        source_id,
        official_split,
        image_path,
        label_path,
        white_threshold,
        black_threshold,
        min_component_area,
        morphology_kernel,
    ) = task
    _, _, stats = preprocess_pair(
        image_path,
        label_path,
        white_threshold,
        black_threshold,
        min_component_area,
        morphology_kernel,
    )
    return {
        "source_id": source_id,
        "official_split": official_split,
        "image_path": str(image_path),
        "label_path": str(label_path),
        **stats,
    }


def save_task(task):
    (
        source_id,
        image_path,
        label_path,
        final_split,
        image_output_dir,
        label_output_dir,
        white_threshold,
        black_threshold,
        min_component_area,
        morphology_kernel,
    ) = task
    image, label, _ = preprocess_pair(
        image_path,
        label_path,
        white_threshold,
        black_threshold,
        min_component_area,
        morphology_kernel,
    )
    filename = f"{source_id}.tif"
    Image.fromarray(image, mode="RGB").save(
        image_output_dir / filename,
        format="TIFF",
        compression="tiff_deflate",
    )
    Image.fromarray(label, mode="L").save(
        label_output_dir / filename,
        format="TIFF",
        compression="tiff_deflate",
    )
    return final_split, source_id


def prepare_output_dirs(target_root, overwrite):
    output_dirs = {
        split: {
            "images": target_root / "images" / split,
            "labels": target_root / "zones" / split,
        }
        for split in ("train", "test")
    }
    existing = []
    for split_dirs in output_dirs.values():
        for output_dir in split_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
            existing.extend(path for path in output_dir.iterdir() if path.is_file())
    legacy_val_dirs = (target_root / "images" / "val", target_root / "zones" / "val")
    for directory in legacy_val_dirs:
        if directory.is_dir():
            existing.extend(path for path in directory.iterdir() if path.is_file())
    if existing and not overwrite:
        raise FileExistsError(
            f"raw_data output contains {len(existing)} files. Use an empty target "
            "or pass --overwrite."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for directory in legacy_val_dirs:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return output_dirs


def preprocess_dataset(
    source_root,
    target_root,
    train_ratio=0.8,
    split_seed=42,
    max_blank_ratio=0.5,
    white_threshold=245,
    black_threshold=5,
    min_component_area=4096,
    morphology_kernel=5,
    workers=0,
    overwrite=False,
):
    source_root = Path(source_root)
    target_root = Path(target_root)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= max_blank_ratio <= 1:
        raise ValueError("max_blank_ratio must be in [0, 1].")
    if not 0 <= black_threshold < white_threshold <= 255:
        raise ValueError("Require 0 <= black_threshold < white_threshold <= 255.")
    if min_component_area <= 0:
        raise ValueError("min_component_area must be positive.")
    if morphology_kernel <= 0 or morphology_kernel % 2 == 0:
        raise ValueError("morphology_kernel must be a positive odd integer.")
    if workers < 0:
        raise ValueError("workers cannot be negative.")

    output_dirs = prepare_output_dirs(target_root, overwrite)
    pairs = collect_official_pairs(source_root)
    worker_count = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    inspect_tasks = [
        (
            source_id,
            official_split,
            image_path,
            label_path,
            white_threshold,
            black_threshold,
            min_component_area,
            morphology_kernel,
        )
        for source_id, official_split, image_path, label_path in pairs
    ]
    print(f"Inspecting {len(inspect_tasks)} merged source images with {worker_count} workers...")
    if worker_count == 1:
        records = list(tqdm(map(inspect_task, inspect_tasks), total=len(inspect_tasks)))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                tqdm(
                    executor.map(inspect_task, inspect_tasks, chunksize=1),
                    total=len(inspect_tasks),
                )
            )

    retained = [row for row in records if row["blank_ratio"] <= max_blank_ratio]
    rejected = [row for row in records if row["blank_ratio"] > max_blank_ratio]
    if len(retained) < 2:
        raise RuntimeError("Fewer than two images remain after blank-area filtering.")
    retained_ids = [row["source_id"] for row in retained]
    train_ids, test_ids = train_test_split(
        retained_ids,
        train_size=train_ratio,
        random_state=split_seed,
        shuffle=True,
    )
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    if train_ids & test_ids or train_ids | test_ids != set(retained_ids):
        raise RuntimeError("Source-level train/test split integrity check failed.")

    pair_lookup = {
        source_id: (image_path, label_path)
        for source_id, _, image_path, label_path in pairs
    }
    save_tasks = []
    for row in retained:
        source_id = row["source_id"]
        final_split = "train" if source_id in train_ids else "test"
        row["final_split"] = final_split
        image_path, label_path = pair_lookup[source_id]
        save_tasks.append(
            (
                source_id,
                image_path,
                label_path,
                final_split,
                output_dirs[final_split]["images"],
                output_dirs[final_split]["labels"],
                white_threshold,
                black_threshold,
                min_component_area,
                morphology_kernel,
            )
        )
    for row in rejected:
        row["final_split"] = "rejected"

    print(f"Saving {len(save_tasks)} retained full-size pairs...")
    if worker_count == 1:
        saved = list(tqdm(map(save_task, save_tasks), total=len(save_tasks)))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            saved = list(
                tqdm(
                    executor.map(save_task, save_tasks, chunksize=1),
                    total=len(save_tasks),
                )
            )
    saved_counts = {
        split: sum(saved_split == split for saved_split, _ in saved)
        for split in ("train", "test")
    }
    expected_counts = {"train": len(train_ids), "test": len(test_ids)}
    if saved_counts != expected_counts:
        raise RuntimeError(f"Saved counts differ from split counts: {saved_counts} != {expected_counts}")

    manifest_fields = [
        "source_id",
        "official_split",
        "final_split",
        "width",
        "height",
        "blank_ratio",
        "road_ratio",
        "removed_road_pixels",
        "image_path",
        "label_path",
    ]
    manifest_path = target_root / "preprocessing_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(
            {name: row[name] for name in manifest_fields}
            for row in sorted(records, key=lambda item: item["source_id"])
        )
    summary = {
        "source_count": len(records),
        "retained_count": len(retained),
        "rejected_count": len(rejected),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "max_blank_ratio": max_blank_ratio,
        "white_threshold": white_threshold,
        "black_threshold": black_threshold,
        "min_component_area": min_component_area,
        "morphology_kernel": morphology_kernel,
    }
    summary_path = target_root / "preprocessing_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nPreprocessing complete")
    print(f"  merged sources: {len(records)}")
    print(f"  rejected (> {max_blank_ratio:.2%} blank): {len(rejected)}")
    print(f"  raw train/test: {len(train_ids)}/{len(test_ids)}")
    print(f"  output: {target_root.resolve()}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess, filter and randomly split Massachusetts full-size images."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-blank-ratio", type=float, default=1.0)
    parser.add_argument("--white-threshold", type=int, default=245)
    parser.add_argument("--black-threshold", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=2048)
    parser.add_argument("--morphology-kernel", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(
        source_root=args.source,
        target_root=args.target,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        max_blank_ratio=args.max_blank_ratio,
        white_threshold=args.white_threshold,
        black_threshold=args.black_threshold,
        min_component_area=args.min_component_area,
        morphology_kernel=args.morphology_kernel,
        workers=args.workers,
        overwrite=args.overwrite,
    )
