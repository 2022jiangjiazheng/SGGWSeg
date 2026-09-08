"""Tile an already preprocessed Massachusetts raw_data train/test split.

Training images are tiled with configurable overlap. Test images are centred
in a zero-padded canvas whose dimensions are multiples of the tile size, then
cut into non-overlapping tiles.
"""

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_SOURCE = Path("raw_data")
DEFAULT_TARGET = Path("data")
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def collect_pairs(source_root, split):
    image_dir = source_root / "images" / split
    label_dir = source_root / "zones" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Missing raw_data {split} directories:\n  {image_dir}\n  {label_dir}"
        )
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    labels = {
        path.stem: path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            f"Unpaired raw_data files in {split}; missing labels={missing_labels[:10]}, "
            f"missing images={missing_images[:10]}"
        )
    return [(source_id, images[source_id], labels[source_id]) for source_id in sorted(images)]


def load_pair(image_path, label_path):
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(label_path) as label:
        gray = np.asarray(label.convert("L"), dtype=np.uint8)
    if rgb.shape[:2] != gray.shape:
        raise ValueError(
            f"Image/label shape mismatch: {image_path.name}={rgb.shape[:2]}, "
            f"{label_path.name}={gray.shape}"
        )
    unique_values = np.unique(gray)
    unexpected = unique_values[(unique_values != 0) & (unique_values != 255)]
    if unexpected.size:
        raise ValueError(
            f"Label must contain only 0 and 255, got {unique_values.tolist()}: {label_path}"
        )
    return rgb, gray


def sliding_starts(length, tile_size, overlap):
    """Cover a dimension; the final tile may overlap more to reach its boundary."""
    if length < tile_size:
        raise ValueError(f"Training dimension {length} is smaller than tile size {tile_size}.")
    if length == tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def train_tiles(image, label, tile_size, overlap):
    height, width = label.shape
    y_starts = sliding_starts(height, tile_size, overlap)
    x_starts = sliding_starts(width, tile_size, overlap)
    for row, top in enumerate(y_starts):
        for column, left in enumerate(x_starts):
            yield (
                row,
                column,
                left,
                top,
                image[top : top + tile_size, left : left + tile_size],
                label[top : top + tile_size, left : left + tile_size],
                0,
                0,
                0,
                0,
                width,
                height,
            )


def test_tiles(image, label, tile_size):
    """Centre-pad to tile-size multiples and yield non-overlapping tiles."""
    height, width = label.shape
    padded_height = max(tile_size, math.ceil(height / tile_size) * tile_size)
    padded_width = max(tile_size, math.ceil(width / tile_size) * tile_size)
    pad_left = (padded_width - width) // 2
    pad_right = padded_width - width - pad_left
    pad_top = (padded_height - height) // 2
    pad_bottom = padded_height - height - pad_top
    padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded_label = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded_image[pad_top : pad_top + height, pad_left : pad_left + width] = image
    padded_label[pad_top : pad_top + height, pad_left : pad_left + width] = label

    for row, top in enumerate(range(0, padded_height, tile_size)):
        for column, left in enumerate(range(0, padded_width, tile_size)):
            yield (
                row,
                column,
                left,
                top,
                padded_image[top : top + tile_size, left : left + tile_size],
                padded_label[top : top + tile_size, left : left + tile_size],
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                padded_width,
                padded_height,
            )


def save_png(image, label, image_path, label_path, compress_level):
    Image.fromarray(image, mode="RGB").save(
        image_path, format="PNG", compress_level=compress_level
    )
    Image.fromarray(label, mode="L").save(
        label_path, format="PNG", compress_level=compress_level
    )


def process_pair(task):
    (
        split,
        source_id,
        image_path,
        label_path,
        image_output_dir,
        label_output_dir,
        tile_size,
        train_overlap,
        compress_level,
    ) = task
    image, label = load_pair(image_path, label_path)
    original_height, original_width = label.shape
    iterator = (
        train_tiles(image, label, tile_size, train_overlap)
        if split == "train"
        else test_tiles(image, label, tile_size)
    )
    rows = []
    for tile_index, tile in enumerate(iterator, start=1):
        (
            row,
            column,
            left,
            top,
            image_tile,
            label_tile,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            padded_width,
            padded_height,
        ) = tile
        filename = f"{source_id}_part{tile_index}.png"
        save_png(
            image_tile,
            label_tile,
            image_output_dir / filename,
            label_output_dir / filename,
            compress_level,
        )
        rows.append(
            {
                "split": split,
                "source_id": source_id,
                "tile_id": f"part{tile_index}",
                "filename": filename,
                "row": row,
                "column": column,
                "left": left,
                "top": top,
                "tile_width": tile_size,
                "tile_height": tile_size,
                "original_width": original_width,
                "original_height": original_height,
                "padded_width": padded_width,
                "padded_height": padded_height,
                "pad_left": pad_left,
                "pad_right": pad_right,
                "pad_top": pad_top,
                "pad_bottom": pad_bottom,
            }
        )
    return split, rows


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
            f"Tiled output contains {len(existing)} files. Use an empty target "
            "or pass --overwrite."
        )
    if overwrite:
        for path in existing:
            path.unlink()
        for directory in legacy_val_dirs:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return output_dirs


def split_dataset(
    source_root,
    target_root,
    tile_size=1024,
    train_overlap=512,
    png_compress_level=1,
    workers=0,
    overwrite=False,
):
    source_root = Path(source_root)
    target_root = Path(target_root)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if train_overlap < 0 or train_overlap >= tile_size:
        raise ValueError("train_overlap must be in [0, tile_size).")
    if not 0 <= png_compress_level <= 9:
        raise ValueError("png_compress_level must be in [0, 9].")
    if workers < 0:
        raise ValueError("workers cannot be negative.")

    pairs = {split: collect_pairs(source_root, split) for split in ("train", "test")}
    train_ids = {source_id for source_id, _, _ in pairs["train"]}
    test_ids = {source_id for source_id, _, _ in pairs["test"]}
    if train_ids & test_ids:
        raise ValueError(f"Source leakage across train/test: {sorted(train_ids & test_ids)[:10]}")
    output_dirs = prepare_output_dirs(target_root, overwrite)
    tasks = []
    for split in ("train", "test"):
        for source_id, image_path, label_path in pairs[split]:
            tasks.append(
                (
                    split,
                    source_id,
                    image_path,
                    label_path,
                    output_dirs[split]["images"],
                    output_dirs[split]["labels"],
                    tile_size,
                    train_overlap,
                    png_compress_level,
                )
            )

    worker_count = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    print(f"Tiling {len(tasks)} full-size pairs with {worker_count} workers...")
    manifest_rows = []
    tile_counts = {"train": 0, "test": 0}
    if worker_count == 1:
        results = map(process_pair, tasks)
        for split, rows in tqdm(results, total=len(tasks)):
            manifest_rows.extend(rows)
            tile_counts[split] += len(rows)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(process_pair, tasks, chunksize=1)
            for split, rows in tqdm(results, total=len(tasks)):
                manifest_rows.extend(rows)
                tile_counts[split] += len(rows)

    for split, expected in tile_counts.items():
        image_count = len(list(output_dirs[split]["images"].glob("*.png")))
        label_count = len(list(output_dirs[split]["labels"].glob("*.png")))
        if image_count != expected or label_count != expected:
            raise RuntimeError(
                f"{split} tile count mismatch: expected {expected}, "
                f"images={image_count}, labels={label_count}"
            )

    manifest_rows.sort(
        key=lambda row: (row["split"], row["source_id"], row["row"], row["column"])
    )
    manifest_path = target_root / "tiles_manifest.csv"
    fieldnames = [
        "split",
        "source_id",
        "tile_id",
        "filename",
        "row",
        "column",
        "left",
        "top",
        "tile_width",
        "tile_height",
        "original_width",
        "original_height",
        "padded_width",
        "padded_height",
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nTiling complete")
    print(f"  raw train/test sources: {len(pairs['train'])}/{len(pairs['test'])}")
    print(f"  train/test tiles: {tile_counts['train']}/{tile_counts['test']}")
    print(f"  output: {target_root.resolve()}")
    return tile_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile preprocessed Massachusetts raw_data into train/test patches."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--train-overlap", type=int, default=0)
    parser.add_argument("--png-compress-level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        source_root=args.source,
        target_root=args.target,
        tile_size=args.tile_size,
        train_overlap=args.train_overlap,
        png_compress_level=args.png_compress_level,
        workers=args.workers,
        overwrite=args.overwrite,
    )
