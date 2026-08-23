# Project repository: https://github.com/2022jiangjiazheng
"""Create paired training patches from image and binary-mask rasters."""

import json
import re
import shutil
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# When this file is executed directly, put the project root before
# `data_processing/` so its local `utils.py` does not shadow `utils/raster.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.raster import (
    crop_patch,
    offset_geotransform,
    pad_bottom_right,
    pair_rasters,
    read_raster,
    spatial_shape,
    write_geotiff,
    write_png,
)


GROUP_DATE_PATTERN = re.compile(
    r"^(?P<group>.+)_\d{4}-\d{2}-\d{2}(?:_.+)?$"
)


def calculate_padding(length: int, patch_size: int, stride: int) -> int:
    """Return the minimum trailing padding required for complete regular coverage."""
    if length <= patch_size:
        return patch_size - length
    remainder = (length - patch_size) % stride
    return 0 if remainder == 0 else stride - remainder


def ensure_single_band_label(label: np.ndarray) -> np.ndarray:
    if label.ndim == 2:
        return label
    if label.ndim == 3 and (label.shape[0] == 1 or np.all(label == label[0])):
        return label[0]
    raise ValueError(f"Expected a single-band label, got shape {label.shape}")


def output_extension(output_format: str, geotransform, projection: str) -> str:
    if output_format == "png":
        return ".png"
    if output_format == "tif":
        return ".tif"
    return ".tif" if geotransform is not None or projection else ".png"


def write_patch(array, path: Path, geotransform, projection: str):
    if path.suffix == ".png":
        write_png(array, path)
    else:
        write_geotiff(array, path, geotransform, projection)


def process_pair(
    pair,
    split: str,
    output_dir: Path,
    patch_size: int,
    overlap: int,
    output_format: str,
    background_value: int,
    min_foreground_pixels: int,
    keep_empty: bool,
):
    """Read one image/label pair once and write valid patches incrementally."""
    identifier, image_path, label_path = pair
    image, image_transform, image_projection = read_raster(image_path)
    label, label_transform, label_projection = read_raster(label_path)
    label = ensure_single_band_label(label)

    if spatial_shape(image) != spatial_shape(label):
        raise ValueError(
            f"Image/label shape mismatch for {identifier}: "
            f"{spatial_shape(image)} != {spatial_shape(label)}"
        )

    height, width = spatial_shape(label)
    stride = patch_size - overlap
    bottom = calculate_padding(height, patch_size, stride)
    right = calculate_padding(width, patch_size, stride)
    image = pad_bottom_right(image, bottom, right)
    label = pad_bottom_right(label, bottom, right)

    image_extension = output_extension(output_format, image_transform, image_projection)
    label_extension = output_extension(output_format, label_transform, label_projection)
    image_output = output_dir / "images" / split
    label_output = output_dir / "zones" / split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    padded_height, padded_width = spatial_shape(label)
    saved = 0
    total = 0
    for row in range(0, padded_height - patch_size + 1, stride):
        for col in range(0, padded_width - patch_size + 1, stride):
            label_patch = crop_patch(label, row, col, patch_size)
            foreground_pixels = int(np.count_nonzero(label_patch != background_value))
            if not keep_empty and foreground_pixels < min_foreground_pixels:
                total += 1
                continue

            image_patch = crop_patch(image, row, col, patch_size)
            suffix = f"__{bottom}_{right}_{total}_{row}_{col}"
            image_patch_transform = offset_geotransform(image_transform, row, col)
            label_patch_transform = offset_geotransform(label_transform, row, col)

            write_patch(
                image_patch,
                image_output / f"{identifier}{suffix}{image_extension}",
                image_patch_transform,
                image_projection,
            )
            write_patch(
                label_patch,
                label_output / f"{identifier}_mask{suffix}{label_extension}",
                label_patch_transform,
                label_projection,
            )
            saved += 1
            total += 1

    return identifier, saved, total


def sample_group(identifier: str) -> str:
    """Return the wall identifier/name preceding the acquisition date."""
    match = GROUP_DATE_PATTERN.fullmatch(identifier)
    if match is None:
        raise ValueError(
            f"Cannot extract a wall group from '{identifier}'. Expected "
            "'<wall_id>_YYYY-MM-DD_<other_information>'."
        )
    return match.group("group")


def split_training_pairs(pairs, validation_ratio: float, seed: int):
    """Split by wall group so one wall can never occur in both train and val."""
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must satisfy 0 <= ratio < 1")
    groups = sorted({sample_group(pair[0]) for pair in pairs})
    if validation_ratio == 0 or len(groups) < 2:
        return pairs, []

    rng = np.random.default_rng(seed)
    validation_count = max(1, int(round(len(groups) * validation_ratio)))
    validation_count = min(validation_count, len(groups) - 1)
    shuffled_groups = rng.permutation(groups)
    validation_groups = set(shuffled_groups[:validation_count].tolist())
    train_pairs = [pair for pair in pairs if sample_group(pair[0]) not in validation_groups]
    validation_pairs = [pair for pair in pairs if sample_group(pair[0]) in validation_groups]
    return train_pairs, validation_pairs


def _select_indexed_pairs(pairs, identifiers, split_name: str):
    """Select raster pairs in the exact order recorded by a split-index file."""
    if not isinstance(identifiers, list) or not all(
        isinstance(identifier, str) for identifier in identifiers
    ):
        raise ValueError(f"Index entry '{split_name}' must be a list of sample identifiers")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Index entry '{split_name}' contains duplicate identifiers")

    pairs_by_id = {pair[0]: pair for pair in pairs}
    unknown = sorted(set(identifiers) - pairs_by_id.keys())
    if unknown:
        raise ValueError(f"Unknown identifiers in index entry '{split_name}': {unknown}")
    return [pairs_by_id[identifier] for identifier in identifiers]


def split_from_index(index_file: Path, training_pool, test_pairs):
    """Load an explicit, reproducible train/val/test split from JSON."""
    if not index_file.is_file():
        raise FileNotFoundError(f"Split-index file does not exist: {index_file}")
    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    if not isinstance(index_data, dict):
        raise ValueError("Split-index root must be a JSON object")
    required_entries = {"train", "val", "test"}
    missing_entries = sorted(required_entries - index_data.keys())
    if missing_entries:
        raise ValueError(f"Split-index file is missing entries: {missing_entries}")

    train_ids = index_data["train"]
    val_ids = index_data["val"]
    test_ids = index_data["test"]
    for split_name, identifiers in (
        ("train", train_ids),
        ("val", val_ids),
        ("test", test_ids),
    ):
        if not isinstance(identifiers, list) or not all(
            isinstance(identifier, str) for identifier in identifiers
        ):
            raise ValueError(
                f"Index entry '{split_name}' must be a list of sample identifiers"
            )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Index entry '{split_name}' contains duplicate identifiers")
    train_set = set(train_ids)
    val_set = set(val_ids)
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ValueError(f"Split-index train and val entries overlap: {overlap}")

    available_training_ids = {pair[0] for pair in training_pool}
    assigned_training_ids = train_set | val_set
    unassigned = sorted(available_training_ids - assigned_training_ids)
    unknown_training = sorted(assigned_training_ids - available_training_ids)
    if unassigned or unknown_training:
        raise ValueError(
            "Split-index does not exactly cover the training pool. "
            f"Unassigned={unassigned}; unknown={unknown_training}"
        )

    available_test_ids = {pair[0] for pair in test_pairs}
    if set(test_ids) != available_test_ids:
        raise ValueError(
            "Split-index test entry does not exactly match the test directory. "
            f"Unassigned={sorted(available_test_ids - set(test_ids))}; "
            f"unknown={sorted(set(test_ids) - available_test_ids)}"
        )

    return {
        "train": _select_indexed_pairs(training_pool, train_ids, "train"),
        "val": _select_indexed_pairs(training_pool, val_ids, "val"),
        "test": _select_indexed_pairs(test_pairs, test_ids, "test"),
    }


def prepare_splits(
    input_dir: Path,
    validation_ratio: float,
    seed: int,
    split_index_file: Path = None,
):
    train_pairs = pair_rasters(input_dir / "images" / "train", input_dir / "zones" / "train")
    explicit_validation = input_dir / "images" / "val"
    validation_pairs = []
    if explicit_validation.is_dir():
        validation_pairs = pair_rasters(explicit_validation, input_dir / "zones" / "val")

    test_images = input_dir / "images" / "test"
    test_pairs = (
        pair_rasters(test_images, input_dir / "zones" / "test")
        if test_images.is_dir() else []
    )

    if split_index_file is not None:
        training_pool = train_pairs + validation_pairs
        if len({pair[0] for pair in training_pool}) != len(training_pool):
            raise ValueError("Duplicate identifiers found across the input train and val folders")
        return split_from_index(split_index_file, training_pool, test_pairs)

    if explicit_validation.is_dir():
        train_groups = {sample_group(pair[0]) for pair in train_pairs}
        validation_groups = {sample_group(pair[0]) for pair in validation_pairs}
        overlapping_groups = sorted(train_groups & validation_groups)
        if overlapping_groups:
            raise ValueError(
                "The explicit train and val folders contain the same wall groups: "
                f"{overlapping_groups}"
            )
    else:
        train_pairs, validation_pairs = split_training_pairs(train_pairs, validation_ratio, seed)

    splits = {"train": train_pairs, "val": validation_pairs}
    if test_pairs:
        splits["test"] = test_pairs
    return splits


def preprocess_dataset(args):
    input_dir = args.raw_data_dir.resolve()
    output_dir = args.data_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("raw_data_dir and data_dir must be different")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_index_file = (
        args.split_index_file.resolve() if args.split_index_file is not None else None
    )
    splits = prepare_splits(
        input_dir,
        args.validation_ratio,
        args.seed,
        split_index_file,
    )
    manifest = {split: [pair[0] for pair in pairs] for split, pairs in splits.items()}
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overlap_by_split = {
        "train": args.overlap,
        "val": args.overlap_val,
        "test": args.overlap_test,
    }
    for split, pairs in splits.items():
        if not pairs:
            print(f"{split}: no image/label pairs")
            continue
        worker = lambda pair: process_pair(
            pair,
            split,
            output_dir,
            args.patch_size,
            overlap_by_split[split],
            args.output_format,
            args.background_value,
            args.min_foreground_pixels,
            args.keep_empty,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(worker, pairs))
        saved = sum(result[1] for result in results)
        total = sum(result[2] for result in results)
        print(f"{split}: saved {saved}/{total} patches from {len(pairs)} rasters")


def build_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--raw_data_dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0, help="Training overlap in pixels.")
    parser.add_argument("--overlap_val", type=int, default=0)
    parser.add_argument("--overlap_test", type=int, default=0)
    parser.add_argument("--validation_ratio", type=float, default=0.)
    parser.add_argument(
        "--split_index_file",
        type=Path,
        default="data_processing/data_splits/data_split.json",
        help=(
            "Optional JSON file containing explicit train/val/test identifier lists. "
            "When omitted, use the current grouped validation split logic."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--background_value", type=int, default=0)
    parser.add_argument("--min_foreground_pixels", type=int, default=1)
    parser.add_argument("--keep_empty", action="store_true", help="Keep pure-background patches.")
    parser.add_argument(
        "--output_format",
        choices=("auto", "png", "tif"),
        default="auto",
        help="auto: referenced input -> GeoTIFF, unreferenced input -> PNG.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    start = time.time()
    arguments = build_parser().parse_args()
    if arguments.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    for overlap in (arguments.overlap, arguments.overlap_val, arguments.overlap_test):
        if not 0 <= overlap < arguments.patch_size:
            raise ValueError("Every overlap must satisfy 0 <= overlap < patch_size")
    if arguments.workers < 1:
        raise ValueError("workers must be at least 1")
    if arguments.min_foreground_pixels < 0:
        raise ValueError("min_foreground_pixels cannot be negative")
    preprocess_dataset(arguments)
    print(f"Completed in {(time.time() - start) / 60:.2f} minutes")
