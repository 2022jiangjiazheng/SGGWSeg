# Project repository: https://github.com/2022jiangjiazheng
"""Evaluate SGGWSeg after reconstructing predictions at original-image scale.

Two inference sources are supported:

``patches``
    Use image patches already produced by ``data_preprocessing.py``.
``sliding``
    Read each original image and run sliding-window inference directly. This mode
    supports overlap and is recommended for authoritative full-image metrics.
"""

import json
import math
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label as connected_components
from skimage.morphology import skeletonize
from torchvision.transforms import functional as TF

from data_processing.data_postprocessing import (
    PatchInfo,
    ProbabilityAccumulator,
    parse_patch_name,
    save_binary_mask,
    sliding_positions,
)
from models.sggwseg_segmentation import SGGWSeg
from utils.raster import (
    crop_patch,
    list_rasters,
    pad_bottom_right,
    read_raster,
    sample_id,
    spatial_shape,
)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return NaN when a metric is undefined instead of raising a warning."""
    return float(numerator / denominator) if denominator else math.nan


def binary_label(array: np.ndarray) -> np.ndarray:
    """Convert 0/1 or 0/255 labels to a two-class uint8 mask."""
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"A label must contain one band, got shape {array.shape}")
        array = array[0]
    return (array > 0).astype(np.uint8)


def image_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert GDAL band-first image data using the same scaling as training."""
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"SGGWSeg expects a 3-band image, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise TypeError(
            f"Expected an 8-bit image, but received {array.dtype}. "
            "Convert it to uint8 explicitly before evaluation."
        )
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().div_(255.0)

    tensor = TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return tensor


def autocast_context(device: torch.device, precision: str):
    if precision == "32":
        return nullcontext()
    if device.type != "cuda":
        warnings.warn("Mixed precision requires CUDA; falling back to float32.")
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def predict_probabilities(
    model: torch.nn.Module,
    arrays: Sequence[np.ndarray],
    device: torch.device,
    precision: str,
) -> np.ndarray:
    """Return foreground softmax probabilities for one batch."""
    batch = torch.stack([image_tensor(array) for array in arrays]).to(
        device, non_blocking=True
    )
    with autocast_context(device, precision):
        outputs = model(batch)
        mask_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        probabilities = torch.softmax(mask_logits, dim=1)[:, 1]
    return probabilities.float().cpu().numpy()


def batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def raster_index(root: Path, kind: str) -> Dict[str, Path]:
    """Index original rasters across train/val/test directories by sample ID."""
    paths: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        directory = root / kind / split
        if not directory.is_dir():
            continue
        for path in list_rasters(directory):
            identifier = sample_id(path)
            if identifier in paths:
                raise ValueError(
                    f"Duplicate original sample '{identifier}': {paths[identifier]} and {path}"
                )
            paths[identifier] = path
    return paths


def selected_ids(
    split: str,
    raw_data_dir: Path,
    data_dir: Path,
    selection: str,
) -> List[str]:
    """Select full images from the preprocessing manifest or physical raw folder."""
    manifest_path = data_dir / "split_manifest.json"
    manifest_ids: List[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_ids = list(manifest.get(split, []))

    if selection == "manifest" or (selection == "auto" and manifest_ids):
        if not manifest_ids:
            raise ValueError(f"No '{split}' samples found in {manifest_path}")
        return sorted(manifest_ids)

    folder = raw_data_dir / "images" / split
    return sorted(sample_id(path) for path in list_rasters(folder))


def group_patch_files(patch_dir: Path) -> Dict[str, List[Tuple[Path, PatchInfo]]]:
    groups: Dict[str, List[Tuple[Path, PatchInfo]]] = {}
    for path in list_rasters(patch_dir):
        info = parse_patch_name(path)
        groups.setdefault(info.image_id, []).append((path, info))
    if not groups:
        raise FileNotFoundError(f"No preprocessed image patches found in {patch_dir}")
    for group in groups.values():
        group.sort(key=lambda item: item[1].index)
    return dict(sorted(groups.items()))


def infer_from_existing_patches(
    model: torch.nn.Module,
    patch_files: Sequence[Tuple[Path, PatchInfo]],
    original_shape: Tuple[int, int],
    batch_size: int,
    workers: int,
    blend_mode: str,
    device: torch.device,
    precision: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict and reconstruct one image from existing preprocessed patches."""
    first_array, _, _ = read_raster(patch_files[0][0])
    patch_size = spatial_shape(first_array)[0]
    info = patch_files[0][1]
    padded_shape = (
        original_shape[0] + info.padded_bottom,
        original_shape[1] + info.padded_right,
    )
    accumulator = ProbabilityAccumulator(padded_shape, patch_size, blend_mode)

    def load(path_and_info):
        path, patch_info = path_and_info
        array, _, _ = read_raster(path)
        if spatial_shape(array) != (patch_size, patch_size):
            raise ValueError(f"Inconsistent patch shape in {path}")
        return array, patch_info

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch_items in batched(patch_files, batch_size):
            loaded = list(pool.map(load, batch_items))
            probabilities = predict_probabilities(
                model, [item[0] for item in loaded], device, precision
            )
            for probability, (_, patch_info) in zip(probabilities, loaded):
                accumulator.add(probability, patch_info.row, patch_info.col)
    return accumulator.finalize(original_shape)


def infer_with_sliding_window(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: int,
    overlap: int,
    batch_size: int,
    blend_mode: str,
    device: torch.device,
    precision: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict a complete raster with regular, optionally overlapping windows."""
    height, width = spatial_shape(image)
    rows, bottom = sliding_positions(height, patch_size, overlap)
    cols, right = sliding_positions(width, patch_size, overlap)
    padded = pad_bottom_right(image, bottom, right)
    coordinates = [(row, col) for row in rows for col in cols]
    accumulator = ProbabilityAccumulator(
        (height + bottom, width + right), patch_size, blend_mode
    )

    for coordinate_batch in batched(coordinates, batch_size):
        arrays = [crop_patch(padded, row, col, patch_size) for row, col in coordinate_batch]
        probabilities = predict_probabilities(model, arrays, device, precision)
        for probability, (row, col) in zip(probabilities, coordinate_batch):
            accumulator.add(probability, row, col)
    return accumulator.finalize((height, width))


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> Dict[str, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    return {
        "tp": int(np.count_nonzero(prediction & target)),
        "tn": int(np.count_nonzero(~prediction & ~target)),
        "fp": int(np.count_nonzero(prediction & ~target)),
        "fn": int(np.count_nonzero(~prediction & target)),
    }


def relaxed_metrics(
    prediction: np.ndarray, target: np.ndarray, relax_pixels: int, counts: Dict[str, int]
) -> Dict[str, float]:
    prediction_bool = prediction.astype(bool)
    target_bool = target.astype(bool)
    kernel_size = 2 * relax_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    relaxed_target = cv2.dilate(target, kernel, iterations=1).astype(bool)
    relaxed_prediction = cv2.dilate(prediction, kernel, iterations=1).astype(bool)

    # Preserve the original relaxed-metric definitions exactly.
    relaxed_target_intersection = np.count_nonzero(prediction_bool & relaxed_target)
    relaxed_prediction_intersection = np.count_nonzero(relaxed_prediction & target_bool)
    strict_union = counts["tp"] + counts["fp"] + counts["fn"]
    precision = safe_divide(relaxed_target_intersection, np.count_nonzero(prediction_bool))
    recall = safe_divide(relaxed_prediction_intersection, np.count_nonzero(target_bool))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    iou = safe_divide(
        relaxed_target_intersection
        + relaxed_prediction_intersection
        - counts["tp"],
        strict_union,
    )
    return {
        "relaxed_precision": precision,
        "relaxed_recall": recall,
        "relaxed_f1": f1,
        "relaxed_greatwall_iou": iou,
    }


def connectivity_metric(prediction: np.ndarray, target: np.ndarray) -> float:
    _, predicted_count = connected_components(
        prediction, connectivity=2, background=0, return_num=True
    )
    _, target_count = connected_components(
        target, connectivity=2, background=0, return_num=True
    )
    if predicted_count == target_count == 0:
        return 1.0
    return safe_divide(min(predicted_count, target_count), max(predicted_count, target_count))


def centerline_metrics(
    prediction: np.ndarray, target: np.ndarray, relax_pixels: int
) -> Dict[str, float]:
    predicted_skeleton = skeletonize(prediction > 0)
    target_skeleton = skeletonize(target > 0)
    kernel_size = 2 * relax_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    relaxed_target = cv2.dilate(target_skeleton.astype(np.uint8), kernel).astype(bool)
    relaxed_prediction = cv2.dilate(predicted_skeleton.astype(np.uint8), kernel).astype(bool)
    predicted_in_target = np.count_nonzero(predicted_skeleton & relaxed_target)
    target_in_prediction = np.count_nonzero(target_skeleton & relaxed_prediction)
    target_missed = np.count_nonzero(target_skeleton & ~relaxed_prediction)
    return {
        # Corr is precision-like; Comp is recall-like.
        "correctness": safe_divide(predicted_in_target, np.count_nonzero(predicted_skeleton)),
        "completeness": safe_divide(target_in_prediction, np.count_nonzero(target_skeleton)),
        "quality": safe_divide(
            predicted_in_target,
            np.count_nonzero(predicted_skeleton) + target_missed,
        ),
    }


def image_metrics(
    image_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    coverage: np.ndarray,
    relax_pixels: int,
) -> Dict[str, float]:
    counts = confusion_counts(prediction, target)
    foreground_iou = safe_divide(
        counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]
    )
    background_iou = safe_divide(
        counts["tn"], counts["tn"] + counts["fp"] + counts["fn"]
    )
    precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    result = {
        "image_id": image_id,
        "background_iou": background_iou,
        "greatwall_iou": foreground_iou,
        "miou": float(np.nanmean([background_iou, foreground_iou])),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage_ratio": float(np.mean(coverage)),
        **counts,
    }
    result.update(relaxed_metrics(prediction, target, relax_pixels, counts))
    result["connectivity"] = connectivity_metric(prediction, target)
    result.update(centerline_metrics(prediction, target, relax_pixels))
    return result


def metadata_from_id(image_id: str) -> Dict[str, object]:
    """Extract Great Wall number, acquisition date, season, and resolution."""
    parts = image_id.rsplit("_", 2)
    if len(parts) != 3:
        return {"greatwall_id": image_id, "date": None, "season": None, "resolution": None}
    greatwall_id, date, resolution_text = parts
    try:
        month = int(date.split("-")[1])
        resolution = float(resolution_text)
    except (ValueError, IndexError):
        return {"greatwall_id": greatwall_id, "date": date, "season": None, "resolution": None}
    return {
        "greatwall_id": greatwall_id,
        "date": date,
        "season": "summer" if 4 <= month <= 9 else "winter",
        "resolution": resolution,
    }


def resolve_relax_pixels(
    image_id: str,
    relax_meters: Optional[float],
    relax_pixels: Optional[int],
) -> int:
    """Resolve the per-image tolerance from either meters or fixed pixels."""
    if relax_pixels is not None:
        return relax_pixels

    resolution_cm = metadata_from_id(image_id)["resolution"]
    if resolution_cm is None or resolution_cm <= 0:
        raise ValueError(
            f"Cannot derive a metric tolerance for '{image_id}'. "
            "The filename must end with resolution in cm, or use --relax_pixels."
        )
    return max(1, round(relax_meters / (resolution_cm / 100.0)))


def aggregate_summary(frame: pd.DataFrame) -> Dict[str, object]:
    metric_columns = [
        "background_iou", "greatwall_iou", "miou", "precision", "recall", "f1",
        "relaxed_precision", "relaxed_recall", "relaxed_f1",
        "relaxed_greatwall_iou", "connectivity", "completeness", "correctness",
        "quality", "coverage_ratio",
    ]
    macro = {f"macro_{column}": float(frame[column].mean()) for column in metric_columns}
    totals = frame[["tp", "tn", "fp", "fn"]].sum()
    foreground_iou = safe_divide(totals.tp, totals.tp + totals.fp + totals.fn)
    background_iou = safe_divide(totals.tn, totals.tn + totals.fp + totals.fn)
    micro = {
        "micro_background_iou": background_iou,
        "micro_greatwall_iou": foreground_iou,
        "micro_miou": float(np.nanmean([background_iou, foreground_iou])),
        "micro_precision": safe_divide(totals.tp, totals.tp + totals.fp),
        "micro_recall": safe_divide(totals.tp, totals.tp + totals.fn),
    }
    micro["micro_f1"] = safe_divide(
        2 * micro["micro_precision"] * micro["micro_recall"],
        micro["micro_precision"] + micro["micro_recall"],
    )
    return {"number_of_images": len(frame), **macro, **micro}


def save_reports(frame: pd.DataFrame, summary: Dict[str, object], output_dir: Path) -> None:
    frame.to_csv(output_dir / "per_image_metrics.csv", index=False, encoding="utf-8-sig")
    clean_summary = {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in summary.items()
    }
    (output_dir / "summary_metrics.json").write_text(
        json.dumps(clean_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metric_columns = [column for column in frame.columns if column not in {
        "image_id", "greatwall_id", "date", "season", "resolution", "tp", "tn", "fp", "fn"
    }]
    with pd.ExcelWriter(output_dir / "evaluation_results.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Per_Image", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        for column, sheet_name in (
            ("greatwall_id", "By_GreatWall"),
            ("season", "By_Season"),
            ("resolution", "By_Resolution"),
        ):
            frame.groupby(column, dropna=True)[metric_columns].mean().to_excel(
                writer, sheet_name=sheet_name
            )


def load_model(args, device: torch.device) -> SGGWSeg:
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint}\n"
            "Update the default in build_parser() or pass --checkpoint."
        )

    load_options = {"checkpoint_path": str(checkpoint), "map_location": "cpu"}
    if args.hparams_file is not None:
        load_options["hparams_file"] = str(args.hparams_file)
    model = SGGWSeg.load_from_checkpoint(**load_options)
    model.eval().to(device)
    return model


def evaluate(args) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = load_model(args, device)

    output_dir = args.output_dir
    predictions_dir = output_dir / "complete_images"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    image_paths = raster_index(args.raw_data_dir, "images")
    label_paths = raster_index(args.raw_data_dir, "zones")

    if args.source == "patches":
        patch_groups = group_patch_files(args.data_dir / "images" / args.split)
        identifiers = list(patch_groups)
    else:
        identifiers = selected_ids(
            args.split, args.raw_data_dir, args.data_dir, args.sample_selection
        )

    missing = [identifier for identifier in identifiers if identifier not in image_paths or identifier not in label_paths]
    if missing:
        raise FileNotFoundError(f"Original image or label missing for: {missing}")

    records = []
    print(f"Device: {device}; source: {args.source}; images: {len(identifiers)}")
    for index, identifier in enumerate(identifiers, start=1):
        image, _, _ = read_raster(image_paths[identifier])
        target_array, _, _ = read_raster(label_paths[identifier])
        target = binary_label(target_array)
        if spatial_shape(image) != target.shape:
            raise ValueError(
                f"Image/label size mismatch for {identifier}: {spatial_shape(image)} vs {target.shape}"
            )

        if args.source == "patches":
            probability, coverage = infer_from_existing_patches(
                model, patch_groups[identifier], target.shape, args.batch_size,
                args.workers, args.blend, device, args.precision,
            )
        else:
            probability, coverage = infer_with_sliding_window(
                model, image, args.patch_size, args.overlap, args.batch_size,
                args.blend, device, args.precision,
            )

        coverage_ratio = float(np.mean(coverage))
        if coverage_ratio < 1.0 and args.strict_coverage:
            raise RuntimeError(
                f"{identifier}: existing patches cover only {coverage_ratio:.2%} "
                "of the original image."
            )

        # The two averaged class probabilities sum to one, so argmax is the
        # standard binary decision rule after overlap blending.
        class_probabilities = np.stack((1.0 - probability, probability), axis=0)
        prediction = np.argmax(class_probabilities, axis=0).astype(np.uint8)
        save_binary_mask(prediction, predictions_dir / f"{identifier}.png")
        relax_pixels = resolve_relax_pixels(
            identifier, args.relax_meters, args.relax_pixels
        )
        record = image_metrics(identifier, prediction, target, coverage, relax_pixels)
        record["relax_pixels"] = relax_pixels
        record.update(metadata_from_id(identifier))
        records.append(record)
        print(
            f"[{index}/{len(identifiers)}] {identifier}: "
            f"IoU_bg={record['background_iou']:.4f}, "
            f"IoU_gw={record['greatwall_iou']:.4f}, mIoU={record['miou']:.4f}"
        )

    frame = pd.DataFrame(records)
    summary = aggregate_summary(frame)
    summary.update({
        "split": args.split,
        "source": args.source,
        "patch_size": args.patch_size,
        "overlap": args.overlap if args.source == "sliding" else None,
        "relax_meters": args.relax_meters,
        "relax_pixels": args.relax_pixels,
    })
    save_reports(frame, summary, output_dir)
    print(
        "Completed: "
        f"macro mIoU={summary['macro_miou']:.4f}, "
        f"micro mIoU={summary['micro_miou']:.4f}. Results: {output_dir}"
    )


def build_parser() -> ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=project_dir / "checkpoints" / "run_6" / "-epoch=88-avg_metric_validation=0.74.ckpt",
        help="Lightning checkpoint to evaluate.",
    )
    parser.add_argument(
        "--hparams_file", type=Path, default=None,
        help="Optional hparams.yaml for legacy checkpoints.",
    )
    parser.add_argument(
        "--project_dir", type=Path, default=project_dir,
        help="Project root used to resolve relative paths.",
    )
    parser.add_argument(
        "--raw_data_dir", type=Path, default=project_dir / "raw_data",
        help="Original rasters containing images/ and zones/.",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=project_dir / "data",
        help="Preprocessed patches and split manifest directory.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=project_dir / "evaluation_results" / "run_6",
        help="Output directory for predictions and reports.",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test"), default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--source", choices=("patches", "sliding"), default="patches",
        help="Use existing patches or sliding-window inference.",
    )
    parser.add_argument(
        "--sample_selection", choices=("auto", "manifest", "folder"),
        default="auto",
        help="Sample source; auto prefers split_manifest.json.",
    )
    parser.add_argument(
        "--patch_size", type=int, default=512,
        help="Square sliding-window size in pixels.",
    )
    parser.add_argument(
        "--overlap", type=int, default=0,
        help="Overlap between adjacent sliding windows.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of patches per inference batch.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel patch readers in patches mode.",
    )
    parser.add_argument(
        "--blend", choices=("uniform", "gaussian"), default="uniform",
        help="Weighting method for overlapping predictions.",
    )
    relax_group = parser.add_mutually_exclusive_group()
    relax_group.add_argument(
        "--relax_meters", type=float, default=2,
        help="Ground-distance tolerance converted per image; defaults to 2 m when neither tolerance is set.",
    )
    relax_group.add_argument(
        "--relax_pixels", type=int, default=None,
        help="Fixed pixel tolerance for every image.",
    )
    parser.add_argument(
        "--precision", choices=("32", "16-mixed", "bf16-mixed"),
        default="32",
        help="Inference precision.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Inference device, such as auto, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--strict_coverage", action="store_true", default=False,
        help="Fail when existing patches do not cover the full image.",
    )
    return parser


def validate_arguments(args) -> None:
    for attribute in ("project_dir", "raw_data_dir", "data_dir", "output_dir"):
        path = getattr(args, attribute)
        if not path.is_absolute():
            setattr(args, attribute, (args.project_dir / path).resolve())
    if args.checkpoint is not None and not args.checkpoint.is_absolute():
        args.checkpoint = (args.project_dir / args.checkpoint).resolve()
    if args.hparams_file is not None and not args.hparams_file.is_absolute():
        args.hparams_file = (args.project_dir / args.hparams_file).resolve()
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("batch_size and workers must be at least 1")
    if not 0 <= args.overlap < args.patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
    if args.relax_meters is None and args.relax_pixels is None:
        args.relax_meters = 2.0
    if args.relax_meters is not None and args.relax_meters <= 0:
        raise ValueError("relax_meters must be positive")
    if args.relax_pixels is not None and args.relax_pixels < 0:
        raise ValueError("relax_pixels cannot be negative")


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    evaluate(arguments)
