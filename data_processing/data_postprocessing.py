# Project repository: https://github.com/2022jiangjiazheng
"""Efficient utilities for reconstructing full-image segmentation predictions.

This module only handles patch geometry and probability blending. Model inference
and metric calculation live in ``validate_or_test.py``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PatchInfo:
    """Location metadata encoded by ``data_preprocessing.py`` in a patch name."""

    image_id: str
    padded_bottom: int
    padded_right: int
    index: int
    row: int
    col: int


def parse_patch_name(path: Path) -> PatchInfo:
    """Parse ``<id>__<bottom>_<right>_<index>_<row>_<col>.<ext>``."""
    try:
        image_id, encoded = path.stem.rsplit("__", 1)
        bottom, right, index, row, col = map(int, encoded.rsplit("_", 4))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"Invalid preprocessed patch name: {path.name}") from error
    return PatchInfo(image_id, bottom, right, index, row, col)


def sliding_positions(length: int, patch_size: int, overlap: int) -> Tuple[list, int]:
    """Return regular window origins and the trailing padding size."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0 <= overlap < patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")

    stride = patch_size - overlap
    if length <= patch_size:
        return [0], patch_size - length
    remainder = (length - patch_size) % stride
    padding = 0 if remainder == 0 else stride - remainder
    return list(range(0, length + padding - patch_size + 1, stride)), padding


def blending_weight(patch_size: int, mode: str = "uniform") -> np.ndarray:
    """Create a non-zero 2-D weight map for averaging overlapping predictions."""
    if mode == "uniform":
        return np.ones((patch_size, patch_size), dtype=np.float32)
    if mode != "gaussian":
        raise ValueError(f"Unknown blending mode: {mode}")

    coordinates = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
    gaussian_1d = np.exp(-0.5 * (coordinates / 0.5) ** 2)
    weight = np.outer(gaussian_1d, gaussian_1d)
    return np.maximum(weight / weight.max(), 1e-3).astype(np.float32)


class ProbabilityAccumulator:
    """Vectorized probability averaging for one padded complete image."""

    def __init__(self, shape: Sequence[int], patch_size: int, blend_mode: str):
        height, width = map(int, shape)
        self.probability_sum = np.zeros((height, width), dtype=np.float32)
        self.weight_sum = np.zeros((height, width), dtype=np.float32)
        self.weight = blending_weight(patch_size, blend_mode)

    def add(self, probability: np.ndarray, row: int, col: int) -> None:
        """Add one foreground-probability patch using array slices, not pixel loops."""
        probability = np.asarray(probability, dtype=np.float32)
        patch_height, patch_width = probability.shape
        row_end = min(row + patch_height, self.probability_sum.shape[0])
        col_end = min(col + patch_width, self.probability_sum.shape[1])
        valid_height, valid_width = row_end - row, col_end - col
        if valid_height <= 0 or valid_width <= 0:
            raise ValueError(f"Patch origin ({row}, {col}) is outside the output image")

        weight = self.weight[:valid_height, :valid_width]
        output_slice = np.s_[row:row_end, col:col_end]
        self.probability_sum[output_slice] += probability[:valid_height, :valid_width] * weight
        self.weight_sum[output_slice] += weight

    def finalize(self, original_shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Return cropped probability and a boolean map of pixels covered by patches."""
        height, width = map(int, original_shape)
        weights = self.weight_sum[:height, :width]
        probability = np.divide(
            self.probability_sum[:height, :width],
            weights,
            out=np.zeros((height, width), dtype=np.float32),
            where=weights > 0,
        )
        return probability, weights > 0


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a binary mask as a lossless 0/255 PNG without georeferencing."""
    # Keep GDAL optional when only the reconstruction helpers are imported.
    from utils.raster import write_png

    write_png(np.asarray(mask, dtype=np.uint8) * 255, output_path)
