# Project repository: https://github.com/2022jiangjiazheng
"""Small GDAL helpers shared by conversion and patch extraction scripts."""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from osgeo import gdal, gdal_array


RASTER_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_rasters(directory: Path):
    """Return supported raster files in a deterministic order."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Raster directory does not exist: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in RASTER_EXTENSIONS
    )


def sample_id(path: Path) -> str:
    """Return the shared image/label identifier (`_mask` is ignored)."""
    stem = path.stem
    return stem[:-5] if stem.lower().endswith("_mask") else stem


def pair_rasters(image_dir: Path, label_dir: Path):
    """Pair image and label rasters by filename, ignoring a label `_mask` suffix."""
    image_paths = list_rasters(image_dir)
    label_paths = list_rasters(label_dir)
    lossy_labels = [path for path in label_paths if path.suffix.lower() in {".jpg", ".jpeg"}]
    if lossy_labels:
        raise ValueError(f"JPEG is not allowed for segmentation labels: {lossy_labels}")
    images = {sample_id(path): path for path in image_paths}
    labels = {sample_id(path): path for path in label_paths}
    if len(images) != len(image_paths) or len(labels) != len(label_paths):
        raise ValueError(
            "Duplicate sample identifiers found. Do not keep multiple raster formats "
            "with the same filename stem in one directory."
        )
    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        raise ValueError(
            f"Unpaired rasters. Missing labels={missing_labels}; missing images={missing_images}"
        )
    return [(key, images[key], labels[key]) for key in sorted(images)]


def read_raster(path: Path):
    """Read a raster as GDAL band-first data: (H, W) or (C, H, W)."""
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"GDAL cannot open raster: {path}")

    array = dataset.ReadAsArray()
    if array is None:
        raise RuntimeError(f"GDAL cannot read raster pixels: {path}")

    geotransform = dataset.GetGeoTransform(can_return_null=True)
    projection = dataset.GetProjection() or ""
    dataset = None
    return array, geotransform, projection


def spatial_shape(array: np.ndarray) -> Tuple[int, int]:
    if array.ndim == 2:
        return array.shape
    if array.ndim == 3:
        return array.shape[-2:]
    raise ValueError(f"Expected a 2D or band-first 3D raster, got {array.shape}")


def pad_bottom_right(array: np.ndarray, bottom: int, right: int) -> np.ndarray:
    """Pad spatial dimensions with zeros while preserving band order."""
    if array.ndim == 2:
        padding = ((0, bottom), (0, right))
    elif array.ndim == 3:
        padding = ((0, 0), (0, bottom), (0, right))
    else:
        raise ValueError(f"Expected a 2D or band-first 3D raster, got {array.shape}")
    return np.pad(array, padding, mode="constant", constant_values=0)


def crop_patch(array: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    if array.ndim == 2:
        return array[row:row + size, col:col + size]
    return array[:, row:row + size, col:col + size]


def offset_geotransform(geotransform, row: int, col: int):
    """Calculate the correct affine transform for a pixel-window patch."""
    if geotransform is None:
        return None
    return (
        geotransform[0] + col * geotransform[1] + row * geotransform[2],
        geotransform[1],
        geotransform[2],
        geotransform[3] + col * geotransform[4] + row * geotransform[5],
        geotransform[4],
        geotransform[5],
    )


def _as_bands(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected a 2D or band-first 3D raster, got {array.shape}")


def _gdal_data_type(dtype: np.dtype) -> int:
    data_type = gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(dtype).type)
    if data_type in (None, gdal.GDT_Unknown):
        raise TypeError(f"Unsupported GDAL dtype: {dtype}")
    return data_type


def _memory_dataset(array: np.ndarray):
    bands = _as_bands(np.ascontiguousarray(array))
    count, height, width = bands.shape
    dataset = gdal.GetDriverByName("MEM").Create(
        "", width, height, count, _gdal_data_type(bands.dtype)
    )
    for band_index, band in enumerate(bands, start=1):
        dataset.GetRasterBand(band_index).WriteArray(band)
    return dataset


def write_png(array: np.ndarray, path: Path) -> None:
    """Write pixels only. Projection, affine transform and source metadata are omitted."""
    bands = _as_bands(array)
    if bands.shape[0] not in (1, 2, 3, 4):
        raise ValueError("PNG supports 1-4 bands; use TIFF for additional bands.")
    if bands.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise TypeError(f"PNG requires uint8 or uint16 data, got {bands.dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)
    memory = _memory_dataset(bands)
    output = gdal.GetDriverByName("PNG").CreateCopy(
        str(path), memory, strict=1, options=["ZLEVEL=6"]
    )
    if output is None:
        raise RuntimeError(f"Failed to write PNG: {path}")
    output = None
    memory = None

    # Do not leave GDAL PAM sidecars that could contain copied metadata.
    auxiliary_file = Path(f"{path}.aux.xml")
    if auxiliary_file.exists():
        auxiliary_file.unlink()


def write_geotiff(
    array: np.ndarray,
    path: Path,
    geotransform=None,
    projection: str = "",
) -> None:
    """Write a losslessly compressed TIFF, preserving georeferencing when supplied."""
    bands = _as_bands(np.ascontiguousarray(array))
    count, height, width = bands.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    output = gdal.GetDriverByName("GTiff").Create(
        str(path),
        width,
        height,
        count,
        _gdal_data_type(bands.dtype),
        options=["TILED=YES", "COMPRESS=DEFLATE"],
    )
    if output is None:
        raise RuntimeError(f"Failed to write GeoTIFF: {path}")
    if geotransform is not None:
        output.SetGeoTransform(geotransform)
    if projection:
        output.SetProjection(projection)
    for band_index, band in enumerate(bands, start=1):
        output.GetRasterBand(band_index).WriteArray(band)
    output = None
