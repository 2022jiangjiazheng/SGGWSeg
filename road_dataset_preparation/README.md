# Road Dataset Preparation

This directory contains three standalone scripts for preparing road-segmentation
datasets. Run commands from this directory unless absolute paths are supplied.

## Requirements

Install Python 3 and the required packages:

```bash
pip install numpy pillow opencv-python scikit-learn tqdm
```

## 1. Preprocess the Massachusetts dataset

`preprocess_massachusetts_dataset.py` merges the official train, validation,
and test sets; removes large near-white or near-black border regions; filters
images by invalid-area ratio; and creates a reproducible train/test split.

Expected source layout:

```text
tiff/
├── train/          (*.tiff)
├── train_labels/   (*.tif)
├── val/            (*.tiff)
├── val_labels/     (*.tif)
├── test/           (*.tiff)
└── test_labels/    (*.tif)
```

Example:

```bash
python preprocess_massachusetts_dataset.py --source /path/to/tiff --target raw_data
```

Use `--overwrite` only when existing files in the target may be deleted and
regenerated. Run `python preprocess_massachusetts_dataset.py --help` for all
filtering, split, and worker options.

## 2. Tile the Massachusetts dataset

`split_massachusetts_dataset.py` reads the preprocessed `raw_data` directory.
Training images can use overlapping tiles; test images are center-padded and
cut into non-overlapping tiles. It also writes `tiles_manifest.csv`.

Expected source layout:

```text
raw_data/
├── images/train/
├── images/test/
├── zones/train/
└── zones/test/
```

Example:

```bash
python split_massachusetts_dataset.py --source raw_data --target data --tile-size 1024 --train-overlap 0
```

Run `python split_massachusetts_dataset.py --help` for all options. Use
`--overwrite` only when the existing tiled output may be replaced.

## 3. Split the DeepGlobe dataset

`split_deepglobe_dataset.py` randomly assigns paired DeepGlobe files to train
and test sets using a fixed seed of 42. By default it reads `raw_data`, writes
to `data`, and selects 4,696 training pairs and 1,530 test pairs.

Expected source file names:

```text
raw_data/
├── 104_sat.jpg
├── 104_mask.png
└── ...
```

Edit `SOURCE`, `TARGET`, or the requested counts at the bottom of the script if
needed, then run:

```bash
python split_deepglobe_dataset.py
```

The output uses the common layout `images/{train,test}` and
`zones/{train,test}`. The script copies source files and does not delete them.

## Recommended order

For Massachusetts, run the preprocessing script first and the tiling script
second. The DeepGlobe splitter is independent and should be run only for the
DeepGlobe dataset.
