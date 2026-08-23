<h1 align="center">SGGWSeg</h1>

<h3 align="center">
  <em>Tracing the Great Wall from Satellite Imagery: A Skeleton-Guided Dual-Task Framework for Linear Remains Extraction</em>
</h3>

<p align="center">
  <a href="#introduction">Introduction</a> ·
  <a href="#news">News</a> ·
  <a href="#graphical-abstract">Graphical Abstract</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#dataset">Dataset</a> ·
  <a href="#downloads">Downloads</a> ·
  <a href="#citation">Citation</a>
</p>

<a id="introduction"></a>

## Introduction

This is the official implementation of **SGGWSeg**, a skeleton-guided dual-task collaborative framework for extracting linear cultural relics, such as the Great Wall, from high-resolution remote-sensing imagery. SGGWSeg jointly learns semantic regions and skeleton structures, improving the continuity and integrity of narrow, fragmented remains in complex environments.

<a id="news"></a>

## 📌 News

- **[2026/08]** The SGGWSeg code is publicly available.
- **[2026/07]** SGGWSeg won the **Gold Award** in the **AI for Science** track of the [3rd Global Digital Intelligence Education Innovation Competition](https://diidea.pku.edu.cn/competition2026/).
- **[2026/06]** Our paper was prepared for submission to *ISPRS Journal of Photogrammetry and Remote Sensing*.
- **[Coming soon]** The GansuGW dataset and compatible environment wheels will be released through Baidu Netdisk.

<a id="graphical-abstract"></a>

## 🖼️ Graphical Abstract

<p align="center">
  <img src="fig/SGGWSeg_GA.png" alt="SGGWSeg graphical abstract" width="90%">
</p>

The graphical abstract summarizes satellite observation, construction of the multi-temporal and multi-resolution GansuGW dataset, the skeleton-guided dual-task network, and its improvements in segmentation continuity and structural quality.

<a id="methodology"></a>

## 🏗️ Methodology

<p align="center">
  <img src="fig/Framework.png" alt="SGGWSeg framework architecture" width="95%">
</p>

SGGWSeg uses an ImageNet-1K-pretrained SegMAN-S encoder and two collaborative branches for relic semantic segmentation and relic skeleton extraction. Multi-level features are aggregated by the multi-scale context aggregation module, while skeleton-guided enhancement injects linear structural cues into the semantic branch. The two tasks are optimized jointly with segmentation and skeleton supervision.

<a id="installation"></a>

### Environment Installation

The tested environment is Linux x86-64 with Python 3.9.18, PyTorch 2.1.0, and CUDA 11.8. More details are available in [README_env.md](README_env.md).

#### 1. Create the environment

```bash
conda create -n sggwseg python=3.9.18 -y
conda activate sggwseg
python -m pip install --upgrade pip setuptools wheel
```

#### 2. Install PyTorch

```bash
python -m pip install \
  torch==2.1.0+cu118 \
  torchvision==0.16.0+cu118 \
  torchaudio==2.1.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

#### 4. Install the compatible wheels

Install the provided wheels with `--no-deps` to prevent pip from replacing the matching PyTorch build.

```bash
python -m pip install --no-deps \
  wheels/GDAL-3.4.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

python -m pip install --no-deps \
  wheels/natten-0.17.3+torch210cu118-cp39-cp39-linux_x86_64.whl

python -m pip install --no-deps \
  wheels/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

python -m pip install --no-deps mmcv-lite==2.1.0
```

#### 5. Verify the environment

```bash
python - <<'PY'
import torch
import natten
import mamba_ssm
from osgeo import gdal
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from models.sggwseg_segmentation import SGGWSeg

assert torch.__version__.startswith("2.1.0")
assert torch.version.cuda == "11.8"
assert torch.cuda.is_available()
assert torch._C._GLIBCXX_USE_CXX11_ABI is False

print("PyTorch:", torch.__version__)
print("NATTEN:", natten.__version__)
print("Mamba-SSM:", mamba_ssm.__version__)
print("GDAL:", gdal.VersionInfo())
print("Environment check passed.")
PY

python -m pip check
```

Place the ImageNet-1K-pretrained SegMAN-S weights at:

```text
pretrained/SegMAN_Encoder_s.pth.tar
```

### Code Entry Points

| Stage | File | Purpose |
|---|---|---|
| Data preparation | [`data_processing/data_preprocessing.py`](data_processing/data_preprocessing.py) | Split source rasters and extract paired image/label patches. |
| Training | [`train.py`](train.py) | Train SGGWSeg and save checkpoints and TensorBoard logs. |
| Validation and testing | [`validate_or_test.py`](validate_or_test.py) | Perform full-image inference, reconstruct predictions, and generate evaluation reports. |

Run the corresponding file with `--help` to view its parameters.

<a id="dataset"></a>

## 📊 GansuGW Dataset

<p align="center">
  <img src="fig/Studyarea.png" alt="GansuGW study area and dataset distribution" width="95%">
</p>

We introduce **GansuGW**, the first large-scale, multi-temporal, and multi-resolution benchmark dataset specifically curated for Great Wall remains.

- **Total samples:** 15,914 high-quality annotated image patches.
- **Spatial resolution:** 0.25 m to 1 m.
- **Temporal span:** 2003 to 2025.
- **Annotations:** three-band optical images with binary Great Wall masks.

GansuGW covers diverse Great Wall sections, acquisition dates, spatial resolutions, preservation conditions, historical periods, and surrounding environments.

<a id="downloads"></a>

## Downloads

| Resource | Contents | Download |
|---|---|---|
| GansuGW dataset | RGB images, binary labels, and dataset splits | **Baidu Netdisk: coming soon** |
| Environment wheels | Linux wheels for GDAL 3.4.3, NATTEN 0.17.3, and Mamba-SSM 2.2.4 | **Baidu Netdisk: coming soon** |
| SegMAN-S pretrained encoder | ImageNet-1K pretrained backbone | [SegMAN](https://github.com/yunxiangfu2001/SegMAN) |

The Baidu Netdisk links and extraction codes will be added after the archives are uploaded.

<a id="citation"></a>

## Citation

The paper and BibTeX entry will be added after publication. If this repository or the GansuGW dataset is useful for your research, please cite the forthcoming SGGWSeg paper.

## Acknowledgements

SGGWSeg uses the [SegMAN](https://github.com/yunxiangfu2001/SegMAN) encoder and builds on PyTorch, PyTorch Lightning, Mamba-SSM, NATTEN, and GDAL. The data-processing and full-image evaluation workflow also benefited from [Calving Fronts and Where to Find Them](https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them). We sincerely thank the authors and maintainers of these projects.
