# 🌊 Orange Basin 3D Seismic Volume Reconstruction

> **Comparative Analysis of Deep Neural Network Architectures for 3D Seismic Volume Reconstruction**
> UNet3D · ResAttUNet3D · Vision Transformer (ViT3D)

---

## Overview

This project benchmarks three deep learning architectures on the task of reconstructing augmented 3D seismic volumes back to their clean originals, using the **Orange Basin** dataset. Reconstruction quality is assessed across five metrics: **PSNR**, **MSE**, **RMSE**, **SSIM**, and **MS-SSIM**.

| Model | Architecture | Highlights |
|---|---|---|
| `UNet3D` | Encoder–Decoder (baseline) | Fast, proven, lightweight |
| `ResAttUNet3D` | Residual + Attention | Channel & spatial attention gates |
| `ViT3D` | Transformer patch autoencoder | Global context via self-attention |

---

## Dataset

The Orange Basin seismic volume is provided as a **SEG-Y / SEGY** or **NumPy (`.npy`)** file.

- **Target volume dimensions:** ~560 × 773 × 805 *(inline × crossline × time)*
- Volumes are normalised to **[−1, 1]** using min–max scaling prior to patching
- Augmentation applies **Gaussian noise**, **speckle noise**, and **3D Gaussian blur**

Place the dataset at:
```
orange_basin_reconstruction/src/data/Orange Basin_cropped 1
```

---

## Project Structure

```
orange_basin_reconstruction/
├── data/                          # Created during data preparation
│   ├── raw/                       # Raw volume saved as .npy
│   ├── splits/                    # Train / val / test split JSON
│   ├── augmented/                 # Saved augmented patch datasets (.npz)
│   └── samples/                   # Saved comparison images (aug vs orig)
│
├── runs/                          # Created during model training
│   └── <model>/<timestamp>/       # Checkpoints and logs
│
├── src/
│   ├── data/
│   │   ├── dataset.py             # Loaders, patch extraction, augmentation
│   │   └── Orange Basin_cropped 1 # ← Place dataset here
│   ├── metrics/
│   │   └── metrics.py             # MSE, RMSE, PSNR, SSIM3D, MS-SSIM3D
│   ├── models/
│   │   ├── unet3d.py
│   │   ├── res_attention_unet3d.py
│   │   └── vit3d.py
│   └── utils/
│       ├── seg_loader.py          # Optional SEG-Y loading via segyio
│       └── train_utils.py
│
├── scripts/
│   ├── prepare_data.py            # Load → split → augment → save
│   ├── train.py                   # Train a single model
│   ├── evaluate.py                # Evaluate metrics + save predictions
│   └── run_all.py                 # End-to-end pipeline (see warning below)
│
└── requirements.txt
```

---

## Installation

Install dependencies with pip. Match your PyTorch build to your CUDA version.

```bash
# CPU-only (for testing / data prep)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib tqdm segyio
```

> `segyio` is optional — only required for direct SEG-Y ingestion. If unavailable, convert to `.npy` first.

---

## Quick Start

### 1 — Prepare Data

**From SEG-Y:**
```bash
python scripts/prepare_data.py \
  --input path/to/orange_basin.segy \
  --format segy \
  --outdir data \
  --patch-size 64 64 64 \
  --train-patches 2000 \
  --val-patches 400
```

**From NumPy:**
```bash
python scripts/prepare_data.py \
  --input path/to/volume.npy \
  --format numpy \
  --outdir data \
  --patch-size 64 64 64 \
  --train-patches 2000 \
  --val-patches 400
```

> **Wits HPC cluster:** use `run_cluster_prepare_data.sh`

---

### 2 — Train a Model

```bash
# UNet3D (baseline)
python scripts/train.py \
  --model unet3d \
  --data-dir data \
  --batch-size 32 --epochs 50 --lr 1e-3 \
  --patch-size 64 64 64

# ResAttUNet3D
python scripts/train.py \
  --model resattunet3d \
  --data-dir data \
  --batch-size 4 --epochs 50 --lr 1e-3 \
  --patch-size 64 64 64

# ViT3D (requires smaller batch + lower LR)
python scripts/train.py \
  --model vit3d \
  --data-dir data \
  --batch-size 2 --epochs 50 --lr 1e-4 \
  --patch-size 64 64 64
```

> **Wits HPC cluster:**
> - UNet3D → `run_cluster_unet3d.sh`
> - ResAttUNet3D → `run_cluster_resattunet3d.sh`
> - ViT3D → `run_cluster_vit3d.sh`

---

### 3 — Evaluate

```bash
python scripts/evaluate.py \
  --model unet3d \
  --data-dir data \
  --ckpt runs/unet3d/<timestamp>/best.pt
```

> **Wits HPC cluster:** use `run_cluster_evaluate.sh`

---

## Running All Models End-to-End

> ⚠️ **Not recommended** unless you have substantial compute. Training all three models sequentially is resource-intensive.

```bash
python scripts/run_all.py \
  --input "src/data/Orange Basin_cropped 1" \
  --format auto \
  --outdir data \
  --patch-size 64 64 64 \
  --train-patches 2000 \
  --val-patches 400 \
  --epochs 50 \
  --batch-size 32
```

- ViT3D automatically uses a reduced batch size and learning rate
- Per-model metrics are written to `runs/<model>/metrics.json`
- A consolidated summary is saved to `runs/summary.json` and `runs/summary.csv`, including train, eval, and total timing

> **Wits HPC cluster:** use `run_cluster.sh`

---

## Metrics

All five metrics are computed on the validation and test sets after training:

| Metric | Measures |
|---|---|
| **MSE** | Mean squared error (pixel-level fidelity) |
| **RMSE** | Root mean squared error |
| **PSNR** | Peak signal-to-noise ratio (higher = better) |
| **SSIM** | Structural similarity (perceptual quality) |
| **MS-SSIM** | Multi-scale structural similarity |

SSIM and MS-SSIM are computed in 3D using a Gaussian window. For large volumes, patch-wise evaluation is used automatically.

---

## Notes & Troubleshooting

| Issue | Fix |
|---|---|
| `segyio` not installed | Convert SEG-Y to `.npy` first, or `pip install segyio` |
| Out-of-memory error | Reduce `--patch-size` and/or `--batch-size` |
| Windows path with spaces | Wrap paths in quotes: `"path/to/Orange Basin_cropped 1"` |
| ViT3D instability | Use `--lr 1e-4` and `--batch-size 2` or lower |

---

## Citation

If you use this codebase or dataset pipeline in your research, please cite accordingly and acknowledge the **Orange Basin** seismic dataset source.

---

*Built for comparative deep learning research on real-world 3D seismic reconstruction.*