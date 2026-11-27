# Improving Skin Lesion Boundary Segmentation Using VM-UNet

MCTA 4364 Machine Vision, Semester 1 2026/2027 · Project progress report (`progress-v1`)

| | |
|---|---|
| Member | **Muhammad Nazhan bin Mohamed Nadzri** (matric **2313703**), group representative, solo group |
| Track | Track 2: research paper extension |
| Paper | J. Ruan, J. Li, S. Xiang, *VM-UNet: Vision Mamba UNet for Medical Image Segmentation*, arXiv:2402.02491, 2024. [PDF](https://arxiv.org/pdf/2402.02491) |
| Official code | https://github.com/JCruan519/VM-UNet (commit `b87827e`, Apache-2.0) |
| Dataset | ISIC 2018 Task 1, lesion boundary segmentation. [challenge.isic-archive.com/data](https://challenge.isic-archive.com/data/), CC-0 |

## What this repository contains

The baseline is the official VM-UNet. The official code needs an NVIDIA GPU, because its
Mamba layers call the CUDA kernel in `mamba_ssm`. This project runs on a PC with an AMD GPU,
so the one CUDA function (`selective_scan_fn`) was rewritten for the CPU. Nothing else in
the model was changed.

```
vmunet/                     official VM-UNet code, imported unchanged in the first commit
  models/vmunet/vmamba.py   only change: falls back to cpu_scan/ when mamba_ssm is missing
  pre_trained_weights/      put vmamba_small_e238_ema.pth here (ImageNet VMamba-S, not in git)
cpu_scan/
  selective_scan_cpu.py     ours: Numba CPU selective scan, forward + hand-written backward
tests/test_scan.py          ours: checks the CPU scan against a plain PyTorch loop (output and all gradients)
scripts/prepare_isic18.py   ours: builds data/isic18 from the official zips (resize, fixed split)
scripts/eda.py              ours: dataset figures and statistics
train_cpu.py                ours: CPU training loop reusing the official model, loss, augmentation and optimiser
evaluate.py                 ours: test metrics (paper formulas), per-image Dice/IoU/HD95, border errors, examples
data/isic18/splits.csv      the exact train/val/test split used (ids)
results/                    logs and metrics of the runs reported (checkpoints not in git)
figures/                    figures used in the report
```

## 1. Environment

Tested on Windows 11, Python 3.12, Intel i7-8700 (12 threads), 16 GB RAM, no NVIDIA GPU.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests/test_scan.py         # must print PASS
```

On a machine with an NVIDIA GPU you can instead install `mamba_ssm==1.0.1` as in the
official README; the model then uses the CUDA kernel automatically.

## 2. Data

Download the four ISIC 2018 Task 1 zips into one folder:

- `ISIC2018_Task1-2_Training_Input.zip` (10.6 GB) and `ISIC2018_Task1_Training_GroundTruth.zip`
- `ISIC2018_Task1-2_Validation_Input.zip` and `ISIC2018_Task1_Validation_GroundTruth.zip`

All four are at `https://isic-archive.s3.amazonaws.com/challenges/2018/<file name>`. Then:

```bash
python scripts/prepare_isic18.py --zips <folder with the zips> --out data/isic18
python scripts/eda.py --data data/isic18 --out figures
```

This gives 2,694 images at 256×256: train 1,697 / val 189 / test 808, with seed 42. The paper
splits 7:3 (1,886 / 808). We take 10% of its training part as a validation set for choosing
the checkpoint, so the test set is used only once.

## 3. Baseline

Pretrained weights: from the authors' [Google Drive folder](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx),
download `pre_trained_weights/vmamba_small_e238_ema.pth` into `vmunet/pre_trained_weights/`.

```bash
python train_cpu.py --epochs 8 --batch-size 8 --hours 6 --out results/baseline
python evaluate.py --ckpt results/baseline/best.pth --out results/baseline/test
```

Training resumes from `results/baseline/latest.pth` if it is interrupted.

To check the authors' released checkpoint (`best-ckpt/best-vmunet-isic18.pth` in the same Drive
folder) on our test split:

```bash
python evaluate.py --ckpt checkpoints/best-vmunet-isic18.pth --out results/authors_ckpt/test --label "authors' ckpt"
```

## 4. Results

RESULTS_PLACEHOLDER

## Credits and AI use

- Model, losses, augmentation and pretrained weights: the official VM-UNet repository and VMamba (reused, cited).
- Written for this project: `cpu_scan/`, `tests/`, `scripts/`, `train_cpu.py`, `evaluate.py` and the fallback import in `vmamba.py`.
- AI tools: Claude (Anthropic) was used to help write the CPU scan, scripts and report drafts. All code was run and checked locally, and the scan is verified against a reference implementation in `tests/test_scan.py`.
