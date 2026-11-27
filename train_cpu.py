"""Train the VM-UNet baseline on ISIC 2018 on a CPU (no NVIDIA GPU needed).

Reused from the official code (vmunet/): the VM-UNet model, the ImageNet
VMamba-S weight loading, the BceDice loss, the augmentations and normalisation,
and the AdamW + CosineAnnealingLR settings.
Written for this project: this loop (device-agnostic instead of .cuda()), a
separate validation split for choosing the checkpoint, resuming, a wall-clock
budget, and a CSV log per epoch. The Mamba scan runs on the CPU through
cpu_scan/selective_scan_cpu.py.

Example:
  python train_cpu.py --epochs 8 --batch-size 8 --hours 6 --out results/baseline
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "vmunet"))
from models.vmunet.vmunet import VMUNet  # noqa: E402
from utils import (BceDiceLoss, myNormalize, myRandomHorizontalFlip,  # noqa: E402
                   myRandomRotation, myRandomVerticalFlip, myResize, myToTensor)


class ISICSplit(Dataset):
    """Same loading as the official NPY_datasets, but any split folder."""

    def __init__(self, root, split, transform):
        img_dir, msk_dir = os.path.join(root, split, "images"), os.path.join(root, split, "masks")
        self.ids = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".png"))
        self.img_dir, self.msk_dir, self.transform = img_dir, msk_dir, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]
        img = np.array(Image.open(os.path.join(self.img_dir, iid + ".png")).convert("RGB"))
        msk = np.expand_dims(np.array(Image.open(os.path.join(self.msk_dir, iid + ".png")).convert("L")), 2) / 255
        img, msk = self.transform((img, msk))
        return img.float(), msk.float(), iid


def transforms_for(train):
    # Identical to the official configs/config_setting.py for 'isic18'.
    if train:
        return transforms.Compose([myNormalize("isic18", train=True), myToTensor(),
                                   myRandomHorizontalFlip(p=0.5), myRandomVerticalFlip(p=0.5),
                                   myRandomRotation(p=0.5, degree=[0, 360]), myResize(256, 256)])
    return transforms.Compose([myNormalize("isic18", train=False), myToTensor(), myResize(256, 256)])


def build_model(pretrained):
    model = VMUNet(num_classes=1, input_channels=3, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
                   drop_path_rate=0.2, load_ckpt_path=pretrained if pretrained else None)
    if pretrained:
        model.load_from()
    return model


def pixel_metrics(tp, fp, tn, fn):
    """Same formulas as the official engine.py (pooled over all test pixels)."""
    return {
        "miou": tp / max(tp + fp + fn, 1),
        "dsc": 2 * tp / max(2 * tp + fp + fn, 1),
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        "spe": tn / max(tn + fp, 1),
        "sen": tp / max(tp + fn, 1),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    tp = fp = tn = fn = 0
    losses = []
    for img, msk, _ in loader:
        out = model(img)
        losses.append(criterion(out, msk).item())
        p = out >= threshold
        g = msk >= 0.5
        tp += (p & g).sum().item()
        fp += (p & ~g).sum().item()
        fn += (~p & g).sum().item()
        tn += (~p & ~g).sum().item()
    m = pixel_metrics(tp, fp, tn, fn)
    m["loss"] = float(np.mean(losses))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "isic18"))
    ap.add_argument("--pretrained", default=os.path.join(ROOT, "vmunet", "pre_trained_weights", "vmamba_small_e238_ema.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "baseline"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hours", type=float, default=0, help="stop after the epoch that passes this many hours (0 = no limit)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--max-iters", type=int, default=0, help="debug: stop each epoch after N iterations")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    import random
    random.seed(a.seed)
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    train_ds = ISICSplit(a.data, "train", transforms_for(True))
    val_ds = ISICSplit(a.data, "val", transforms_for(False))
    train_ld = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"train {len(train_ds)} images, val {len(val_ds)} images, {len(train_ld)} iterations per epoch")

    model = build_model(a.pretrained if os.path.exists(a.pretrained) else None)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    criterion = BceDiceLoss(wb=1, wd=1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    # The paper uses T_max=50 over 300 epochs; with a short CPU run we let the
    # cosine schedule finish within the epochs we actually train.
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)

    start_epoch, best = 1, float("inf")
    latest = os.path.join(a.out, "latest.pth")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed after epoch {ck['epoch']}")

    log_path = os.path.join(a.out, "log.csv")
    new_log = not os.path.exists(log_path)
    t_start = time.time()
    for epoch in range(start_epoch, a.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for it, (img, msk, _) in enumerate(train_ld, 1):
            opt.zero_grad()
            loss = criterion(model(img), msk)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if it % 10 == 0 or it == 1:
                el = time.time() - t0
                eta = el / it * (len(train_ld) - it)
                print(f"epoch {epoch} it {it}/{len(train_ld)} loss {np.mean(losses):.4f} "
                      f"{el / it:.1f}s/it, epoch ETA {eta / 60:.0f} min", flush=True)
            if a.max_iters and it >= a.max_iters:
                break
        sch.step()
        train_time = time.time() - t0
        vm = evaluate(model, val_ld, criterion)
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f}, val loss {vm['loss']:.4f}, "
              f"val DSC {vm['dsc']:.4f}, val mIoU {vm['miou']:.4f}, {train_time / 60:.1f} min", flush=True)
        if vm["loss"] < best:
            best = vm["loss"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pth"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "epoch": epoch, "best": best}, latest)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_log:
                w.writerow(["epoch", "train_loss", "val_loss", "val_dsc", "val_miou", "val_acc",
                            "val_spe", "val_sen", "lr", "train_min"])
                new_log = False
            w.writerow([epoch, f"{np.mean(losses):.5f}", f"{vm['loss']:.5f}", f"{vm['dsc']:.5f}",
                        f"{vm['miou']:.5f}", f"{vm['acc']:.5f}", f"{vm['spe']:.5f}", f"{vm['sen']:.5f}",
                        f"{opt.param_groups[0]['lr']:.2e}", f"{train_time / 60:.2f}"])
        if a.hours and time.time() - t_start > a.hours * 3600:
            print(f"time budget of {a.hours} h reached after epoch {epoch}")
            break
    print("training finished")


if __name__ == "__main__":
    main()
