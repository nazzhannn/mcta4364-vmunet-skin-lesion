"""Evaluate a VM-UNet checkpoint on the ISIC 2018 test split.

Outputs (in --out):
  metrics.json     pooled pixel metrics (same formulas as the official engine.py,
                   so they compare with the paper's Table I), plus mean per-image
                   Dice/IoU/HD95 and the share of wrong pixels near lesion borders
  per_image.csv    Dice, IoU, HD95, lesion size and border-error share per image
  examples.png     best and worst test images: image, ground truth, prediction
Written for this project (MCTA4364).

Example:
  python evaluate.py --ckpt results/baseline/best.pth --out results/baseline/test
"""
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_cpu import ISICSplit, build_model, pixel_metrics, transforms_for  # noqa: E402

BAND = 5  # pixels: "near the border" means within 5 px of the true lesion edge


def hd95(p, g):
    if p.sum() == 0 or g.sum() == 0:
        return float("nan")
    pe = p ^ ndimage.binary_erosion(p)
    ge = g ^ ndimage.binary_erosion(g)
    dg = ndimage.distance_transform_edt(~ge)
    dp = ndimage.distance_transform_edt(~pe)
    d = np.concatenate([dg[pe], dp[ge]])
    return float(np.percentile(d, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "isic18"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--label", default="VM-UNet")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    model = build_model(None)
    sd = torch.load(a.ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {a.ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    assert not missing, missing[:5]
    model.eval()

    ds = ISICSplit(a.data, a.split, transforms_for(False))
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    tp = fp = tn = fn = 0
    rows, probs = [], {}
    t0 = time.time()
    with torch.no_grad():
        for img, msk, ids in ld:
            out = model(img)
            for k in range(len(ids)):
                pr = out[k, 0].numpy()
                p = pr >= 0.5
                g = msk[k, 0].numpy() >= 0.5
                tp += int((p & g).sum()); fp += int((p & ~g).sum())
                fn += int((~p & g).sum()); tn += int((~p & ~g).sum())
                inter, union = (p & g).sum(), (p | g).sum()
                dice = 2 * inter / max(p.sum() + g.sum(), 1)
                iou = inter / max(union, 1)
                edge = g ^ ndimage.binary_erosion(g)
                near = ndimage.distance_transform_edt(~edge) <= BAND if edge.any() else np.zeros_like(g)
                wrong = p ^ g
                border_share = float((wrong & near).sum() / wrong.sum()) if wrong.sum() else float("nan")
                rows.append({"id": ids[k], "dice": float(dice), "iou": float(iou), "hd95": hd95(p, g),
                             "lesion_frac": float(g.mean()), "wrong_px": int(wrong.sum()),
                             "border_error_share": border_share})
                probs[ids[k]] = pr
            print(f"  {len(rows)}/{len(ds)}", flush=True)
    secs = time.time() - t0

    pm = pixel_metrics(tp, fp, tn, fn)
    d = np.array([r["dice"] for r in rows])
    res = {
        "checkpoint": a.ckpt, "split": a.split, "images": len(rows),
        "pooled": {k: round(100 * v, 2) for k, v in pm.items()},
        "per_image": {
            "dice_mean": round(100 * float(d.mean()), 2), "dice_median": round(100 * float(np.median(d)), 2),
            "iou_mean": round(100 * float(np.mean([r["iou"] for r in rows])), 2),
            "hd95_mean_px": round(float(np.nanmean([r["hd95"] for r in rows])), 2),
            "images_dice_below_0.7": int((d < 0.7).sum()),
            "border_error_share_mean": round(100 * float(np.nanmean([r["border_error_share"] for r in rows])), 1),
            "border_band_px": BAND,
        },
        "cpu_seconds_per_image": round(secs / len(rows), 3),
    }
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(a.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(res, indent=2))

    # qualitative examples: 3 best, 3 worst
    srt = sorted(rows, key=lambda r: r["dice"])
    pick = [("worst", r) for r in srt[:3]] + [("best", r) for r in srt[::-1][:3]]
    fig, ax = plt.subplots(len(pick), 3, figsize=(7.5, 2.6 * len(pick)))
    for i, (tag, r) in enumerate(pick):
        img = Image.open(os.path.join(a.data, a.split, "images", r["id"] + ".png"))
        gt = np.array(Image.open(os.path.join(a.data, a.split, "masks", r["id"] + ".png"))) > 127
        pr = probs[r["id"]] >= 0.5
        ax[i, 0].imshow(img); ax[i, 0].set_title(f"{tag}: {r['id']}", fontsize=8)
        ax[i, 1].imshow(gt, cmap="gray"); ax[i, 1].set_title("ground truth", fontsize=8)
        ov = np.array(img).copy()
        ov[pr & ~gt] = [255, 0, 0]      # false positive: red
        ov[~pr & gt] = [0, 90, 255]     # false negative: blue
        ax[i, 2].imshow(ov); ax[i, 2].contour(pr, levels=[0.5], colors="yellow", linewidths=0.8)
        ax[i, 2].set_title(f"{a.label} Dice {r['dice']:.3f} (red FP, blue FN)", fontsize=8)
        for j in range(3):
            ax[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "examples.png"), dpi=150)
    print("saved", os.path.join(a.out, "examples.png"))


if __name__ == "__main__":
    main()
