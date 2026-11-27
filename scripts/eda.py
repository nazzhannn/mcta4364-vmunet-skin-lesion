"""Exploratory analysis of the prepared ISIC 2018 dataset (written for MCTA4364).

Reads data/isic18/stats.csv (made by prepare_isic18.py) and the 256x256 images.
Writes into figures/:
  samples.png         8 training images with their masks outlined
  lesion_size.png     lesion area (% of image) per split
  resolution.png      original image sizes
  eda_summary.json    counts, class imbalance and quality notes
"""
import argparse
import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/isic18")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(a.data, "stats.csv"))))
    for r in rows:
        r["lesion_frac"] = float(r["lesion_frac"])
        r["orig_w"], r["orig_h"] = int(r["orig_w"]), int(r["orig_h"])
        r["mean_intensity"] = float(r["mean_intensity"])
    splits = ["train", "val", "test"]

    # samples
    rng = np.random.default_rng(0)
    tr = [r for r in rows if r["split"] == "train"]
    pick = [tr[i] for i in rng.choice(len(tr), 8, replace=False)]
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.4))
    for k, r in enumerate(pick):
        img = Image.open(os.path.join(a.data, "train", "images", r["id"] + ".png"))
        m = np.array(Image.open(os.path.join(a.data, "train", "masks", r["id"] + ".png"))) > 127
        axk = ax[k // 4, k % 4]
        axk.imshow(img)
        axk.contour(m, levels=[0.5], colors="lime", linewidths=1)
        axk.set_title(f"{r['id']}\nlesion {100 * r['lesion_frac']:.0f}% of image", fontsize=7)
        axk.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "samples.png"), dpi=150)
    plt.close()

    # lesion size per split
    plt.figure(figsize=(6, 3.2))
    for s in splits:
        v = [100 * r["lesion_frac"] for r in rows if r["split"] == s]
        plt.hist(v, bins=40, range=(0, 100), histtype="step", density=True, label=f"{s} (n={len(v)})", linewidth=1.5)
    plt.xlabel("lesion area (% of image pixels)")
    plt.ylabel("density")
    plt.legend()
    plt.title("Lesion size distribution per split")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "lesion_size.png"), dpi=150)
    plt.close()

    # original resolution
    res = Counter(f"{r['orig_w']}x{r['orig_h']}" for r in rows)
    top = res.most_common(8)
    plt.figure(figsize=(6, 3.2))
    plt.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#2a6f97")
    plt.xlabel("images")
    plt.title(f"Most common original sizes ({len(res)} distinct sizes)")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "resolution.png"), dpi=150)
    plt.close()

    fr = np.array([r["lesion_frac"] for r in rows])
    summ = {
        "images": len(rows),
        "per_split": {s: sum(r["split"] == s for r in rows) for s in splits},
        "distinct_original_sizes": len(res),
        "most_common_sizes": top[:5],
        "orig_width_range": [min(r["orig_w"] for r in rows), max(r["orig_w"] for r in rows)],
        "orig_height_range": [min(r["orig_h"] for r in rows), max(r["orig_h"] for r in rows)],
        "lesion_pixel_share_overall_pct": round(100 * float(fr.mean()), 1),
        "lesion_frac_median_pct": round(100 * float(np.median(fr)), 1),
        "lesion_frac_per_split_mean_pct": {s: round(100 * float(np.mean([r["lesion_frac"] for r in rows if r["split"] == s])), 1) for s in splits},
        "small_lesions_lt_5pct": int((fr < 0.05).sum()),
        "large_lesions_gt_70pct": int((fr > 0.7).sum()),
        "masks_touching_whole_image_gt_95pct": int((fr > 0.95).sum()),
    }
    with open(os.path.join(a.out, "eda_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
