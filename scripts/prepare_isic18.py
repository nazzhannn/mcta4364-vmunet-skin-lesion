"""Build the ISIC 2018 Task 1 dataset in the folder layout VM-UNet expects.

Source: ISIC 2018 Challenge, Task 1 (lesion boundary segmentation), official
training set (2,594 images) + validation set (100 images) = 2,694 images, the
same total the VM-UNet paper reports. Licence: CC-0 (challenge.isic-archive.com).

Steps:
  1. Read every image/mask pair straight from the downloaded zip files.
  2. Resize images to 256x256 (bilinear) and masks to 256x256 (nearest, kept binary).
  3. Split with a fixed seed: 30% test (808) as in the paper's 7:3 split, and the
     remaining 70% (1,886) into train (1,697) and validation (189, 10%).
     The paper uses its test set to pick the best checkpoint; we keep a separate
     validation set for that so the test set is only used once, at the end.
  4. Write data/isic18/{train,val,test}/{images,masks}/<id>.png, splits.csv and
     stats.csv (original size and lesion area, used by scripts/eda.py).

Usage: python scripts/prepare_isic18.py --zips <folder with the 4 zips> --out data/isic18
"""
import argparse
import csv
import io
import os
import random
import zipfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

ZIPS = {
    "train": ("ISIC2018_Task1-2_Training_Input.zip", "ISIC2018_Task1_Training_GroundTruth.zip"),
    "valid": ("ISIC2018_Task1-2_Validation_Input.zip", "ISIC2018_Task1_Validation_GroundTruth.zip"),
}
SIZE = 256
SEED = 42


def list_pairs(zdir, names):
    pairs = []
    for img_zip, msk_zip in (ZIPS[n] for n in names):
        with zipfile.ZipFile(os.path.join(zdir, img_zip)) as zi, zipfile.ZipFile(os.path.join(zdir, msk_zip)) as zm:
            masks = {os.path.basename(m)[:-len("_segmentation.png")]: m
                     for m in zm.namelist() if m.endswith("_segmentation.png")}
            for f in zi.namelist():
                if f.lower().endswith(".jpg"):
                    iid = os.path.basename(f)[:-4]
                    pairs.append((iid, img_zip, f, msk_zip, masks[iid]))
    return sorted(pairs)


_open = {}


def _zip(zdir, name):
    if name not in _open:
        _open[name] = zipfile.ZipFile(os.path.join(zdir, name))
    return _open[name]


def work(args):
    zdir, out, split, (iid, img_zip, img_f, msk_zip, msk_f) = args
    img = Image.open(io.BytesIO(_zip(zdir, img_zip).read(img_f)))
    w, h = img.size
    img.draft("RGB", (2 * SIZE, 2 * SIZE))  # fast JPEG decode at reduced scale
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    msk = Image.open(io.BytesIO(_zip(zdir, msk_zip).read(msk_f))).convert("L")
    msk = msk.resize((SIZE, SIZE), Image.NEAREST)
    m = (np.array(msk) > 127).astype(np.uint8) * 255
    img.save(os.path.join(out, split, "images", iid + ".png"))
    Image.fromarray(m).save(os.path.join(out, split, "masks", iid + ".png"))
    return iid, split, w, h, float((m > 0).mean()), float(np.array(img).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True)
    ap.add_argument("--out", default="data/isic18")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    pairs = list_pairs(a.zips, ["train", "valid"])
    print(len(pairs), "image/mask pairs")
    rng = random.Random(SEED)
    order = pairs[:]
    rng.shuffle(order)
    n_test = round(0.3 * len(order))
    test, rest = order[:n_test], order[n_test:]
    n_val = round(0.1 * len(rest))
    val, train = rest[:n_val], rest[n_val:]
    jobs = []
    for split, group in (("train", train), ("val", val), ("test", test)):
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(a.out, split, sub), exist_ok=True)
        jobs += [(a.zips, a.out, split, p) for p in group]
    print(f"train {len(train)}, val {len(val)}, test {len(test)}")

    with Pool(a.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=8), 1):
            rows.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    rows.sort()
    with open(os.path.join(a.out, "stats.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "split", "orig_w", "orig_h", "lesion_frac", "mean_intensity"])
        wr.writerows(rows)
    with open(os.path.join(a.out, "splits.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "split"])
        wr.writerows((r[0], r[1]) for r in rows)
    print("done")


if __name__ == "__main__":
    main()
