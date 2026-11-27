"""Build the progress report (HTML -> PDF with Edge/Chrome headless) from the result files.

Every number in the report is read from results/ and figures/, so the PDF matches
the code at the progress-v1 tag.
  python report/build_report.py            -> report/MCTA4364_Progress_<Name>.pdf
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "report")
NAME = "Muhammad Nazhan bin Mohamed Nadzri"
MATRIC = "2313703"
REPO = "https://github.com/nazzhannn/mcta4364-vmunet-skin-lesion"
PDF_NAME = "MCTA4364_Progress_MuhammadNazhan.pdf"
PAPER = {"miou": 81.35, "dsc": 89.71, "acc": 94.91, "spe": 96.13, "sen": 91.12}


def jload(*p):
    path = os.path.join(ROOT, *p)
    return json.load(open(path)) if os.path.exists(path) else None


def curve(log_rows, out):
    ep = [int(r["epoch"]) for r in log_rows]
    fig, ax1 = plt.subplots(figsize=(5.2, 2.6))
    ax1.plot(ep, [float(r["train_loss"]) for r in log_rows], "o-", label="train loss", color="#1f3b5a")
    ax1.plot(ep, [float(r["val_loss"]) for r in log_rows], "s-", label="val loss", color="#a26b00")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BceDice loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, [100 * float(r["val_dsc"]) for r in log_rows], "^--", label="val DSC", color="#2d6a35")
    ax2.set_ylabel("val DSC (%)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.set_xticks(ep)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()


def f2(x):
    return f"{x:.2f}"


def main():
    eda = jload("figures", "eda_summary.json")
    base = jload("results", "baseline", "test", "metrics.json")
    auth = jload("results", "authors_ckpt", "test", "metrics.json")
    log = list(csv.DictReader(open(os.path.join(ROOT, "results", "baseline", "log.csv"))))
    curve(log, os.path.join(ROOT, "figures", "training_curve.png"))
    ep_done = len(log)
    min_per_ep = sum(float(r["train_min"]) for r in log) / ep_done
    best_ep = min(log, key=lambda r: float(r["val_loss"]))
    bp, ap_ = base["pooled"], auth["pooled"]
    bi, ai = base["per_image"], auth["per_image"]

    def row(label, m, note=""):
        return (f"<tr><td>{label}</td>" + "".join(f"<td>{f2(m[k])}</td>" for k in ("miou", "dsc", "acc", "spe", "sen"))
                + f"<td class='n'>{note}</td></tr>")

    table = (row("VM-UNet, paper Table I (300 epochs, A6000 GPU)", PAPER, "reported")
             + row("Authors' released checkpoint, our test split (CPU inference)", ap_, "optimistic, see text")
             + row(f"<b>Our baseline: VM-UNet trained on CPU, {ep_done} epochs</b>", bp, "this work"))
    gap = PAPER["dsc"] - bp["dsc"]
    gap_auth = PAPER["dsc"] - ap_["dsc"]

    html = open(os.path.join(R, "template.html"), encoding="utf-8").read()
    subs = {
        "NAME": NAME, "MATRIC": MATRIC, "REPO": REPO,
        "N_IMG": eda["images"], "N_TRAIN": eda["per_split"]["train"], "N_VAL": eda["per_split"]["val"],
        "N_TEST": eda["per_split"]["test"], "N_SIZES": eda["distinct_original_sizes"],
        "W_MIN": eda["orig_width_range"][0], "W_MAX": eda["orig_width_range"][1],
        "H_MIN": eda["orig_height_range"][0], "H_MAX": eda["orig_height_range"][1],
        "TOP_SIZE": eda["most_common_sizes"][0][0], "TOP_SIZE_N": eda["most_common_sizes"][0][1],
        "LES_SHARE": eda["lesion_pixel_share_overall_pct"], "LES_MED": eda["lesion_frac_median_pct"],
        "LES_TR": eda["lesion_frac_per_split_mean_pct"]["train"], "LES_VA": eda["lesion_frac_per_split_mean_pct"]["val"],
        "LES_TE": eda["lesion_frac_per_split_mean_pct"]["test"],
        "SMALL": eda["small_lesions_lt_5pct"], "LARGE": eda["large_lesions_gt_70pct"], "FULL": eda["masks_touching_whole_image_gt_95pct"],
        "TABLE": table, "EPOCHS": ep_done, "MIN_PER_EP": f"{min_per_ep:.0f}", "BEST_EP": best_ep["epoch"],
        "B_DSC": f2(bp["dsc"]), "B_MIOU": f2(bp["miou"]), "B_SEN": f2(bp["sen"]), "B_SPE": f2(bp["spe"]),
        "A_DSC": f2(ap_["dsc"]), "A_MIOU": f2(ap_["miou"]),
        "GAP": f2(gap), "GAP_AUTH": f2(gap_auth),
        "B_DMEAN": f2(bi["dice_mean"]), "B_DMED": f2(bi["dice_median"]), "B_HD": f2(bi["hd95_mean_px"]),
        "B_LOW": bi["images_dice_below_0.7"], "B_BORDER": bi["border_error_share_mean"], "BAND": bi["border_band_px"],
        "A_DMEAN": f2(ai["dice_mean"]), "A_DMED": f2(ai["dice_median"]), "A_HD": f2(ai["hd95_mean_px"]), "A_LOW": ai["images_dice_below_0.7"],
        "A_BORDER": ai["border_error_share_mean"],
        "SEC_IMG": base["cpu_seconds_per_image"],
    }
    for k, v in subs.items():
        html = html.replace("{{" + k + "}}", str(v))
    assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
    out_html = os.path.join(R, "report.html")
    open(out_html, "w", encoding="utf-8").write(html)

    browser = next((b for b in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                                shutil.which("chromium"), shutil.which("google-chrome")] if b and os.path.exists(b)), None)
    if not browser:
        sys.exit("no Edge/Chrome found; open report/report.html and print to PDF")
    pdf = os.path.join(R, PDF_NAME)
    if os.path.exists(pdf):
        os.remove(pdf)
    # a separate profile so an open Edge window does not swallow the job
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--user-data-dir=" + os.path.join(os.environ.get("TEMP", R), "mcta_edge_profile"),
                    f"--print-to-pdf={pdf}", "file:///" + out_html.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Edge can return before the file is fully written: wait until its size settles
    import time
    last = -1
    for _ in range(120):
        size = os.path.getsize(pdf) if os.path.exists(pdf) else -1
        if size > 0 and size == last:
            break
        last = size
        time.sleep(1)
    print("wrote", pdf, f"({last} bytes)")


if __name__ == "__main__":
    main()
