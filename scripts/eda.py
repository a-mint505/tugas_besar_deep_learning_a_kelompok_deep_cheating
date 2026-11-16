"""EDA TXL-PBC dengan fokus platelet. Membaca COCO JSON hasil konversi."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COCO_DIR = Path("data/processed/coco_clean")
LOG_DIR, FIG_DIR = Path("logs"), Path("figures")
SPLITS = ["train", "val", "test"]
TOP_SOURCES = ["575x575", "360x363", "640x480", "256x256"]  # dari validate_dataset.py
YOLO_IMGSZ = [320, 480, 640, 800]
FRCNN_MIN = [320, 480, 640, 800]      # min_size; bawaan torchvision (seingat saya) 800
FRCNN_MAX = 1333                      # max_size bawaan torchvision (seingat saya)
ANCHOR_SIZES = [32, 64, 128, 256, 512]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]       # rasio tinggi/lebar


def load():
    rows, imgs = [], []
    for s in SPLITS:
        coco = json.loads((COCO_DIR / f"{s}.json").read_text())
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        im = {i["id"]: i for i in coco["images"]}
        for i in coco["images"]:
            imgs.append({"split": s, "image_id": i["id"], "W": i["width"], "H": i["height"]})
        for a in coco["annotations"]:
            i = im[a["image_id"]]
            _, _, w, h = a["bbox"]
            rows.append({"split": s, "image_id": a["image_id"], "cls": cats[a["category_id"]],
                         "w": w, "h": h, "W": i["width"], "H": i["height"]})
    boxes, images = pd.DataFrame(rows), pd.DataFrame(imgs)
    for d in (boxes, images):
        src = d["W"].astype(str) + "x" + d["H"].astype(str)
        d["source"] = src.where(src.isin(TOP_SOURCES), "lainnya")
    boxes["sqrt_area"] = np.sqrt(boxes.w * boxes.h)
    boxes["aspect"] = boxes.w / boxes.h
    return boxes, images


def scaled_tables(p):
    """Ukuran objek setelah diubah ke resolusi masukan model."""
    yolo_rows, frcnn_rows = [], []

    # YOLOv8: asumsi sisi terpanjang diubah ke imgsz
    for sz in YOLO_IMGSZ:
        a = p.sqrt_area * (sz / np.maximum(p.W, p.H))
        for cls, g in a.groupby(p.cls):
            yolo_rows.append({"imgsz": sz, "cls": cls, "median_px": g.median(),
                              "p10_px": g.quantile(.1),
                              "pct_below_8px": 100 * (g < 8).mean(),
                              "pct_below_16px": 100 * (g < 16).mean()})

    # Faster R-CNN: sisi terpendek diubah ke min_size (dibatasi max_size),
    # lalu IoU bentuk terbaik terhadap anchor bawaan
    sizes, ratios = np.array(ANCHOR_SIZES, float), np.array(ANCHOR_RATIOS)
    aw = (sizes[:, None] / np.sqrt(ratios)[None, :]).ravel()
    ah = (sizes[:, None] * np.sqrt(ratios)[None, :]).ravel()
    for ms in FRCNN_MIN:
        s = np.minimum(ms / np.minimum(p.W, p.H), FRCNN_MAX / np.maximum(p.W, p.H))
        w, h = (p.w * s).to_numpy()[:, None], (p.h * s).to_numpy()[:, None]
        inter = np.minimum(w, aw) * np.minimum(h, ah)
        iou = (inter / (w * h + aw * ah - inter)).max(axis=1)
        d = pd.DataFrame({"cls": p.cls.to_numpy(), "iou": iou})
        for cls, g in d.groupby("cls"):
            frcnn_rows.append({"min_size": ms, "cls": cls, "median_best_iou": g.iou.median(),
                               "pct_iou_below_0.5": 100 * (g.iou < .5).mean(),
                               "pct_iou_below_0.7": 100 * (g.iou < .7).mean()})
    return pd.DataFrame(yolo_rows), pd.DataFrame(frcnn_rows)


def main():
    boxes, images = load()
    LOG_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    boxes.to_csv("data/processed/boxes.csv", index=False)   # turunan, tidak di-commit

    is_plt = boxes.cls.str.lower().str.contains("platelet")
    assert is_plt.any(), "kelas platelet tidak ditemukan di nama kelas"
    plt_boxes = boxes[is_plt]
    pd.set_option("display.width", 200)

    # 1. jumlah kotak per kelas per split
    counts = pd.crosstab(boxes.split, boxes.cls).reindex(SPLITS)
    counts["total"] = counts.sum(axis=1)
    print("\n== Kotak per kelas ==\n", counts)
    counts.to_csv(LOG_DIR / "eda_class_counts.csv")

    # 2. platelet per citra
    idx = pd.MultiIndex.from_frame(images[["split", "image_id"]])
    per_img = plt_boxes.groupby(["split", "image_id"]).size().reindex(idx, fill_value=0)
    rows = []
    for s in SPLITS:
        v = per_img.xs(s, level="split")
        rows.append({"split": s, "images": len(v), "platelets": int(v.sum()),
                     "mean_per_img": round(v.mean(), 2), "max_per_img": int(v.max()),
                     "pct_img_without_platelet": round(100 * (v == 0).mean(), 1)})
    plt_img = pd.DataFrame(rows)
    print("\n== Platelet per citra ==\n", plt_img)
    plt_img.to_csv(LOG_DIR / "eda_platelet_per_image.csv", index=False)

    # 3. sumber (ukuran citra) per split dan platelet per sumber
    mix_img = pd.crosstab(images.source, images.split).reindex(columns=SPLITS, fill_value=0)
    mix_plt = pd.crosstab(plt_boxes.source, plt_boxes.split).reindex(columns=SPLITS, fill_value=0)
    src = pd.DataFrame({
        "images": images.groupby("source").size(),
        "platelets": plt_boxes.groupby("source").size(),
        "median_platelet_sqrt_area_px": plt_boxes.groupby("source").sqrt_area.median()}).fillna(0)
    src["platelets_per_img"] = src.platelets / src.images
    print("\n== Citra per sumber x split ==\n", mix_img)
    print("\n== Platelet per sumber x split ==\n", mix_plt)
    print("\n== Ringkasan per sumber ==\n", src.round(2))
    mix_img.to_csv(LOG_DIR / "eda_images_source_split.csv")
    mix_plt.to_csv(LOG_DIR / "eda_platelets_source_split.csv")
    src.round(3).to_csv(LOG_DIR / "eda_platelet_by_source.csv")

    # 4. ukuran kotak per kelas pada resolusi asli
    q = boxes.groupby("cls").sqrt_area.quantile([.05, .25, .5, .75, .95]).unstack().round(1)
    bins = pd.cut(boxes.sqrt_area, [0, 32, 96, np.inf], labels=["small<32", "medium", "large>96"])
    pct = (pd.crosstab(boxes.cls, bins, normalize="index") * 100).round(1)
    print("\n== sqrt(luas) kuantil (px, resolusi asli) ==\n", q)
    print("\n== Persen kotak per kategori ukuran COCO ==\n", pct)
    q.to_csv(LOG_DIR / "eda_box_size_quantiles.csv")
    pct.to_csv(LOG_DIR / "eda_box_size_coco_bins.csv")

    # 5. ukuran setelah resize + kecocokan anchor
    yolo_t, frcnn_t = scaled_tables(boxes)
    print("\n== YOLO: ukuran setelah resize ==\n", yolo_t.round(1))
    print("\n== Faster R-CNN: IoU bentuk terbaik terhadap anchor bawaan ==\n", frcnn_t.round(2))
    yolo_t.round(2).to_csv(LOG_DIR / "eda_size_after_resize_yolo.csv", index=False)
    frcnn_t.round(3).to_csv(LOG_DIR / "eda_anchor_fit_frcnn.csv", index=False)

    # 6. gambar
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    counts.drop(columns="total").plot.bar(ax=ax[0, 0], logy=True, rot=0,
                                          title="Jumlah kotak per kelas (skala log)")
    edges = np.linspace(0, 200, 81)
    for cls, g in boxes.groupby("cls"):
        ax[0, 1].hist(g.sqrt_area.clip(upper=200), bins=edges, alpha=.5, density=True, label=cls)
    ax[0, 1].axvline(32, ls="--", c="k")
    ax[0, 1].legend()
    ax[0, 1].set_title("sqrt(luas kotak), px asli (>200 dipotong); garis = 32 px")
    order = sorted(plt_boxes.source.unique())
    ax[0, 2].boxplot([plt_boxes[plt_boxes.source == s].sqrt_area for s in order])
    ax[0, 2].set_xticklabels(order)
    ax[0, 2].set_title("Ukuran platelet (px asli) per sumber citra")
    for s in SPLITS:
        v = per_img.xs(s, level="split")
        ax[1, 0].hist(v, bins=range(0, int(per_img.max()) + 2), alpha=.5, density=True, label=s)
    ax[1, 0].legend()
    ax[1, 0].set_title("Jumlah platelet per citra")
    ax[1, 1].hist(plt_boxes.aspect.clip(.3, 3), bins=60)
    ax[1, 1].set_title("Rasio lebar/tinggi kotak platelet (dipotong 0.3-3)")
    (pd.crosstab(images.split, images.source, normalize="index")
       .reindex(SPLITS).plot.bar(stacked=True, ax=ax[1, 2], rot=0,
                                 title="Komposisi sumber citra per split"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_overview.png", dpi=130)
    print("\nGambar:", FIG_DIR / "eda_overview.png")


if __name__ == "__main__":
    main()