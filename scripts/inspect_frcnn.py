"""Dua perintah bantu untuk Anggota 3.

  python scripts/inspect_frcnn.py describe
      Cetak versi, bobot, min_size/max_size, anchor, parameter RPN/ROI dari model yang dibangun.
      Butuh torch+torchvision dan internet (unduh bobot COCO). TEMPEL KELUARANNYA ke laporan
      sebagai verifikasi nilai bawaan.

  python scripts/inspect_frcnn.py anchorfit --split train
      Kecocokan BENTUK anchor-kotak (IoU bentuk terbaik, kotak dan anchor dipusatkan di titik
      yang sama), memakai rumus yang sama dengan scripts/eda.py tetapi dengan definisi resolusi
      dan himpunan anchor yang bisa dipilih. Hanya numpy; tidak butuh torch.
      Pakai split TRAIN (bukan val/test) untuk memilih lengan ablasi.

Catatan: ini batas atas kecocokan (mengabaikan grid posisi anchor/stride) dan bukan prediksi
AP. RPN torchvision (seingat saya) memberi label positif bila IoU >= 0.7, dan juga anchor
terbaik tiap GT (allow_low_quality_matches) -> kotak yang IoU terbaiknya < 0.7 masih bisa
punya anchor positif; verifikasi nilai ini lewat `describe`.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COCO_DIR = Path("data/processed/coco_clean")
CLASS_IDS = {1: "WBC", 2: "RBC", 3: "Platelets"}
RATIOS = (0.5, 1.0, 2.0)       # rasio tinggi/lebar (sama dengan eda.py)


def load_boxes(split):
    coco = json.loads((COCO_DIR / f"{split}.json").read_text())
    im = {i["id"]: i for i in coco["images"]}
    rows = [(a["category_id"], a["bbox"][2], a["bbox"][3], im[a["image_id"]]["width"], im[a["image_id"]]["height"])
            for a in coco["annotations"]]
    return np.array(rows, dtype=float)    # kolom: cat, w, h, W, H


def scale_factor(mode, n, W, H, max_size=1333):
    if mode == "longest":                  # min_size=max_size=n  ->  n / sisi_panjang
        return n / np.maximum(W, H)
    if mode == "shortest":                 # min_size=n, max_size=1333
        return np.minimum(n / np.minimum(W, H), max_size / np.maximum(W, H))
    raise ValueError(mode)


def best_shape_iou(w, h, sizes):
    s = np.array(sizes, float)
    r = np.array(RATIOS)
    aw = (s[:, None] / np.sqrt(r)[None, :]).ravel()
    ah = (s[:, None] * np.sqrt(r)[None, :]).ravel()
    inter = np.minimum(w[:, None], aw) * np.minimum(h[:, None], ah)
    iou = inter / (w[:, None] * h[:, None] + aw * ah - inter)
    return iou.max(axis=1)


def cmd_anchorfit(a):
    b = load_boxes(a.split)
    anchor_sets = dict(kv.split("=") for kv in a.anchor_sets)
    out_rows = []
    for mode in a.size_modes:
        for n in a.sizes:
            s = scale_factor(mode, n, b[:, 3], b[:, 4])
            w, h = b[:, 1] * s, b[:, 2] * s
            for tag, sizes in anchor_sets.items():
                iou = best_shape_iou(w, h, [int(x) for x in sizes.split(",")])
                for cid, name in CLASS_IDS.items():
                    m = b[:, 0] == cid
                    out_rows.append({"split": a.split, "size_mode": mode, "N": n, "anchors": tag,
                                     "cls": name, "n_boxes": int(m.sum()),
                                     "median_side_px": round(float(np.median(np.sqrt(w[m] * h[m]))), 1),
                                     "median_best_iou": round(float(np.median(iou[m])), 3),
                                     "pct_iou_lt_0.5": round(100 * float((iou[m] < .5).mean()), 1),
                                     "pct_iou_lt_0.7": round(100 * float((iou[m] < .7).mean()), 1)})
    cols = list(out_rows[0])
    print("\t".join(cols))
    for r in out_rows:
        print("\t".join(str(r[c]) for c in cols))
    if a.out:
        Path(a.out).parent.mkdir(exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=cols)
            w_.writeheader(); w_.writerows(out_rows)
        print("ditulis:", a.out)


def cmd_describe(a):
    from frcnn_lib import build_model, describe_model, resolution_kwargs
    model = build_model(**resolution_kwargs("shortest", 800))   # = konfigurasi bawaan torchvision
    describe_model(model)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("describe")
    f = sub.add_parser("anchorfit")
    f.add_argument("--split", default="train", choices=["train", "val", "test"])
    f.add_argument("--size-modes", nargs="+", default=["longest", "shortest"])
    f.add_argument("--sizes", nargs="+", type=int, default=[480, 640, 800])
    f.add_argument("--anchor-sets", nargs="+",
                   default=["default=32,64,128,256,512", "x0.75=24,48,96,192,384", "x0.5=16,32,64,128,256"])
    f.add_argument("--out", default="logs/anggota3_anchor_fit.csv")
    a = ap.parse_args()
    {"describe": cmd_describe, "anchorfit": cmd_anchorfit}[a.cmd](a)


if __name__ == "__main__":
    main()
