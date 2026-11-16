"""Komponen Faster R-CNN ResNet-50 FPN untuk TXL-PBC (Anggota 3).

Dipakai oleh scripts/train_frcnn.py dan scripts/inspect_frcnn.py.
Jalankan dari akar repo (path relatif: data/raw/txl_pbc, data/processed/coco_clean, logs/).

STATUS: ditulis tanpa bisa dijalankan di mesin penulis (tidak ada torch/data). Jalankan
`python scripts/train_frcnn.py --smoke` di Colab sebelum percaya pada apa pun di sini.
"""
from __future__ import annotations

import contextlib
import io
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

import json

from utils import make_loader  # scripts/utils.py (milik Anggota 1)

DATA_ROOT = Path("data/raw/txl_pbc")
COCO_DIR = Path("data/processed/coco_clean")

NUM_CLASSES = 4                       # 0 = latar belakang + 3 kelas
CLASS_IDS = {1: "WBC", 2: "RBC", 3: "Platelets"}
PLATELET_ID = 3

# Nilai bawaan torchvision (SEINGAT SAYA; dicetak ulang oleh describe_model untuk verifikasi)
DEFAULT_ANCHOR_SIZES = (32, 64, 128, 256, 512)
DEFAULT_ANCHOR_RATIOS = (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------- data
class CocoDetDataset(Dataset):
    """Membaca JSON COCO (hasil convert_yolo_to_coco.py) -> target torchvision.

    Target: {"boxes": float32 [N,4] xyxy piksel, "labels": int64 [N], "image_id": int64 [1]}.
    - Kotak dipotong ke batas citra; kotak dengan sisi < min_box_side piksel dibuang
      (torchvision melempar error untuk kotak lebar/tinggi <= 0). Jumlah yang dibuang
      dicatat di self.stats.
    - Citra tanpa kotak tetap dipertahankan dengan boxes shape (0,4) dan labels shape (0,)
      (torchvision mendukung ini; jadi contoh negatif untuk RPN).
    - train=True: flip horizontal/vertikal acak (sel darah tidak punya orientasi tetap).
      Ini augmentasi yang SANGAT minimal; YOLOv8 punya augmentasi bawaan jauh lebih kaya
      (mosaic, dll.), jadi bedanya harus ditulis di laporan sebagai catatan keterbandingan.
    """

    def __init__(self, ann_file, img_root=DATA_ROOT, train=False, hflip=0.5, vflip=0.0,
                 min_box_side=1.0, limit=None):
        self.ann_file = Path(ann_file)
        coco = json.loads(self.ann_file.read_text())
        self.img_root = Path(img_root)
        self.train, self.hflip, self.vflip = train, hflip, vflip

        self.images = sorted(coco["images"], key=lambda d: d["id"])
        if limit is not None:
            self.images = self.images[:limit]
        keep = {im["id"] for im in self.images}
        raw = defaultdict(list)
        for a in coco["annotations"]:
            if a["image_id"] in keep:
                raw[a["image_id"]].append(a)

        self.stats = {"n_images": len(self.images), "n_boxes_json": 0,
                      "n_dropped_degenerate": 0, "n_dropped_crowd": 0, "n_images_no_box": 0}
        self.targets = []
        for im in self.images:
            W, H = float(im["width"]), float(im["height"])
            boxes, labels = [], []
            for a in raw[im["id"]]:
                self.stats["n_boxes_json"] += 1
                if a.get("iscrowd", 0):
                    self.stats["n_dropped_crowd"] += 1
                    continue
                cid = int(a["category_id"])
                if cid not in CLASS_IDS:
                    raise ValueError(f"category_id {cid} tidak dikenal (harus 1..3), citra {im['file_name']}")
                x, y, w, h = a["bbox"]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if (x2 - x1) < min_box_side or (y2 - y1) < min_box_side:
                    self.stats["n_dropped_degenerate"] += 1
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cid)
            if not boxes:
                self.stats["n_images_no_box"] += 1
            self.targets.append((torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                 torch.tensor(labels, dtype=torch.int64)))

    def __len__(self):
        return len(self.images)

    def image_ids(self):
        return [im["id"] for im in self.images]

    def __getitem__(self, idx):
        im = self.images[idx]
        with Image.open(self.img_root / im["file_name"]) as pil:
            pil = pil.convert("RGB")           # beberapa PNG bisa RGBA / grayscale
        W, H = pil.size
        if (W, H) != (im["width"], im["height"]):
            raise ValueError(f"ukuran citra {pil.size} != JSON {(im['width'], im['height'])}: {im['file_name']}")
        img = TF.to_tensor(pil)                # float32 [3,H,W] dalam [0,1]
        boxes, labels = self.targets[idx][0].clone(), self.targets[idx][1].clone()

        if self.train:
            if self.hflip > 0 and torch.rand(1).item() < self.hflip:
                img = img.flip(-1)
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
            if self.vflip > 0 and torch.rand(1).item() < self.vflip:
                img = img.flip(-2)
                boxes[:, [1, 3]] = H - boxes[:, [3, 1]]

        return img, {"boxes": boxes, "labels": labels,
                     "image_id": torch.tensor([im["id"]], dtype=torch.int64)}


# -------------------------------------------------------------------------- model
def resolution_kwargs(size_mode: str, n: int, default_max: int = 1333):
    """Terjemahkan 'definisi resolusi' ke min_size/max_size torchvision.

    longest : sisi terpanjang = n  (min_size=max_size=n). Sebanding dengan imgsz YOLOv8
              (YOLO: sisi terpanjang -> imgsz). Rumus resize torchvision (seingat saya):
              scale = min(min_size/sisi_pendek, max_size/sisi_panjang) -> = n/sisi_panjang.
              Citra yang lebih kecil dari n ikut DIPERBESAR (sama dengan YOLO).
    shortest: sisi terpendek = n, dibatasi max_size (perilaku bawaan torchvision, n=800).
    """
    if size_mode == "longest":
        return {"min_size": n, "max_size": n}
    if size_mode == "shortest":
        return {"min_size": n, "max_size": default_max}
    raise ValueError(size_mode)


def build_model(min_size=800, max_size=1333, anchor_sizes=None,
                anchor_ratios=DEFAULT_ANCHOR_RATIOS, trainable_backbone_layers=3):
    """fasterrcnn_resnet50_fpn berbobot COCO, box predictor diganti ke NUM_CLASSES=4.

    - num_classes TIDAK diberikan ke builder: dengan weights=..., builder memeriksa
      num_classes terhadap 91 kelas COCO dan menolak nilai lain (seingat saya). Predictor
      diganti sesudahnya.
    - anchor_sizes=None -> anchor bawaan. Jika diisi (5 angka, satu per level FPN),
      AnchorGenerator diganti dengan 3 rasio yang sama -> bentuk kepala RPN tidak berubah,
      bobot RPN pralatih tetap termuat (tetapi dilatih untuk anchor bawaan; lihat catatan
      ancaman validitas di docs/anggota3_frcnn.md).
    """
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size,
                                    trainable_backbone_layers=trainable_backbone_layers)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if anchor_sizes is not None:
        assert len(anchor_sizes) == 5, "harus 5 ukuran (satu per level FPN: P2-P6)"
        ag = AnchorGenerator(sizes=tuple((int(s),) for s in anchor_sizes),
                             aspect_ratios=(tuple(float(r) for r in anchor_ratios),) * len(anchor_sizes))
        n_head = model.rpn.head.cls_logits.out_channels
        assert ag.num_anchors_per_location()[0] == n_head, \
            f"jumlah anchor/lokasi ({ag.num_anchors_per_location()[0]}) != kepala RPN ({n_head})"
        model.rpn.anchor_generator = ag
    return model


def describe_model(model, weights_name="FasterRCNN_ResNet50_FPN_Weights.DEFAULT"):
    """Cetak atribut model untuk memverifikasi nilai bawaan (jangan mengandalkan ingatan)."""
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    def g(obj, name, default="(atribut tidak ada)"):
        return getattr(obj, name, default)

    w = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    print(f"torch {torch.__version__} | torchvision {torchvision.__version__}")
    print(f"{weights_name} -> {w}")
    print("meta (tanpa daftar kategori):", {k: v for k, v in w.meta.items() if k != "categories"})
    t = model.transform
    print("\n[transform]")
    print("  min_size :", g(t, "min_size"), "(tuple; training memilih acak, eval memakai elemen terakhir)")
    print("  max_size :", g(t, "max_size"))
    print("  mean/std :", g(t, "image_mean"), g(t, "image_std"))
    ag = model.rpn.anchor_generator
    print("\n[anchor generator]")
    print("  sizes        :", g(ag, "sizes"))
    print("  aspect_ratios:", g(ag, "aspect_ratios"))
    print("  anchor/lokasi:", ag.num_anchors_per_location())
    print("  RPN head out_channels cls_logits:", model.rpn.head.cls_logits.out_channels)
    r = model.rpn
    print("\n[RPN]")
    print("  pre_nms_top_n  :", g(r, "_pre_nms_top_n"))
    print("  post_nms_top_n :", g(r, "_post_nms_top_n"))
    print("  nms_thresh     :", g(r, "nms_thresh"))
    print("  fg/bg IoU      :", g(g(r, "proposal_matcher"), "high_threshold"), "/",
          g(g(r, "proposal_matcher"), "low_threshold"),
          "| allow_low_quality:", g(g(r, "proposal_matcher"), "allow_low_quality_matches"))
    print("  batch/pos_frac :", g(g(r, "fg_bg_sampler"), "batch_size_per_image"), "/",
          g(g(r, "fg_bg_sampler"), "positive_fraction"))
    h = model.roi_heads
    print("\n[ROI heads]")
    print("  score_thresh        :", g(h, "score_thresh"))
    print("  nms_thresh          :", g(h, "nms_thresh"))
    print("  detections_per_img  :", g(h, "detections_per_img"))
    print("  fg/bg IoU           :", g(g(h, "proposal_matcher"), "high_threshold"), "/",
          g(g(h, "proposal_matcher"), "low_threshold"))
    print("  batch/pos_frac      :", g(g(h, "fg_bg_sampler"), "batch_size_per_image"), "/",
          g(g(h, "fg_bg_sampler"), "positive_fraction"))
    print("  box_predictor       :", h.box_predictor)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[parameter] total {n_all/1e6:.2f} M, dilatih {n_tr/1e6:.2f} M")
    frozen = sorted({n.split('.')[2] if n.startswith('backbone.body.') else n.split('.')[0]
                     for n, p in model.named_parameters() if not p.requires_grad})
    print("  blok beku:", frozen)


# --------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict(model, dataset, device, batch_size=4, num_workers=2, seed=0):
    """Prediksi dalam format hasil COCO (bbox xywh, koordinat piksel citra asli).
    Model torchvision mengembalikan kotak dalam koordinat citra masukan (setelah postprocess)."""
    loader = make_loader(dataset, seed, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    results = []
    for images, targets in loader:
        outs = model([im.to(device) for im in images])
        for out, tgt in zip(outs, targets):
            image_id = int(tgt["image_id"].item())
            boxes, scores, labels = out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()
            for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
                results.append({"image_id": image_id, "category_id": int(l),
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return results


def load_gt(ann_file):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ann_file))


def coco_eval(gt, results, img_ids, verbose=False):
    """Metrik COCO (pycocotools). Kunci: map50, map50_95, ap50_<kelas>, ap_<kelas>.
    Nilai -1/NaN (tidak ada GT) dikembalikan sebagai NaN."""
    from pycocotools.cocoeval import COCOeval

    out = {"map50": 0.0, "map50_95": 0.0}
    for name in CLASS_IDS.values():
        out[f"ap50_{name}"] = 0.0
        out[f"ap_{name}"] = 0.0
    if not results:
        return out
    ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with ctx:
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.evaluate(); ev.accumulate(); ev.summarize()
    nan_if_neg = lambda v: float("nan") if v < 0 else float(v)
    out["map50_95"], out["map50"] = nan_if_neg(ev.stats[0]), nan_if_neg(ev.stats[1])
    prec = ev.eval["precision"]            # [T(IoU), R(recall), K(kelas), A(area), M(maxDets)]
    for k, cid in enumerate(ev.params.catIds):
        name = CLASS_IDS[cid]
        p50 = prec[0, :, k, 0, 2]          # IoU=0.5, area=all, maxDets=100
        pall = prec[:, :, k, 0, 2]
        out[f"ap50_{name}"] = float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")
        out[f"ap_{name}"] = float(pall[pall > -1].mean()) if (pall > -1).any() else float("nan")
    return out


def platelet_count_mae(gt, results, img_ids, score_thr=0.5, cat_id=PLATELET_ID):
    """MAE jumlah platelet per citra: |#prediksi (skor >= score_thr) - #GT|.
    DEFINISI SEMENTARA: samakan ambang & rumus dengan skrip evaluasi Anggota 4."""
    gt_n = {i: 0 for i in img_ids}
    pr_n = {i: 0 for i in img_ids}
    for a in gt.dataset["annotations"]:
        if a["category_id"] == cat_id and a["image_id"] in gt_n:
            gt_n[a["image_id"]] += 1
    for r in results:
        if r["category_id"] == cat_id and r["score"] >= score_thr and r["image_id"] in pr_n:
            pr_n[r["image_id"]] += 1
    return float(np.mean([abs(pr_n[i] - gt_n[i]) for i in img_ids]))


@torch.no_grad()
def measure_latency_ms(model, dataset, device, n=50, warmup=5):
    """Rata-rata ms/citra: batch=1, fp32, termasuk resize di model + postprocess,
    tidak termasuk pembacaan disk. Sinkronisasi CUDA sebelum/sesudah pengukuran."""
    model.eval()
    n = min(n, len(dataset))
    imgs = [dataset[i][0].to(device) for i in range(n)]
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for im in imgs[:min(warmup, n)]:
        model([im])
    sync()
    t0 = time.perf_counter()
    for im in imgs:
        model([im])
    sync()
    return (time.perf_counter() - t0) / n * 1000.0
