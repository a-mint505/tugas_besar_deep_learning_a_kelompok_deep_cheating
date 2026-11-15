"""Konversi anotasi YOLO (TXL-PBC) ke COCO JSON berdasarkan splits/*.txt,
lalu buat contact sheet untuk pemeriksaan visual."""
import json
import random
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from utils import read_split
from validate_dataset import find_label

ROOT = Path("data/raw/txl_pbc")
OUT_DIR = Path("data/processed/coco")
FIG_PATH = Path("figures/conversion_check.png")
SPLITS = ["train", "val", "test"]
CLASS_NAMES = None   # isi manual kalau data.yaml tidak ditemukan, mis. ["WBC", "RBC", "Platelet"]
COLORS = {1: "red", 2: "lime", 3: "cyan", 4: "yellow"}


def load_class_names():
    if CLASS_NAMES:
        return list(CLASS_NAMES)
    yamls = list(ROOT.rglob("data.yaml"))
    if not yamls:
        raise FileNotFoundError("data.yaml tidak ditemukan; isi CLASS_NAMES manual.")
    names = yaml.safe_load(yamls[0].read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def yolo_to_coco(split, class_names):
    images, annotations = [], []
    ann_id, n_yolo, skipped, clipped = 1, 0, 0, 0

    for img_id, img_path in enumerate(read_split(split, str(ROOT)), start=1):
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": img_id,
                       "file_name": img_path.relative_to(ROOT).as_posix(),
                       "width": W, "height": H})

        for line in find_label(img_path).read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            c = int(c)
            cx, cy, w, h = (float(v) for v in (cx, cy, w, h))
            n_yolo += 1

            rx1, ry1 = (cx - w / 2) * W, (cy - h / 2) * H
            rx2, ry2 = (cx + w / 2) * W, (cy + h / 2) * H
            x1, y1 = max(0.0, rx1), max(0.0, ry1)
            x2, y2 = min(float(W), rx2), min(float(H), ry2)
            if rx1 < -0.5 or ry1 < -0.5 or rx2 > W + 0.5 or ry2 > H + 0.5:
                clipped += 1

            bw, bh = x2 - x1, y2 - y1
            if bw < 1 or bh < 1:          # kotak degenerat (< 1 piksel)
                skipped += 1
                continue

            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": c + 1,     # 0 dipakai sebagai latar belakang
                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0})
            ann_id += 1

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]}
    return coco, n_yolo, skipped, clipped


def draw_sheet(cocos, n_random=12, n_platelet=12, seed=42, thumb=360, cols=4):
    rng = random.Random(seed)
    cat_names = {c["id"]: c["name"] for c in cocos["train"]["categories"]}
    platelet_ids = [i for i, n in cat_names.items() if "platelet" in n.lower()]

    items = []   # (split, image_dict, [anns])
    for split, coco in cocos.items():
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            items.append((split, im, by_img.get(im["id"], [])))

    has_plt = [it for it in items if any(a["category_id"] in platelet_ids for a in it[2])]
    if not platelet_ids:
        print("PERINGATAN: kelas 'platelet' tidak ditemukan di nama kelas.")
    picked = rng.sample(has_plt, min(n_platelet, len(has_plt))) if platelet_ids else []
    rest = [it for it in items if it not in picked]
    picked += rng.sample(rest, n_random)

    tiles = []
    for split, im, anns in picked:
        img = Image.open(ROOT / im["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for a in anns:
            x, y, w, h = a["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=COLORS.get(a["category_id"], "white"), width=2)
        d.text((4, 4), f'{split}/{Path(im["file_name"]).name}', fill="white")
        img.thumbnail((thumb, thumb))
        tiles.append(img)

    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * thumb, (i // cols) * thumb))
    FIG_PATH.parent.mkdir(exist_ok=True)
    sheet.save(FIG_PATH)
    print(f"contact sheet: {len(tiles)} citra -> {FIG_PATH}")
    print("legenda warna:", {COLORS.get(i): n for i, n in cat_names.items()})


def main():
    class_names = load_class_names()
    print("urutan kelas (YOLO id -> nama):", dict(enumerate(class_names)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cocos = {}
    for split in SPLITS:
        coco, n_yolo, skipped, clipped = yolo_to_coco(split, class_names)
        cocos[split] = coco
        (OUT_DIR / f"{split}.json").write_text(json.dumps(coco))

        n_coco = len(coco["annotations"])
        per_cls = Counter(a["category_id"] for a in coco["annotations"])
        print(f"{split}: {len(coco['images'])} citra, {n_coco} kotak "
              f"(YOLO {n_yolo}, dibuang {skipped}, terpotong {clipped}), per kelas: {dict(sorted(per_cls.items()))}")
        assert n_coco + skipped == n_yolo, f"{split}: jumlah kotak tidak cocok"

    draw_sheet(cocos)


if __name__ == "__main__":
    main()