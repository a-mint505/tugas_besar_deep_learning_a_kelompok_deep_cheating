import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
EXCLUDE_STEMS = {"example"}
SPLIT_NAMES = {"train": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test"}


def get_split(p: Path) -> str:
    for part in p.parts:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return "unknown"


def find_label(img: Path):
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        cand = Path(*parts).with_suffix(".txt")
        if cand.exists():
            return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def main(root: str, out: str = "logs/dataset_validation.json"):
    root = Path(root)
    images = sorted(p for p in root.rglob("*")
        if p.suffix.lower() in IMG_EXT
        and p.stem.lower() not in EXCLUDE_STEMS)

    stats = defaultdict(lambda: {"images": 0, "boxes": 0,
                                 "classes": Counter(), "no_label": 0,
                                 "empty_label": 0})
    sizes = Counter()
    problems = []
    names_by_split = defaultdict(set)

    for img in images:
        split = get_split(img)
        s = stats[split]
        s["images"] += 1
        names_by_split[split].add(img.stem)

        try:
            with Image.open(img) as im:
                sizes[im.size] += 1
        except Exception as e:
            problems.append(f"citra rusak: {img} ({e})")

        lab = find_label(img)
        if lab is None:
            s["no_label"] += 1
            problems.append(f"tanpa label: {img}")
            continue

        lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            s["empty_label"] += 1
            continue

        for ln, cols in enumerate(lines, 1):
            if len(cols) != 5:
                problems.append(f"{lab}:{ln} kolom != 5 -> {cols}")
                continue
            try:
                c = int(cols[0])
                x, y, w, h = map(float, cols[1:])
            except ValueError:
                problems.append(f"{lab}:{ln} nilai bukan angka -> {cols}")
                continue
            if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                problems.append(f"{lab}:{ln} koordinat di luar 0-1 / w,h<=0")
            s["boxes"] += 1
            s["classes"][c] += 1

    # nama file yang sama muncul di lebih dari satu split (petunjuk awal saja)
    overlap = {}
    keys = list(names_by_split)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            common = names_by_split[keys[i]] & names_by_split[keys[j]]
            if common:
                overlap[f"{keys[i]}&{keys[j]}"] = sorted(common)[:20]

    report = {
        "total_images": len(images),
        "total_boxes": sum(s["boxes"] for s in stats.values()),
        "per_split": {k: {**v, "classes": dict(sorted(v["classes"].items()))}
                      for k, v in stats.items()},
        "image_sizes": {f"{w}x{h}": n for (w, h), n in sizes.most_common()},
        "filename_overlap_between_splits": overlap,
        "n_problems": len(problems),
        "problems_first_50": problems[:50],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/txl_pbc")