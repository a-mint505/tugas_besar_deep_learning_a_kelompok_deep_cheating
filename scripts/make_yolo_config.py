"""Buat daftar citra dan YAML dataset untuk Ultralytics YOLOv8 dari split bersih.
Path ditulis absolut untuk mesin ini, jadi jalankan ulang di setiap mesin (laptop, Colab).

  python scripts/make_yolo_config.py
"""
import argparse
from pathlib import Path

import yaml

from validate_dataset import find_label

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="data/raw/txl_pbc")
p.add_argument("--out", default="data/processed/yolo")
args = p.parse_args()

ROOT = Path(args.data_root)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def class_names():
    f = next(ROOT.rglob("data.yaml"))
    names = yaml.safe_load(f.read_text())["names"]
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def write_list(name):
    src = Path("splits") / f"{name}.txt"
    if not src.exists():
        return None
    rels = [ln.strip() for ln in src.read_text().splitlines() if ln.strip()]
    paths = [(ROOT / r).resolve() for r in rels]
    missing = [q for q in paths if not q.exists() or find_label(q) is None]
    assert not missing, f"{name}: {len(missing)} citra/label tidak ditemukan, mis. {missing[:3]}"
    dst = (OUT / f"{name}.txt").resolve()
    dst.write_text("\n".join(q.as_posix() for q in paths) + "\n")
    print(f"{name}: {len(paths)} citra -> {dst}")
    return dst.as_posix()


names = class_names()
lists = {n: write_list(n) for n in ("train_clean", "val_clean", "test", "test_unflagged")}
assert all(lists[n] for n in ("train_clean", "val_clean", "test")), "daftar split utama tidak lengkap"


def write_yaml(fname, test_key):
    cfg = {"train": lists["train_clean"], "val": lists["val_clean"],
           "test": lists[test_key], "names": names}
    path = OUT / fname
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("yaml:", path.resolve())


write_yaml("txl_pbc_clean.yaml", "test")
if lists["test_unflagged"]:
    write_yaml("txl_pbc_clean_test_unflagged.yaml", "test_unflagged")
print("kelas:", names)