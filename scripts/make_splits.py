from pathlib import Path

from validate_dataset import IMG_EXT, EXCLUDE_STEMS, find_label, get_split

ROOT = Path("data/raw/txl_pbc")
OUT = Path("splits")
EXPECTED = {"train": 882, "val": 252, "test": 126}

OUT.mkdir(exist_ok=True)

groups = {"train": [], "val": [], "test": []}
for p in sorted(ROOT.rglob("*")):
    if p.suffix.lower() not in IMG_EXT or p.stem.lower() in EXCLUDE_STEMS:
        continue
    split = get_split(p)
    if split in groups and find_label(p) is not None:
        groups[split].append(p.relative_to(ROOT).as_posix())

for split, files in groups.items():
    # newline="\n": di Windows, tanpa ini file tertulis dengan CRLF
    (OUT / f"{split}.txt").write_text("\n".join(files) + "\n", newline="\n")
    print(f"{split}: {len(files)}")

# Pemeriksaan: jumlah sesuai hasil validasi, dan tidak ada citra di dua split
for split, n in EXPECTED.items():
    assert len(groups[split]) == n, f"{split}: {len(groups[split])} != {n}"
all_files = [f for fs in groups.values() for f in fs]
assert len(all_files) == len(set(all_files)), "ada citra di lebih dari satu split"
print("OK: jumlah sesuai dan tidak ada tumpang tindih path")

def read_split(name: str, data_root: str = "data/raw/txl_pbc",
               splits_dir: str = "splits") -> list[Path]:
    """Kembalikan daftar path citra untuk split 'train', 'val', atau 'test'."""
    lines = (Path(splits_dir) / f"{name}.txt").read_text().splitlines()
    return [Path(data_root) / ln.strip() for ln in lines if ln.strip()]