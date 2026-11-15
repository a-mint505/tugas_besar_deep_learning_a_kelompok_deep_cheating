"""Split bersih dari logs/leakage_auto_pairs.csv (tanpa hitung ulang SIFT).
  python scripts/make_clean_from_auto.py        -> tabel sensitivitas ambang saja
  python scripts/make_clean_from_auto.py 30     -> tulis split bersih dengan ambang 30
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import read_split
from validate_dataset import find_label

ROOT = Path("data/raw/txl_pbc")
RANK = {"test": 2, "val": 1, "train": 0}
PAIRS = pd.read_csv("logs/leakage_auto_pairs.csv")
PATHS = {s: [p.as_posix() for p in read_split(s)] for s in ("train", "val", "test")}
_cls = {}


def boxes(path):
    if path not in _cls:
        _cls[path] = Counter(int(l.split()[0]) for l in
                             find_label(Path(path)).read_text().splitlines() if l.strip())
    return _cls[path]


def plan(T):
    flag = PAIRS[(PAIRS.inliers >= T) | (PAIRS.sumber == "manual")]
    drop = {}
    for _, r in flag.iterrows():
        a, b = (r["file_a"], r["split_a"]), (r["file_b"], r["split_b"])
        keep, rem = (a, b) if RANK[a[1]] >= RANK[b[1]] else (b, a)
        drop[rem[0]] = (rem[1], keep[0], int(r["inliers"]))
    return flag, drop


def affected(flag, q, pool):
    s = set(flag.loc[(flag.split_a == q) & (flag.split_b == pool), "file_a"])
    s |= set(flag.loc[(flag.split_b == q) & (flag.split_a == pool), "file_b"])
    return len(s)


def table():
    rows = []
    for T in (20, 25, 30, 40, 50, 61):
        flag, drop = plan(T)
        row = {"T": T, "pasangan": len(flag),
               "train-val": int((flag.pair == "train-val").sum()),
               "train-test": int((flag.pair == "train-test").sum()),
               "val-test": int((flag.pair == "val-test").sum())}
        for s in ("train", "val"):
            rem = [p for p in PATHS[s] if p in drop]
            row[f"{s}_dibuang"] = len(rem)
            row[f"{s}_platelet_hilang"] = sum(boxes(p)[2] for p in rem)
        rows.append(row)
    return pd.DataFrame(rows)


if len(sys.argv) < 2:
    print("Kelas 2 = Platelets (train asli: 882 citra, 382 platelet; val: 252 citra, 112 platelet)\n")
    print(table().to_string(index=False))
    sys.exit()

T = int(sys.argv[1])
flag, drop = plan(T)
summary = {"threshold": T, "pairs": len(flag),
           "pair_types": flag.pair.value_counts().to_dict(),
           "val_dengan_kembaran_train": affected(flag, "val", "train"),
           "test_dengan_kembaran_train": affected(flag, "test", "train"),
           "test_dengan_kembaran_val": affected(flag, "test", "val"), "dibuang": {}}

for s in ("train", "val"):
    kept = [p for p in PATHS[s] if p not in drop]
    rem = [p for p in PATHS[s] if p in drop]
    Path("splits", f"{s}_clean.txt").write_text(
        "\n".join(Path(p).relative_to(ROOT).as_posix() for p in kept) + "\n", newline="\n")
    cls = Counter()
    for p in rem:
        cls += boxes(p)
    summary["dibuang"][s] = {"asal": len(PATHS[s]), "sisa": len(kept), "dibuang": len(rem),
                             "kotak_hilang_per_kelas": dict(sorted(cls.items()))}

pd.DataFrame([{"file": f, "split": s, "kembaran": k, "inlier": i}
              for f, (s, k, i) in drop.items()]).to_csv("logs/leakage_removed.csv", index=False)
Path("logs/leakage_final_summary.json").write_text(json.dumps(summary, indent=2))

kept_sets = {s: set(PATHS[s]) - set(drop) for s in ("train", "val")}
kept_sets["test"] = set(PATHS["test"])
for _, r in flag.iterrows():
    assert not (r["file_a"] in kept_sets[r["split_a"]] and r["file_b"] in kept_sets[r["split_b"]])

print(json.dumps(summary, indent=2))
print("\nOK: tidak ada pasangan terflag yang tersisa lintas split. test.txt tidak diubah.")