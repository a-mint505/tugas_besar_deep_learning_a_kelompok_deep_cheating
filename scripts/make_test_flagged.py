"""Buat splits/test_flagged.txt dan test_unflagged.txt (ambang T), plus ringkasan tingkat bukti."""
from collections import Counter
from pathlib import Path

import pandas as pd

from utils import read_split
from validate_dataset import find_label

T = 30
ROOT = Path("data/raw/txl_pbc")
P = pd.read_csv("logs/leakage_auto_pairs.csv")
f = P[(P.inliers >= T) | (P.sumber == "manual")]

flagged = set(f.loc[f.split_a == "test", "file_a"]) | set(f.loc[f.split_b == "test", "file_b"])
test = [p.as_posix() for p in read_split("test")]
fl = [p for p in test if p in flagged]
un = [p for p in test if p not in flagged]


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def plt_count(ps):
    return sum(1 for p in ps for ln in find_label(Path(p)).read_text().splitlines()
               if ln.strip() and int(ln.split()[0]) == 2)


Path("splits/test_flagged.txt").write_text("\n".join(rel(p) for p in fl) + "\n", newline="\n")
Path("splits/test_unflagged.txt").write_text("\n".join(rel(p) for p in un) + "\n", newline="\n")
print(f"test: {len(test)} | terflag: {len(fl)} | tidak terflag: {len(un)}")
print(f"kotak platelet di test tidak terflag: {plt_count(un)} dari {plt_count(test)}")

tier = pd.cut(f.inliers, [0, 60, 90, 10**6], labels=["30-60 (lemah)", "61-90 (sedang)", ">90 (kuat)"], right=True)
print("\nTingkat bukti pasangan terflag per jenis:\n", pd.crosstab(f.pair, tier).to_string())