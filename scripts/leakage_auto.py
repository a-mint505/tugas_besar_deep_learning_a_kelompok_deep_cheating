"""Deteksi otomatis citra hampir-duplikat lintas split + pembuatan split bersih.
Tahap 1: kandidat dari korelasi thumbnail (8 transformasi) + hash yang sudah ada.
Tahap 2: verifikasi SIFT + RANSAC (transformasi similaritas, dengan versi cermin).
Ambang dikalibrasi dari pasangan acak (negatif) dan pasangan positif yang sudah dikonfirmasi."""
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from utils import read_split
from validate_dataset import find_label

SPLITS = ["train", "val", "test"]
ROOT = Path("data/raw/txl_pbc")
LOG = Path("logs")
TOPK, N_NEG, MIN_T, LONG_SIDE = 20, 3000, 10, 512
RANK = {"test": 2, "val": 1, "train": 0}
RNG = np.random.default_rng(42)

sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.02)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
bf = cv2.BFMatcher(cv2.NORM_L2)
_cache = {}


def thumb(p, size=32):
    return np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR), np.float32)


def norm(x):
    x = x.reshape(len(x), -1)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def feats(path):
    if path not in _cache:
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        s = LONG_SIDE / max(g.shape)
        g = clahe.apply(cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)),
                                   interpolation=cv2.INTER_CUBIC))
        out = []
        for img in (g, cv2.flip(g, 1)):          # asli dan cermin
            kp, des = sift.detectAndCompute(img, None)
            pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
            out.append((pts, None if des is None else des.astype(np.uint8)))
        _cache[path] = out
    return _cache[path]


def inliers(pa_path, pb_path):
    pa, da = feats(pa_path)[0]
    if da is None or len(da) < 8:
        return 0
    da = da.astype(np.float32)
    best = 0
    for pb, db in feats(pb_path):
        if db is None or len(db) < 8:
            continue
        m = bf.knnMatch(da, db.astype(np.float32), k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < 6:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if mask is not None:
            best = max(best, int(mask.sum()))
    return best


def main():
    paths, splits = [], []
    for s in SPLITS:
        ps = [p.as_posix() for p in read_split(s)]
        paths += ps
        splits += [s] * len(ps)
    n, sp = len(paths), np.array(splits)
    pos = {p: i for i, p in enumerate(paths)}
    idx = {s: np.where(sp == s)[0] for s in SPLITS}
    print(f"{n} citra")

    # --- tahap 1: kandidat ---
    th = np.stack([thumb(p) for p in paths])
    base = norm(th)
    C = np.full((n, n), -1.0, np.float32)
    for k in range(4):
        for flip in (False, True):
            t = np.rot90(th[:, :, ::-1] if flip else th, k, axes=(1, 2))
            C = np.maximum(C, norm(np.ascontiguousarray(t)) @ base.T)

    cand = set()

    def add(i, j):
        cand.add((int(min(i, j)), int(max(i, j))))

    for q, pool in (("val", "train"), ("test", "train"), ("test", "val")):
        P = idx[pool]
        for i in idx[q]:
            for j in P[np.argsort(-C[i, P])[:TOPK]]:
                add(i, j)

    if (LOG / "leakage_candidates.csv").exists():
        old = pd.read_csv(LOG / "leakage_candidates.csv")
        for a, b, sa, sb in zip(old.file_a, old.file_b, old.split_a, old.split_b):
            if sa != sb and a in pos and b in pos:
                add(pos[a], pos[b])

    manual = set()
    if (LOG / "leakage_confirmed.csv").exists():
        conf = pd.read_csv(LOG / "leakage_confirmed.csv", keep_default_na=False)
        for a, b, v in zip(conf.file_a, conf.file_b, conf.verdict):
            if v == "sama" and a in pos and b in pos:
                manual.add((min(pos[a], pos[b]), max(pos[a], pos[b])))
    cand |= manual
    print(f"kandidat pasangan lintas split: {len(cand)} (positif manual: {len(manual)})")

    # --- tahap 2: verifikasi ---
    neg = set()
    while len(neg) < N_NEG:
        i, j = (int(v) for v in RNG.integers(0, n, 2))
        if sp[i] != sp[j] and (min(i, j), max(i, j)) not in cand:
            neg.add((min(i, j), max(i, j)))

    def score(pairs, label):
        res = []
        for k, (a, b) in enumerate(sorted(pairs), 1):
            res.append(inliers(paths[a], paths[b]))
            if k % 500 == 0:
                print(f"  {label}: {k}/{len(pairs)}")
        return np.array(res)

    cand_sorted = sorted(cand)
    cand_in = score(cand, "kandidat")
    neg_in = score(neg, "negatif acak")

    neg_max = int(neg_in.max())
    T = max(MIN_T, neg_max + 1)
    bins = [0, 5, 10, 15, 25, 50, 10 ** 6]
    print("\n== Kalibrasi (jumlah inlier SIFT/RANSAC) ==")
    print("kandidat      :", np.histogram(cand_in, bins)[0].tolist(), "bin", bins[:-1])
    print("negatif acak  :", np.histogram(neg_in, bins)[0].tolist(), f"| maks={neg_max}, p99.9={np.percentile(neg_in, 99.9):.1f}")
    print(f"AMBANG T = {T} (pasangan 'sama' jika inlier >= T)")
    man_in = [int(cand_in[cand_sorted.index(p)]) for p in sorted(manual)]
    caught = sum(v >= T for v in man_in)
    print(f"positif yang sudah dikonfirmasi: inlier {man_in} -> tertangkap otomatis {caught}/{len(man_in)}")
    if caught < len(man_in):
        print("PERINGATAN: detektor melewatkan positif yang diketahui; ambang atau metode perlu ditinjau.")

    rows = []
    for k, ((a, b), inl) in enumerate(zip(cand_sorted, cand_in), 1):
        is_man = (a, b) in manual
        rows.append({"no": k, "file_a": paths[a], "split_a": sp[a], "file_b": paths[b], "split_b": sp[b],
                     "pair": f"{sp[a]}-{sp[b]}", "corr": round(float(C[a, b]), 3), "inliers": int(inl),
                     "verdict": "sama" if (inl >= T or is_man) else "beda",
                     "sumber": "manual" if is_man else "otomatis"})
    df = pd.DataFrame(rows)
    df.to_csv(LOG / "leakage_auto_pairs.csv", index=False)
    same = df[df.verdict == "sama"]
    print("\n== Pasangan 'sama' per jenis ==\n", same.pair.value_counts().to_string())
    print(f"pasangan dengan inlier di sekitar ambang ({T}-{T + 4}): {int(((df.inliers >= T) & (df.inliers < T + 5)).sum())}")

    def n_aff(q, pool):
        s = set(same.loc[(same.split_a == q) & (same.split_b == pool), "file_a"])
        s |= set(same.loc[(same.split_b == q) & (same.split_a == pool), "file_b"])
        return len(s)

    aff = {"val_dengan_kembaran_train": n_aff("val", "train"),
           "test_dengan_kembaran_train": n_aff("test", "train"),
           "test_dengan_kembaran_val": n_aff("test", "val")}
    print("\n", aff)

    # --- pembersihan: buang yang peringkat split-nya lebih rendah ---
    drop = {}
    for _, r in same.iterrows():
        a, b = (r["file_a"], r["split_a"]), (r["file_b"], r["split_b"])
        keep, rem = (a, b) if RANK[a[1]] >= RANK[b[1]] else (b, a)
        drop[rem[0]] = (rem[1], keep[0])

    summary = {"threshold": T, "neg_max": neg_max, "n_candidates": len(cand),
               "n_same_pairs": int(len(same)), "affected": aff, "dibuang": {}}
    for s in ("train", "val"):
        ps = read_split(s)
        kept = [p for p in ps if p.as_posix() not in drop]
        removed = [p for p in ps if p.as_posix() in drop]
        Path("splits", f"{s}_clean.txt").write_text(
            "\n".join(p.relative_to(ROOT).as_posix() for p in kept) + "\n", newline="\n")
        cls = Counter(int(ln.split()[0]) for p in removed
                      for ln in find_label(p).read_text().splitlines() if ln.strip())
        summary["dibuang"][s] = {"asal": len(ps), "sisa": len(kept), "dibuang": len(removed),
                                 "kotak_per_kelas_hilang": dict(sorted(cls.items()))}
        print(f"{s}: {len(ps)} -> {len(kept)} (dibuang {len(removed)}), kotak hilang per kelas: {dict(sorted(cls.items()))}")
    pd.DataFrame([{"file": f, "split": s, "kembaran": k} for f, (s, k) in drop.items()]
                 ).to_csv(LOG / "leakage_removed.csv", index=False)
    (LOG / "leakage_auto_summary.json").write_text(json.dumps(summary, indent=2))

    members = {"train": {p for p in paths if p.split("/")[-2] == "train"}, "val": set(), "test": set()}
    kept_sets = {s: {p.as_posix() for p in read_split(s)} - set(drop) for s in ("train", "val")}
    kept_sets["test"] = {p.as_posix() for p in read_split("test")}
    for _, r in same.iterrows():
        assert not (r["file_a"] in kept_sets[r["split_a"]] and r["file_b"] in kept_sets[r["split_b"]])
    print("\nOK: tidak ada pasangan 'sama' yang tersisa lintas split. test.txt tidak diubah.")


if __name__ == "__main__":
    main()