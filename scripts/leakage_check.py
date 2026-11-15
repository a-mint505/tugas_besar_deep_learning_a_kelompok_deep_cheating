"""Cek duplikat / near-duplicate antar-split: MD5 (identik byte) + pHash & dHash
(dengan 8 transformasi rotasi/flip)."""
import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from utils import read_split

SPLITS = ["train", "val", "test"]
REPORT_MAX_DIST = 16          # pasangan dengan jarak pHash <= ini dicatat untuk diperiksa
LOG_DIR, FIG_DIR = Path("logs"), Path("figures")

T = Image.Transpose
TRANSFORMS = [
    ("id", lambda im: im),
    ("rot90", lambda im: im.transpose(T.ROTATE_90)),
    ("rot180", lambda im: im.transpose(T.ROTATE_180)),
    ("rot270", lambda im: im.transpose(T.ROTATE_270)),
    ("flip_lr", lambda im: im.transpose(T.FLIP_LEFT_RIGHT)),
    ("flip_tb", lambda im: im.transpose(T.FLIP_TOP_BOTTOM)),
    ("transpose", lambda im: im.transpose(T.TRANSPOSE)),
    ("transverse", lambda im: im.transpose(T.TRANSVERSE)),
]
TF = dict(TRANSFORMS)


def hash_all(paths):
    n, nt = len(paths), len(TRANSFORMS)
    P = np.zeros((nt, n, 64), np.float32)
    D = np.zeros((nt, n, 64), np.float32)
    md5 = []
    for i, p in enumerate(paths):
        md5.append(hashlib.md5(p.read_bytes()).hexdigest())
        with Image.open(p) as im:
            im = im.convert("RGB")
            for t, (_, f) in enumerate(TRANSFORMS):
                x = f(im)
                P[t, i] = imagehash.phash(x).hash.ravel()
                D[t, i] = imagehash.dhash(x).hash.ravel()
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n}")
    return P, D, md5


def dist(A, B):
    """Jarak Hamming antar semua pasangan baris (A, B bernilai 0/1)."""
    return 64 - (A @ B.T + (1 - A) @ (1 - B).T)


def pair_sheet(rows, path, start=1, thumb=256):
    if rows.empty:
        return
    tiles = []
    for num, (_, r) in enumerate(rows.iterrows(), start=start):
        a = TF[r["transform"]](Image.open(r["file_a"]).convert("RGB"))
        b = Image.open(r["file_b"]).convert("RGB")
        a.thumbnail((thumb, thumb))
        b.thumbnail((thumb, thumb))
        tile = Image.new("RGB", (2 * thumb, thumb + 16), "black")
        tile.paste(a, (0, 16))
        tile.paste(b, (thumb, 16))
        ImageDraw.Draw(tile).text(
            (2, 2),
            f'#{num} {r["split_a"]}|{r["split_b"]}  dP={r["d_phash"]} dD={r["d_dhash"]}  a:{r["transform"]}',
            fill="white")
        tiles.append(tile)
    cols = 2
    nrows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 2 * thumb, nrows * (thumb + 16)), "black")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * 2 * thumb, (k // cols) * (thumb + 16)))
    FIG_DIR.mkdir(exist_ok=True)
    sheet.save(path)
    print("gambar:", path)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    paths, splits = [], []
    for s in SPLITS:
        ps = read_split(s)
        paths += ps
        splits += [s] * len(ps)
    n = len(paths)
    sp = np.array(splits)
    print(f"{n} citra; menghitung hash (8 transformasi x pHash+dHash), tunggu beberapa menit...")
    P, D, md5 = hash_all(paths)
    nt = len(TRANSFORMS)

    # 1. duplikat identik byte
    df = pd.DataFrame({"file": [p.as_posix() for p in paths], "split": splits, "md5": md5})
    dup = df[df.duplicated("md5", keep=False)]
    n_groups = dup.md5.nunique()
    n_cross_groups = int((dup.groupby("md5").split.nunique() > 1).sum())
    print(f"\n== MD5 ==\nkelompok file identik: {n_groups} (lintas split: {n_cross_groups})")
    dup.sort_values("md5").to_csv(LOG_DIR / "leakage_exact_duplicates.csv", index=False)

    # 2. jarak pHash minimum atas 8 transformasi
    dp = np.stack([dist(P[t], P[0]) for t in range(nt)])      # (nt, n, n)
    dmin, tbest = dp.min(0), dp.argmin(0)
    del dp
    iu = np.triu_indices(n, 1)
    dsym = np.minimum(dmin, dmin.T)[iu]
    pct = np.percentile(dsym, [0.01, 0.1, 1, 5, 50])
    baseline = dict(zip(["p0.01", "p0.1", "p1", "p5", "p50"], pct.round(1).tolist()))
    print("\n== Distribusi jarak pHash seluruh pasangan (1 pasangan = 2 citra) ==")
    print(f"{len(dsym)} pasangan; persentil: {baseline}")

    # 3. pasangan kandidat + konfirmasi dHash
    m = dsym <= REPORT_MAX_DIST
    I, J = iu[0][m], iu[1][m]
    fwd = dmin[I, J] <= dmin[J, I]
    A, B = np.where(fwd, I, J), np.where(fwd, J, I)
    Tb = tbest[A, B]
    dph = dmin[A, B]
    dd = 64 - ((D[Tb, A] * D[0, B]).sum(1) + ((1 - D[Tb, A]) * (1 - D[0, B])).sum(1))
    cand = pd.DataFrame({
        "file_a": [paths[a].as_posix() for a in A], "split_a": sp[A],
        "file_b": [paths[b].as_posix() for b in B], "split_b": sp[B],
        "transform": [TRANSFORMS[t][0] for t in Tb],
        "d_phash": dph.astype(int), "d_dhash": dd.astype(int)})
    cand["pair"] = ["-".join(sorted(x)) for x in zip(cand.split_a, cand.split_b)]
    cand = cand.sort_values(["d_phash", "d_dhash"]).reset_index(drop=True)
    cand.to_csv(LOG_DIR / "leakage_candidates.csv", index=False)

    # 4. tabel kumulatif per jenis pasangan
    pair_types = ["train-train", "val-val", "test-test", "train-val", "test-train", "test-val"]
    rows = []
    for thr in [0, 2, 4, 6, 8, 10, 12]:
        c = cand[cand.d_phash <= thr]
        cross = c[c.split_a != c.split_b]
        in_test = set(cross.file_a[cross.split_a == "test"]) | set(cross.file_b[cross.split_b == "test"])
        row = {"d_phash<=": thr}
        row.update({k: int((c.pair == k).sum()) for k in pair_types})
        row["test_img_dengan_pasangan_lintas_split"] = len(in_test)
        rows.append(row)
    table = pd.DataFrame(rows)
    print("\n== Jumlah pasangan kandidat (kumulatif) ==\n", table.to_string(index=False))
    table.to_csv(LOG_DIR / "leakage_summary_table.csv", index=False)

    # 5. gambar untuk pemeriksaan visual
    cross = cand[cand.split_a != cand.split_b]
    review = (cross[(cross.d_phash <= 16) & (cross.d_dhash <= 16)]
              .sort_values(["d_dhash", "d_phash"]).reset_index(drop=True))
    review.index = review.index + 1
    review.to_csv(LOG_DIR / "leakage_review_list.csv", index_label="no")
    print(f"pasangan lintas split untuk ditinjau (dP<=16 dan dD<=16): {len(review)}")
    for k in range(0, len(review), 12):
        pair_sheet(review.iloc[k:k + 12], FIG_DIR / f"leakage_review_{k // 12 + 1}.png", start=k + 1)
        
    (LOG_DIR / "leakage_summary.json").write_text(json.dumps({
        "n_images": n, "md5_groups": int(n_groups), "md5_groups_cross_split": n_cross_groups,
        "phash_baseline_percentiles": baseline, "report_max_dist": REPORT_MAX_DIST,
        "n_candidates_total": int(len(cand)), "n_candidates_cross_split": int(len(cross))}, indent=2))
    print("\nSelesai. Periksa logs/leakage_* dan figures/leakage_*.png")


if __name__ == "__main__":
    main()