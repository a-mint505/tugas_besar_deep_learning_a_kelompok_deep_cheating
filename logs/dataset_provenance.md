# Asal data

- Dataset: TXL-PBC
- Sumber: https://github.com/lugan113/TXL-PBC_Dataset
- Diperoleh: 14 November 2026, via git clone
- Commit: fd9aadd9636d590ce5b25a1936e61bff4e581a48
- Lisensi repo: MIT, Copyright (c) 2024 lugan (lihat file `LICENSE` di repo sumber)
- Catatan lisensi: MIT berlaku untuk repo TXL-PBC. Citra aslinya berasal dari BCCD, BCDD, PBC, dan Raabin-WBC yang punya ketentuan sendiri; belum diverifikasi. Dataset tidak didistribusikan ulang dalam repo ini (diunduh lewat skrip).
- Sitasi: Gan & Li, "TXL-PBC: a freely accessible labeled peripheral blood cell dataset", arXiv:2407.13214

## Isi dataset (hasil `scripts/validate_dataset.py`)

- Jumlah: 1.260 citra, 18.143 kotak (train 882 / val 252 / test 126)
- Urutan kelas dari `data.yaml`: 0 = WBC, 1 = RBC, 2 = Platelets
- `example.png` (1119x940, tanpa label) dikeluarkan karena gambar contoh pada repo sumber, bukan bagian dataset
- Catatan: ringkasan makalah menyebut pembagian 1008/288/144 (= 1.440), tidak sesuai isi repo; angka di atas dihitung langsung dari repo

## Pemeriksaan kebocoran

- MD5: 0 kelompok file identik.
- Kandidat: 17.923 pasangan lintas split dari korelasi thumbnail (8 transformasi), pHash/dHash 64-bit, dan 6 pasangan hasil konfirmasi visual awal. Diverifikasi dengan pencocokan fitur SIFT + RANSAC (transformasi similaritas, termasuk versi cermin).
- Kalibrasi: 3.000 pasangan acak lintas split menghasilkan maksimum 60 inlier (persentil 99,9 = 53). Pada T=61 hanya 4 dari 6 pasangan hasil konfirmasi visual tertangkap (inlier terendah 31 dan 60), sehingga ambang diturunkan ke T=30 (6/6 tertangkap; 31 adalah inlier terendah yang dikonfirmasi). Pertimbangan: membuang citra latih keliru lebih murah daripada membiarkan kebocoran.
- Hasil (T=30): 124 pasangan terflag (train-val 70, train-test 39, val-test 15). Citra val dengan kembaran terflag di train: 24; test dengan kembaran di train: 17; test dengan kembaran di val: 7.
- Penanganan: train_clean.txt (787 dari 882; 95 dibuang), val_clean.txt (239 dari 252; 13 dibuang), test.txt tidak diubah. Platelet yang hilang: train 17 dari 382, val 1 dari 112. Daftar yang dibuang: logs/leakage_removed.csv. Citra test terflag dicatat di splits/test_flagged.txt untuk analisis sensitivitas.
- Keterbatasan: presisi detektor tidak terukur. Pada rentang inlier 25-50, jumlah pasangan kandidat tidak melebihi yang diharapkan dari pasangan acak (126 vs sekitar 161), sehingga sebagian pasangan terflag mungkin bukan duplikat nyata; bukti kuat hanya pada inlier tinggi. Kandidat hanya 20 tetangga terdekat per citra val/test, jadi duplikat di luar itu tidak diperiksa. Citra buram menghasilkan sedikit titik fitur. Hash dan fitur lokal tidak dapat mendeteksi apusan atau pasien yang sama dengan bidang pandang berbeda. Hanya pasangan lintas split yang diperiksa.