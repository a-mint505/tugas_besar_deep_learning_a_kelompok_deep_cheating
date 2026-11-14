# Asal data

- Dataset: TXL-PBC
- Sumber: https://github.com/lugan113/TXL-PBC_Dataset
- Diperoleh: 14 November 2026, via git clone
- Commit: <hash dari git rev-parse HEAD; hapus baris ini jika memakai ZIP>
- Lisensi repo: MIT, Copyright (c) 2024 lugan (lihat file `LICENSE` di repo sumber)
- Catatan lisensi: MIT berlaku untuk repo TXL-PBC. Citra aslinya berasal dari BCCD, BCD, PBC, dan Raabin-WBC yang punya ketentuan sendiri; belum diverifikasi. Dataset tidak didistribusikan ulang dalam repo ini (diunduh lewat skrip).
- Sitasi: Gan & Li, "TXL-PBC: a freely accessible labeled peripheral blood cell dataset", arXiv:2407.13214

## Isi dataset (hasil `scripts/validate_dataset.py`)

- Jumlah: 1.260 citra, 18.143 kotak (train 882 / val 252 / test 126)
- Urutan kelas dari `data.yaml`: 0 = WBC, 1 = RBC, 2 = Platelets
- `example.png` (1119x940, tanpa label) dikeluarkan karena gambar contoh pada repo sumber, bukan bagian dataset
- Catatan: ringkasan makalah menyebut pembagian 1008/288/144 (= 1.440), tidak sesuai isi repo; angka di atas dihitung langsung dari repo