# Tugas Besar Deep Learning (CVL): T1-B, platelet sebagai objek kecil pada TXL-PBC

Kelompok: Deep Cheating. Anggota: M. Aminuddin Marhan (240441100018), Sahrul (240441100058), Moh. Fadil (240441100072), Alief Noor Aliansyah (240441100130).

## Ringkasan
Deteksi tiga kelas sel darah (WBC, RBC, platelet) pada TXL-PBC. Fokus T1-B: pengaruh resolusi, FPN, dan anchor terhadap deteksi platelet. Model: YOLOv8 (one-stage) dan Faster R-CNN ResNet-50 FPN (two-stage), keduanya berbobot pralatih.

## Struktur
- `scripts/`: unduh/validasi data, pembuatan split, konversi anotasi, EDA, audit kebocoran, utilitas (`utils.py`)
- `splits/`: daftar file split (`train_clean`, `val_clean`, `test`, `test_unflagged`; split resmi di `train/val/test`)
- `logs/`: log eksperimen per anggota (`runs_<nama>.csv`), `final_test_results.csv`, laporan validasi data dan audit kebocoran
- `configs/`, `notebooks/`, `figures/`
- `data/` tidak ada di repo (diunduh lewat skrip)

## Setup
Laptop (venv):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Colab: lihat sel pembuka di `notebooks/` (jangan menginstal ulang torch/torchvision bawaan Colab).

## Data
1. `git clone https://github.com/lugan113/TXL-PBC_Dataset data/raw/txl_pbc`, lalu `git -C data/raw/txl_pbc checkout fd9aadd9636d590ce5b25a1936e61bff4e581a48`
2. `python scripts/validate_dataset.py data/raw/txl_pbc`
3. `python scripts/convert_yolo_to_coco.py clean` (Faster R-CNN, hasil di `data/processed/coco_clean/`)
4. `python scripts/make_yolo_config.py` (YOLOv8, hasil di `data/processed/yolo/`; berisi path absolut, jalankan di setiap mesin)

Ringkasan data: 1.260 citra, 18.143 kotak. Split resmi 882/252/126. Split bersih setelah audit kebocoran: train 787, val 239, test 126 (tidak diubah). Detail audit di `logs/dataset_provenance.md`.

## Pemetaan kelas (penting)
- YOLO: 0 = WBC, 1 = RBC, 2 = Platelets
- COCO/torchvision: `category_id` = ID YOLO + 1 (0 = latar belakang). 1 = WBC, 2 = RBC, 3 = Platelets
- Samakan pemetaan ini sebelum menghitung metrik lintas model.

## Aturan penggunaan split
- Latih dan pilih model/hiperparameter hanya dengan `train_clean` dan `val_clean`.
- `test` dipakai **sekali**, untuk konfigurasi final, dan dilaporkan juga pada `test_unflagged`.
- Hasil uji hanya dicatat di `logs/final_test_results.csv`.

## Reproduksibilitas
- Seed: `set_seed()` dan `make_loader()` di `scripts/utils.py`; YOLOv8 memakai argumen seed sendiri. Perbandingan utama memakai 3 seed (<seed>).
- Hasil GPU tidak dijamin identik 100% antar-run walau seed sama.
- Lingkungan: Python 3.11.9, torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.215, GPU <jenis> (catat dari sesi Colab yang menghasilkan angka utama).
- Bobot final: <tautan unduh>

## Pembagian peran
<tabel dari lembar kontribusi>

## Penggunaan AI generatif
<satu paragraf, sesuai syarat dosen>

## Sitasi dan lisensi
Dataset: Gan & Li, "TXL-PBC: a freely accessible labeled peripheral blood cell dataset", arXiv:2407.13214. Lisensi repo dataset: MIT (lihat `logs/dataset_provenance.md`).