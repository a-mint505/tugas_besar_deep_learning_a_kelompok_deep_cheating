import os
import random
from pathlib import Path

RUN_FIELDS = [
    "run_id", "date", "member", "model", "purpose", "ablation_factor", "factor_value",
    "seed", "train_split", "val_split", "imgsz_or_min_size", "epochs", "batch_size", "lr",
    "pretrained", "env", "config_path", "commit",
    "val_map50", "val_map50_95", "val_ap50_platelet", "val_mae_platelet",
    "params_m", "latency_ms", "status", "notes",
]


def log_run(member: str, **row):
    """Tambah satu baris ke logs/runs_<member>.csv (dibuat jika belum ada).
    purpose: baseline | main | ablation | tuning; status: ok | failed | diverged | partial."""
    import csv
    from datetime import date

    path = Path("logs") / f"runs_{member}.csv"
    path.parent.mkdir(exist_ok=True)
    row.setdefault("date", date.today().isoformat())
    row["member"] = member
    unknown = set(row) - set(RUN_FIELDS)
    assert not unknown, f"kolom tidak dikenal: {unknown}"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Atur seed untuk Python, NumPy, dan PyTorch (jika terpasang).

    Panggil sekali di awal skrip pelatihan, sebelum membuat model/DataLoader.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Dibutuhkan agar operasi cuBLAS deterministik di GPU
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return seed  # lingkungan tanpa torch (skrip data/EDA di laptop)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # aman dipanggil walau tanpa GPU
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: operasi tanpa versi deterministik hanya memberi
        # peringatan, tidak membuat pelatihan berhenti dengan error
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def seed_worker(worker_id: int) -> None:
    """Seragamkan seed di tiap worker DataLoader."""
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def detection_collate(batch):
    """Collate untuk detektor torchvision (jumlah kotak per citra berbeda)."""
    return tuple(zip(*batch))


def make_loader(dataset, seed: int, batch_size: int = 8, shuffle: bool = True,
                num_workers: int = 2, collate_fn=detection_collate):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=g, collate_fn=collate_fn)


def read_split(name: str, data_root: str = "data/raw/txl_pbc",
               splits_dir: str = "splits") -> list[Path]:
    """Kembalikan daftar path citra untuk split 'train', 'val', atau 'test'."""
    lines = (Path(splits_dir) / f"{name}.txt").read_text().splitlines()
    return [Path(data_root) / ln.strip() for ln in lines if ln.strip()]