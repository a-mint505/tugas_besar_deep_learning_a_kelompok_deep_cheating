import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Atur seed untuk Python, NumPy, dan PyTorch (CPU dan GPU).

    Panggil sekali di awal skrip pelatihan, sebelum membuat model/DataLoader.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Dibutuhkan agar operasi cuBLAS deterministik di GPU
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def detection_collate(batch):
    """Collate untuk detektor torchvision (jumlah kotak per citra berbeda)."""
    return tuple(zip(*batch))


def make_loader(dataset, seed: int, batch_size: int = 8, shuffle: bool = True,
                num_workers: int = 2, collate_fn=detection_collate) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=g, collate_fn=collate_fn)
    
set_seed(42); a = torch.rand(3)
set_seed(42); b = torch.rand(3)
assert torch.equal(a, b)