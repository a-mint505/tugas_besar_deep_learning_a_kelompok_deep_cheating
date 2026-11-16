"""Latih Faster R-CNN ResNet-50 FPN pada train_clean, evaluasi pada val_clean SAJA.

Contoh (dari akar repo, di Colab):
  python scripts/train_frcnn.py --smoke                              # uji cepat, tidak dicatat
  python scripts/train_frcnn.py --purpose baseline --tag b0 --seed 42 \
      --size-mode longest --size 640 --epochs 20 \
      --out-dir /content/drive/MyDrive/dl_tubes/anggota3/runs \
      --drive-dir /content/drive/MyDrive/dl_tubes/anggota3/logs
  python scripts/train_frcnn.py --purpose ablation --tag a_half --seed 42 \
      --ablation-factor anchor_sizes --anchor-sizes 16,32,64,128,256 ...

Skrip ini TIDAK menyentuh test. Semua nilai bawaan hiperparameter di bawah adalah pilihan
awal penulis (bukan hasil tuning); catat sebagai hiperparameter dan ubah lewat argumen.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from frcnn_lib import (COCO_DIR, DEFAULT_ANCHOR_SIZES, CocoDetDataset, build_model, coco_eval,
                       load_gt, measure_latency_ms, platelet_count_mae, predict, resolution_kwargs)
from utils import log_run, make_loader, set_seed

MEMBER = "anggota3"
MODEL_NAME = "faster_rcnn_r50_fpn"


class DivergedError(RuntimeError):
    pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", default="baseline", choices=["baseline", "main", "ablation", "tuning"])
    ap.add_argument("--tag", default="b0", help="nama pendek konfigurasi, masuk ke run_id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=100)
    ap.add_argument("--size-mode", default="longest", choices=["longest", "shortest"])
    ap.add_argument("--size", type=int, default=640, help="N untuk size-mode (longest: sisi terpanjang; shortest: min_size)")
    ap.add_argument("--anchor-sizes", default=None, help="mis. 16,32,64,128,256 (5 angka). Kosong = bawaan torchvision")
    ap.add_argument("--ablation-factor", default="", help="mis. anchor_sizes (hanya untuk purpose=ablation)")
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--hflip", type=float, default=0.5)
    ap.add_argument("--vflip", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true", help="campuran presisi (fp16) saat latihan")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--select-metric", default="map50_95", choices=["map50_95", "map50", "ap50_Platelets"])
    ap.add_argument("--eval-score-thresh", type=float, default=0.001,
                    help="roi_heads.score_thresh saat evaluasi. Bawaan torchvision 0.05 (seingat saya); "
                         "0.001 dipilih agar kurva PR tidak terpotong (samakan dengan Anggota 4)")
    ap.add_argument("--mae-score-thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default="runs_local", help="checkpoint/history/config. Di Colab: folder di Drive")
    ap.add_argument("--drive-dir", default=None, help="salin logs/runs_anggota3.csv ke sini setelah run")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="16 citra latih, 8 val, 1 epoch, TIDAK dicatat ke log")
    return ap.parse_args()


def env_string():
    import torchvision
    try:
        import pycocotools
        pc = getattr(pycocotools, "__version__", "?")
    except ImportError:
        pc = "tidak terpasang"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return (f"python {platform.python_version()}; torch {torch.__version__}; torchvision {torchvision.__version__}; "
            f"pycocotools {pc}; gpu {gpu}")


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "-uno"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def save_atomic(obj, path: Path):
    """Tulis ke file sementara lalu ganti nama, supaya sesi Colab yang putus tidak merusak checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_scaler(device, enabled):
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):       # torch lama
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.tag = 1, "smoke"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_sizes = tuple(int(x) for x in args.anchor_sizes.split(",")) if args.anchor_sizes else None
    shown_anchor = anchor_sizes or DEFAULT_ANCHOR_SIZES
    run_id = f"frcnn_{args.purpose}_{args.tag}_s{args.seed}"
    out = Path(args.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.json"

    row = dict(run_id=run_id, model=MODEL_NAME, purpose=args.purpose,
               ablation_factor=args.ablation_factor,
               factor_value="-".join(map(str, shown_anchor)) if args.ablation_factor == "anchor_sizes" else "",
               seed=args.seed, train_split="train_clean", val_split="val_clean",
               imgsz_or_min_size=f"{args.size_mode}{args.size}", epochs=args.epochs,
               batch_size=args.batch_size, lr=args.lr, pretrained="COCO (weights=DEFAULT)",
               config_path=str(cfg_path), commit=git_commit())
    notes = []

    def finish(status, metrics=None, extra=None):
        r = dict(row, status=status, notes="; ".join(notes))
        r.update(extra or {})
        if metrics:
            r.update(metrics)
        if args.smoke:
            print("[smoke] tidak dicatat ke log:", {k: r[k] for k in ("status", "notes")})
            return
        log_run(MEMBER, **r)
        if args.drive_dir:
            Path(args.drive_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(Path("logs") / f"runs_{MEMBER}.csv", args.drive_dir)

    try:
        row["env"] = env_string()
        set_seed(args.seed)

        limit_tr, limit_va = (16, 8) if args.smoke else (None, None)
        train_ds = CocoDetDataset(COCO_DIR / "train.json", train=True, hflip=args.hflip, vflip=args.vflip, limit=limit_tr)
        val_ds = CocoDetDataset(COCO_DIR / "val.json", train=False, limit=limit_va)
        print("train stats:", train_ds.stats)
        print("val stats  :", val_ds.stats)
        notes.append(f"train {train_ds.stats['n_images']} img, dibuang degenerat {train_ds.stats['n_dropped_degenerate']}")

        model = build_model(anchor_sizes=anchor_sizes, trainable_backbone_layers=args.trainable_layers,
                            **resolution_kwargs(args.size_mode, args.size)).to(device)
        row["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
        print(f"transform min_size={model.transform.min_size} max_size={model.transform.max_size}; "
              f"anchor sizes={model.rpn.anchor_generator.sizes}")

        gt = load_gt(COCO_DIR / "val.json")
        val_ids = val_ds.image_ids()

        train_loader = make_loader(train_ds, args.seed, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        total_iters = args.epochs * len(train_loader)
        warm = max(1, min(args.warmup_iters, total_iters // 5))

        def lr_lambda(it):
            if it < warm:
                return (it + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (it - warm) / max(1, total_iters - warm)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        scaler = make_scaler(device, args.amp and device.type == "cuda")

        start_epoch, best, best_epoch = 0, -1.0, -1
        last_ckpt, best_ckpt = out / "last.pt", out / "best.pt"
        if args.resume and last_ckpt.exists():
            ck = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_epoch, best, best_epoch = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
            # urutan data tidak identik dengan run tanpa putus -> catat
            train_loader = make_loader(train_ds, args.seed + start_epoch, batch_size=args.batch_size,
                                       shuffle=True, num_workers=args.num_workers)
            notes.append(f"resumed dari epoch {start_epoch} (tidak identik bit-per-bit dengan run tanpa putus)")
            print("resume dari epoch", start_epoch)

        cfg = {"args": vars(args), "run_id": run_id, "env": row["env"], "commit": row["commit"],
               "min_size": list(model.transform.min_size), "max_size": model.transform.max_size,
               "anchor_sizes": [list(s) for s in model.rpn.anchor_generator.sizes],
               "anchor_ratios": [list(r) for r in model.rpn.anchor_generator.aspect_ratios]}
        cfg_path.write_text(json.dumps(cfg, indent=2))

        hist_path = out / "history.csv"
        hist_fields = ["epoch", "loss", "lr", "val_map50", "val_map50_95", "val_ap50_platelet", "epoch_sec"]
        if not (args.resume and hist_path.exists()):
            hist_path.write_text(",".join(hist_fields) + "\n")

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            model.train()
            run_loss, n_it = 0.0, 0
            for images, targets in train_loader:
                images = [im.to(device, non_blocking=True) for im in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise DivergedError(f"loss tidak hingga/NaN pada epoch {epoch}, iterasi {n_it}: "
                                        f"{ {k: float(v) for k, v in loss_dict.items()} }")
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                run_loss += float(loss); n_it += 1

            m = {"map50": float("nan"), "map50_95": float("nan"), "ap50_Platelets": float("nan")}
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                model.roi_heads.score_thresh = args.eval_score_thresh
                m = coco_eval(gt, predict(model, val_ds, device, num_workers=args.num_workers), val_ids)
                model.roi_heads.score_thresh = 0.05  # kembalikan (bawaan seingat saya; tidak dipakai saat training)
                if m[args.select_metric] > best:
                    best, best_epoch = m[args.select_metric], epoch
                    save_atomic({"model": model.state_dict(), "epoch": epoch, "metrics": m}, best_ckpt)

            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "epoch": epoch, "best": best, "best_epoch": best_epoch}, last_ckpt)
            with open(hist_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, round(run_loss / max(1, n_it), 5), opt.param_groups[0]["lr"],
                                        m["map50"], m["map50_95"], m["ap50_Platelets"], round(time.time() - t0, 1)])
            print(f"epoch {epoch+1}/{args.epochs} loss {run_loss/max(1,n_it):.4f} "
                  f"val mAP50 {m['map50']:.4f} mAP50-95 {m['map50_95']:.4f} AP50-plt {m['ap50_Platelets']:.4f} "
                  f"({time.time()-t0:.0f}s)")

        if best_epoch < 0:
            raise RuntimeError("tidak ada evaluasi yang berjalan; best.pt tidak ada")

        # Evaluasi akhir dari best.pt (memastikan checkpoint yang disimpan menghasilkan angka yang dilaporkan)
        model.load_state_dict(torch.load(best_ckpt, map_location=device)["model"])
        model.roi_heads.score_thresh = args.eval_score_thresh
        results = predict(model, val_ds, device, num_workers=args.num_workers)
        m = coco_eval(gt, results, val_ids, verbose=True)
        mae = platelet_count_mae(gt, results, val_ids, score_thr=args.mae_score_thr)
        model.roi_heads.score_thresh = 0.05
        lat = measure_latency_ms(model, val_ds, device)
        (out / "final_metrics.json").write_text(json.dumps(
            {"best_epoch": best_epoch + 1, "val": m, "val_mae_platelet": mae, "latency_ms": lat}, indent=2))
        notes.append(f"best_epoch={best_epoch+1} by val_{args.select_metric}; score_thresh_eval={args.eval_score_thresh}; "
                     f"mae_thr={args.mae_score_thr}; latency=bs1 fp32 val[:50]")
        finish("ok", metrics={"val_map50": round(m["map50"], 4), "val_map50_95": round(m["map50_95"], 4),
                              "val_ap50_platelet": round(m["ap50_Platelets"], 4),
                              "val_mae_platelet": round(mae, 4)},
               extra={"latency_ms": round(lat, 2), "params_m": row["params_m"]})
        print("per kelas (val, best epoch):", {k: round(v, 4) for k, v in m.items()})

    except DivergedError as e:
        notes.append(str(e)[:300]); finish("diverged"); raise
    except KeyboardInterrupt:
        notes.append("dihentikan manual"); finish("partial"); raise
    except Exception as e:
        notes.append(f"{type(e).__name__}: {str(e)[:300]}"); traceback.print_exc(); finish("failed"); sys.exit(1)


if __name__ == "__main__":
    main()
