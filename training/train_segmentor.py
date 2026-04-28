"""
U-Net 분할 모델 학습 (2단계: encoder freeze → unfreeze + TverskyBCE Loss).

실행:
    python training/train_segmentor.py
    python training/train_segmentor.py --epochs 60 --batch_size 24
"""

import os
import argparse
import copy
import sys
from pathlib import Path

# M4 Pro 24GB 기준: MPS를 강하게 쓰되 스왑/시스템 프리징은 피한다.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.95")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.85")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm
import yaml

from data.ct_hemorrhage_dataset import build_ct_seg_dataloaders
from models.segmentor import StrokeSegmentor
from training.metrics import (
    TverskyBCELoss,
    finalize_segmentation_stats,
    merge_segmentation_stats,
    segmentation_stats,
)
from training.runtime import clear_device_cache, configure_torch_runtime, runtime_summary, suppress_noisy_runtime_warnings

suppress_noisy_runtime_warnings()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast(device_type: str, enabled: bool):
    if not enabled or device_type == "cpu":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if device_type == "mps" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, use_amp, scaler, grad_clip, threshold):
    model.train()
    total_loss = 0.0
    stats_total = {}

    for images, masks in tqdm(loader, desc="  train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with _autocast(device.type, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        pred_masks = (torch.sigmoid(logits.detach()) > threshold).float()
        total_loss += loss.item()
        stats_total = merge_segmentation_stats(
            stats_total,
            segmentation_stats(pred_masks, masks),
        )

    n = len(loader)
    return total_loss / n, finalize_segmentation_stats(stats_total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, threshold):
    model.eval()
    total_loss = 0.0
    stats_total = {}

    for images, masks in tqdm(loader, desc="  eval ", leave=False):
        images, masks = images.to(device), masks.to(device)
        with _autocast(device.type, use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        pred_masks = (torch.sigmoid(logits) > threshold).float()

        total_loss += loss.item()
        stats_total = merge_segmentation_stats(
            stats_total,
            segmentation_stats(pred_masks, masks),
        )

    n = len(loader)
    return total_loss / n, finalize_segmentation_stats(stats_total)


def _format_seg_metrics(metrics: dict, best_metric: str = "dice_positive") -> str:
    return (
        f"{best_metric}={metrics.get(best_metric, 0.0):.4f} | "
        f"dice_all={metrics['dice_all']:.4f} dice_pos={metrics['dice_positive']:.4f} "
        f"iou_all={metrics['iou_all']:.4f} iou_pos={metrics['iou_positive']:.4f} "
        f"empty_fp={metrics['empty_fp_rate']:.4f}"
    )


def _save(model, epoch, val_metrics, s, save_path, best_metric):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_dice": val_metrics["dice_all"],
        "val_iou": val_metrics["iou_all"],
        "val_score": val_metrics.get(best_metric, 0.0),
        "val_metrics": val_metrics,
        "selection_metric": best_metric,
        "config": copy.deepcopy(s),
    }, save_path / "best_segmentor.pth")
    print(f"    → 모델 저장 (best {_format_seg_metrics(val_metrics, best_metric)})")


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    configure_torch_runtime(cfg)
    s = cfg["segmentor"]
    d = cfg["data"]

    epochs        = args.epochs     or s["epochs"]
    batch_size    = args.batch_size or s["batch_size"]
    lr            = args.lr         or s["learning_rate"]
    image_size    = s["image_size"]
    freeze_epochs = min(s.get("freeze_epochs", 5), epochs)
    save_path     = Path(s["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    use_amp = s.get("amp", True) and device.type in ("cuda", "mps")
    grad_clip = s.get("grad_clip", 1.0)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and use_amp else None

    print(f"\n디바이스: {device}  AMP: {use_amp}")
    print(runtime_summary(device))
    print(f"encoder: {s['encoder']}  image_size: {image_size}")
    print(f"epochs: {epochs} (freeze {freeze_epochs} + unfreeze {epochs - freeze_epochs})\n")

    train_loader, val_loader = build_ct_seg_dataloaders(
        data_root=d["ct_hemorrhage_path"],
        image_size=image_size,
        batch_size=batch_size,
        bhsd_processed_dir=d.get("bhsd_processed_dir", "./data/processed/bhsd"),
    )
    print(f"학습: {len(train_loader.dataset)}개  검증: {len(val_loader.dataset)}개\n")

    model = StrokeSegmentor(
        encoder_name=s["encoder"],
        encoder_weights=s["encoder_weights"],
    ).to(device)

    criterion = TverskyBCELoss(
        alpha=s.get("tversky_alpha", 0.3),
        beta=s.get("tversky_beta", 0.7),
        bce_weight=s.get("bce_weight", 0.3),
    )

    best_metric = s.get("best_metric", "dice_positive")
    best_score = -1.0
    patience_counter = 0
    patience = s["early_stopping_patience"]
    seg_threshold = float(s.get("seg_threshold", 0.5))

    # ── 1단계: encoder freeze, decoder만 학습 ────────────────────────────────
    print(f"[Phase 1] encoder freeze — {freeze_epochs} epochs")
    model.freeze_encoder()
    # Phase 1: decoder만 학습 → encoder보다 5× 높은 LR로 빠르게 수렴
    decoder_lr = lr * 5
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=decoder_lr, weight_decay=s["weight_decay"])
    sch1 = OneCycleLR(opt1, max_lr=decoder_lr,
                      steps_per_epoch=len(train_loader),
                      epochs=freeze_epochs, pct_start=0.3)

    for epoch in range(1, freeze_epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(model, train_loader, criterion,
                                              opt1, sch1, device, use_amp, scaler, grad_clip,
                                              seg_threshold)
        vl_loss, vl_metrics = evaluate(model, val_loader, criterion, device, use_amp, seg_threshold)
        print(f"  Epoch {epoch:2d}/{freeze_epochs} | "
              f"train loss={tr_loss:.4f} dice_pos={tr_metrics['dice_positive']:.4f} | "
              f"val loss={vl_loss:.4f} {_format_seg_metrics(vl_metrics, best_metric)}")
        score = vl_metrics.get(best_metric, 0.0)
        if score > best_score:
            best_score = score
            _save(model, epoch, vl_metrics, s, save_path, best_metric)
        clear_device_cache(device)

    # ── 2단계: 전체 unfreeze, 차등 학습률 ────────────────────────────────────
    remaining = epochs - freeze_epochs
    if remaining > 0:
        print(f"\n[Phase 2] full unfreeze — {remaining} epochs  "
              f"(encoder lr={lr*0.1:.6f}, decoder lr={lr:.6f})")
        model.unfreeze_encoder()
        opt2 = AdamW([
            {"params": model.unet.encoder.parameters(), "lr": lr * 0.1},
            {"params": model.unet.decoder.parameters(), "lr": lr},
            {"params": model.unet.segmentation_head.parameters(), "lr": lr},
        ], weight_decay=s["weight_decay"])
        sch2 = CosineAnnealingLR(opt2, T_max=remaining, eta_min=lr * 0.001)

        for epoch in range(1, remaining + 1):
            tr_loss, tr_metrics = train_one_epoch(model, train_loader, criterion,
                                                  opt2, None, device, use_amp, scaler, grad_clip,
                                                  seg_threshold)
            vl_loss, vl_metrics = evaluate(model, val_loader, criterion, device, use_amp, seg_threshold)
            sch2.step()

            print(f"  Epoch {freeze_epochs+epoch:2d}/{epochs} | "
                  f"train loss={tr_loss:.4f} dice_pos={tr_metrics['dice_positive']:.4f} | "
                  f"val loss={vl_loss:.4f} {_format_seg_metrics(vl_metrics, best_metric)}")

            score = vl_metrics.get(best_metric, 0.0)
            if score > best_score:
                best_score = score
                patience_counter = 0
                _save(model, freeze_epochs + epoch, vl_metrics, s, save_path, best_metric)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping (patience={patience})")
                    break
            clear_device_cache(device)

    print(f"\n학습 완료. 저장 경로: {save_path / 'best_segmentor.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--resume",     action="store_true", help="(예약됨) 체크포인트 재개")
    args = parser.parse_args()
    main(args)
