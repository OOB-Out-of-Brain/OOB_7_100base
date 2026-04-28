"""
분류 모델 학습 (2단계: backbone freeze → unfreeze + Focal Loss).

실행:
    python training/train_classifier.py
    python training/train_classifier.py --epochs 50 --batch_size 64
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

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm
import yaml

from data.combined_dataset import build_combined_dataloaders, CLASS_NAMES as CT_CLASS_NAMES
from models.classifier import StrokeClassifier
from training.metrics import (
    FocalLoss,
    accuracy,
    binary_classification_metrics,
    cls_report,
    conf_matrix,
    select_threshold_row,
    threshold_predictions,
    threshold_sweep,
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
                    device, use_amp, scaler, grad_clip):
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with _autocast(device.type, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

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

        total_loss += loss.item()
        total_acc += accuracy(logits.detach().argmax(dim=1), labels)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, threshold=0.5):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        with _autocast(device.type, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        preds = (probs[:, 1] >= threshold).long()

        total_loss += loss.item()
        total_acc += accuracy(preds, labels)
        all_probs.append(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader)
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.array(all_labels)
    preds_np = threshold_predictions(probs_np[:, 1], threshold)
    return total_loss / n, total_acc / n, preds_np, labels_np, probs_np


def _thresholds_from_config(c: dict) -> np.ndarray:
    start = float(c.get("threshold_min", 0.2))
    stop = float(c.get("threshold_max", 0.7))
    step = float(c.get("threshold_step", 0.05))
    if step <= 0:
        raise ValueError("classifier.threshold_step must be > 0")
    count = int(round((stop - start) / step)) + 1
    return np.round(start + np.arange(count) * step, 4)


def _select_validation_metrics(probs: np.ndarray, labels: np.ndarray, c: dict) -> dict:
    rows = threshold_sweep(probs[:, 1], labels, _thresholds_from_config(c))
    return select_threshold_row(
        rows,
        metric_name=c.get("best_metric", "macro_f1"),
        min_recall=float(c.get("min_hemorrhagic_recall", 0.0)),
        min_specificity=float(c.get("min_specificity", 0.0)),
    )


def _format_cls_metrics(metrics: dict) -> str:
    return (
        f"score={metrics['selection_score']:.4f} "
        f"{metrics['metric_name']}@th={metrics['threshold']:.2f} | "
        f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"hem_rec={metrics['hemorrhagic_recall']:.4f} "
        f"hem_f1={metrics['hemorrhagic_f1']:.4f} spec={metrics['specificity']:.4f}"
    )


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    configure_torch_runtime(cfg)
    c = cfg["classifier"]

    epochs      = args.epochs     or c["epochs"]
    batch_size  = args.batch_size or c["batch_size"]
    lr          = args.lr         or c["learning_rate"]
    image_size  = c["image_size"]
    num_workers = cfg["data"].get("num_workers", 6)
    freeze_epochs = min(c.get("freeze_epochs", 5), epochs)
    save_path = Path(c["save_path"])
    save_path.mkdir(parents=True, exist_ok=True)

    device  = get_device()
    use_amp = c.get("amp", True) and device.type in ("cuda", "mps")
    grad_clip = c.get("grad_clip", 1.0)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and use_amp else None

    print(f"\n디바이스: {device}  AMP: {use_amp}")
    print(runtime_summary(device))
    print(f"모델: {c['model_name']}  image_size: {image_size}")
    print(f"epochs: {epochs} (freeze {freeze_epochs} + unfreeze {epochs - freeze_epochs})\n")

    train_loader, val_loader, class_weights = build_combined_dataloaders(
        ct_root=cfg["data"]["ct_hemorrhage_path"],
        tekno21_cache=cfg["data"]["tekno21_cache"],
        image_size=image_size, batch_size=batch_size,
        num_workers=num_workers,
        bhsd_processed_dir=cfg["data"].get("bhsd_processed_dir", "./data/processed/bhsd"),
    )
    print(f"학습: {len(train_loader.dataset)}개  검증: {len(val_loader.dataset)}개\n")

    model = StrokeClassifier(
        model_name=c["model_name"],
        num_classes=c["num_classes"],
        pretrained=True,
        dropout_rate=c["dropout_rate"],
    ).to(device)

    criterion = FocalLoss(
        gamma=c.get("focal_gamma", 2.0),
        weight=class_weights.to(device),
        label_smoothing=c.get("label_smoothing", 0.05),
    )

    best_score = -1.0
    patience_counter = 0
    patience = c["early_stopping_patience"]

    # ── 1단계: backbone freeze, head만 학습 ──────────────────────────────────
    print(f"[Phase 1] backbone freeze — {freeze_epochs} epochs")
    model.freeze_backbone()
    opt1 = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=lr, weight_decay=c["weight_decay"])
    sch1 = OneCycleLR(opt1, max_lr=lr,
                      steps_per_epoch=len(train_loader),
                      epochs=freeze_epochs, pct_start=0.3)

    for epoch in range(1, freeze_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          opt1, sch1, device, use_amp, scaler, grad_clip)
        vl_loss, vl_acc, vl_preds, vl_labels, vl_probs = evaluate(
            model, val_loader, criterion, device, use_amp, c.get("cls_threshold", 0.5)
        )
        vl_metrics = _select_validation_metrics(vl_probs, vl_labels, c)
        print(f"  Epoch {epoch:2d}/{freeze_epochs} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={vl_loss:.4f} {_format_cls_metrics(vl_metrics)}")
        if vl_metrics["selection_score"] > best_score:
            best_score = vl_metrics["selection_score"]
            _save(model, epoch, vl_metrics, c, save_path)
        clear_device_cache(device)

    # ── 2단계: 전체 unfreeze, 차등 학습률 ────────────────────────────────────
    remaining = epochs - freeze_epochs
    if remaining > 0:
        print(f"\n[Phase 2] full unfreeze — {remaining} epochs  (backbone lr={lr*0.1:.5f})")
        model.unfreeze_backbone()
        opt2 = AdamW([
            {"params": model.backbone.parameters(), "lr": lr * 0.1},
            {"params": model.head.parameters(),     "lr": lr},
        ], weight_decay=c["weight_decay"])
        sch2 = CosineAnnealingLR(opt2, T_max=remaining, eta_min=lr * 0.001)

        for epoch in range(1, remaining + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                              opt2, None, device, use_amp, scaler, grad_clip)
            vl_loss, vl_acc, vl_preds, vl_labels, vl_probs = evaluate(
                model, val_loader, criterion, device, use_amp, c.get("cls_threshold", 0.5)
            )
            vl_metrics = _select_validation_metrics(vl_probs, vl_labels, c)
            sch2.step()
            print(f"  Epoch {freeze_epochs+epoch:2d}/{epochs} | "
                  f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val loss={vl_loss:.4f} {_format_cls_metrics(vl_metrics)}")

            if vl_metrics["selection_score"] > best_score:
                best_score = vl_metrics["selection_score"]
                patience_counter = 0
                _save(model, freeze_epochs + epoch, vl_metrics, c, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping (patience={patience})")
                    break
            clear_device_cache(device)

    # ── 최종 리포트 ──────────────────────────────────────────────────────────
    ckpt_path = save_path / "best_classifier.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        threshold = float(ckpt.get("val_threshold", ckpt.get("config", {}).get("cls_threshold", 0.5)))
        _, _, vl_preds, vl_labels, vl_probs = evaluate(
            model, val_loader, criterion, device, use_amp, threshold
        )
        final_metrics = binary_classification_metrics(vl_preds, vl_labels)
        print(f"\n최종 검증 리포트 (best model, epoch={ckpt.get('epoch')}, threshold={threshold:.2f}):")
        print(_format_cls_metrics({
            **final_metrics,
            "threshold": threshold,
            "metric_name": ckpt.get("selection_metric", "macro_f1"),
            "selection_score": final_metrics.get(ckpt.get("selection_metric", "macro_f1"), 0.0),
        }))
        print(cls_report(vl_preds, vl_labels, CT_CLASS_NAMES))
        print("Confusion matrix:")
        print(conf_matrix(vl_preds, vl_labels))
    print(f"\n학습 완료. 저장 경로: {ckpt_path}")


def _save(model, epoch, val_metrics, c, save_path):
    ckpt_config = copy.deepcopy(c)
    ckpt_config["cls_threshold"] = float(val_metrics["threshold"])
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_acc": val_metrics["accuracy"],
        "val_score": val_metrics["selection_score"],
        "val_metrics": val_metrics,
        "val_threshold": float(val_metrics["threshold"]),
        "selection_metric": val_metrics["metric_name"],
        "class_names": CT_CLASS_NAMES,
        "config": ckpt_config,
    }, save_path / "best_classifier.pth")
    print(f"    → 모델 저장 (best {_format_cls_metrics(val_metrics)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--resume",     action="store_true", help="(예약됨) 체크포인트 재개")
    args = parser.parse_args()
    main(args)
