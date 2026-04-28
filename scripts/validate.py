"""검증 리포트 생성: 분류 threshold + 세그멘테이션 positive/empty 지표."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import label as label_components
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.ct_hemorrhage_dataset import (
    CT_CLASS_NAMES,
    build_ct_classifier_dataloaders,
    build_ct_seg_dataloaders,
)
from models.classifier import StrokeClassifier
from models.segmentor import StrokeSegmentor
from training.metrics import (
    binary_classification_metrics,
    cls_report,
    conf_matrix,
    finalize_segmentation_stats,
    merge_segmentation_stats,
    segmentation_stats,
    select_threshold_row,
    threshold_predictions,
    threshold_sweep,
)


def main(args):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = _device()
    ct_path = cfg["data"]["ct_hemorrhage_path"]

    print("=" * 72)
    print("  검증 리포트 (CT Hemorrhage Validation Set)")
    print("=" * 72)
    print(f"device: {device}")

    _validate_classifier(cfg, ct_path, device, args)
    _validate_segmentor(cfg, ct_path, device, args)
    print("\n" + "=" * 72)


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_classifier(cfg, ct_path, device, args):
    print("\n[1] 분류 모델")
    print("-" * 72)
    ckpt = torch.load(args.cls_ckpt, map_location=device, weights_only=False)
    c_cfg = ckpt["config"]
    _warn_config_mismatch(
        "classifier",
        c_cfg,
        cfg["classifier"],
        keys=("model_name", "image_size", "num_classes"),
    )
    model = StrokeClassifier(
        model_name=c_cfg.get("model_name", "efficientnet_b4"),
        num_classes=len(ckpt.get("class_names", CT_CLASS_NAMES)),
        pretrained=False,
        dropout_rate=c_cfg.get("dropout_rate", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, val_loader, _ = build_ct_classifier_dataloaders(
        data_root=ct_path,
        image_size=c_cfg.get("image_size", cfg["classifier"]["image_size"]),
        batch_size=min(args.batch_size, c_cfg.get("batch_size", cfg["classifier"]["batch_size"])),
        num_workers=0,
    )

    probs, labels = [], []
    with torch.no_grad():
        for imgs, lbl in tqdm(val_loader, desc="  classifier eval"):
            out = model(imgs.to(device))
            probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
            labels.append(lbl.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    thresholds = _thresholds(
        c_cfg.get("threshold_min", cfg["classifier"].get("threshold_min", 0.2)),
        c_cfg.get("threshold_max", cfg["classifier"].get("threshold_max", 0.7)),
        c_cfg.get("threshold_step", cfg["classifier"].get("threshold_step", 0.05)),
    )
    rows = threshold_sweep(probs[:, 1], labels, thresholds)
    selected = select_threshold_row(
        rows,
        metric_name=c_cfg.get("best_metric", cfg["classifier"].get("best_metric", "macro_f1")),
        min_recall=float(c_cfg.get("min_hemorrhagic_recall", cfg["classifier"].get("min_hemorrhagic_recall", 0.0))),
        min_specificity=float(c_cfg.get("min_specificity", cfg["classifier"].get("min_specificity", 0.0))),
    )
    ckpt_threshold = float(ckpt.get("val_threshold", c_cfg.get("cls_threshold", 0.5)))
    ckpt_preds = threshold_predictions(probs[:, 1], ckpt_threshold)
    ckpt_metrics = binary_classification_metrics(ckpt_preds, labels)

    print(f"\n최고 에폭      : {ckpt['epoch']}")
    print(f"검증 샘플 수   : {len(labels)}")
    print(f"checkpoint th : {ckpt_threshold:.2f}")
    print(_format_cls_row({**ckpt_metrics, "threshold": ckpt_threshold}))
    print("\nThreshold sweep:")
    print("th    acc     macroF1  hem_rec hem_f1  spec    FP  FN")
    for row in rows:
        marker = "*" if abs(row["threshold"] - selected["threshold"]) < 1e-8 else " "
        print(marker + _format_cls_row(row))

    print("\n클래스별 리포트 (checkpoint threshold):")
    print(cls_report(ckpt_preds, labels, ckpt.get("class_names", CT_CLASS_NAMES)))
    cm = conf_matrix(ckpt_preds, labels)
    print("혼동 행렬:")
    print(f"                  예측:normal  예측:hemorrhagic")
    print(f"실제:normal       {cm[0,0]:>10d}  {cm[0,1]:>16d}")
    print(f"실제:hemorrhagic  {cm[1,0]:>10d}  {cm[1,1]:>16d}")


def _validate_segmentor(cfg, ct_path, device, args):
    print("\n\n[2] 세그멘테이션 모델")
    print("-" * 72)
    ckpt_s = torch.load(args.seg_ckpt, map_location=device, weights_only=False)
    s_cfg = ckpt_s["config"]
    _warn_config_mismatch(
        "segmentor",
        s_cfg,
        cfg["segmentor"],
        keys=("encoder", "image_size", "batch_size"),
    )
    seg = StrokeSegmentor(
        encoder_name=s_cfg.get("encoder", "resnet34"),
        encoder_weights=None,
    )
    seg.load_state_dict(ckpt_s["model_state"])
    seg.to(device).eval()

    _, val_loader_s = build_ct_seg_dataloaders(
        data_root=ct_path,
        image_size=s_cfg.get("image_size", cfg["segmentor"]["image_size"]),
        batch_size=min(args.seg_batch_size, s_cfg.get("batch_size", cfg["segmentor"]["batch_size"])),
        bhsd_processed_dir=cfg["data"].get("bhsd_processed_dir", "./data/processed/bhsd"),
        num_workers=0,
    )

    thresholds = _thresholds(
        cfg["segmentor"].get("threshold_min", 0.2),
        cfg["segmentor"].get("threshold_max", 0.7),
        cfg["segmentor"].get("threshold_step", 0.05),
    )
    stats_by_threshold = {float(th): {} for th in thresholds}
    filter_cfg = _component_filter_cfg(s_cfg, cfg["segmentor"])

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader_s, desc="  segmentor eval"):
            probs = seg.predict_proba(imgs.to(device)).cpu().numpy()[:, 0]
            masks_t = masks.cpu()
            for threshold in thresholds:
                raw = probs >= float(threshold)
                if args.no_component_filter:
                    filtered = raw.astype(np.float32)
                else:
                    filtered = np.stack([
                        _filter_component_mask(raw[i], probs[i], filter_cfg)
                        for i in range(raw.shape[0])
                    ]).astype(np.float32)
                pred = torch.from_numpy(filtered).unsqueeze(1)
                key = float(threshold)
                stats_by_threshold[key] = merge_segmentation_stats(
                    stats_by_threshold[key],
                    segmentation_stats(pred, masks_t),
                )

    finalized = [
        {"threshold": threshold, **finalize_segmentation_stats(stats)}
        for threshold, stats in stats_by_threshold.items()
    ]
    best_metric = s_cfg.get("best_metric", cfg["segmentor"].get("best_metric", "dice_positive"))
    selected = max(finalized, key=lambda row: row.get(best_metric, 0.0))
    ckpt_threshold = float(s_cfg.get("seg_threshold", cfg["segmentor"].get("seg_threshold", 0.5)))
    closest = min(finalized, key=lambda row: abs(row["threshold"] - ckpt_threshold))

    print(f"\n최고 에폭    : {ckpt_s['epoch']}")
    print(f"검증 샘플 수 : {closest['samples']}")
    print(f"positive/empty: {closest['positive_count']} / {closest['empty_count']}")
    print(f"checkpoint th : {ckpt_threshold:.2f}")
    if args.no_component_filter:
        print("component filter: off")
    else:
        print(
            "component filter: "
            f"min_px={filter_cfg['min_component_px']}, "
            f"min_area_pct={filter_cfg['min_component_area_pct']}, "
            f"min_mean_prob={filter_cfg['min_component_mean_prob']}"
        )
    print(_format_seg_row(closest))
    print("\nThreshold sweep:")
    print("th    dice_all dice_pos iou_all iou_pos emptyFP emptyArea% posArea%")
    for row in finalized:
        marker = "*" if abs(row["threshold"] - selected["threshold"]) < 1e-8 else " "
        print(marker + _format_seg_row(row))


def _thresholds(start, stop, step):
    start, stop, step = float(start), float(stop), float(step)
    count = int(round((stop - start) / step)) + 1
    return np.round(start + np.arange(count) * step, 4)


def _format_cls_row(row):
    return (
        f"{row['threshold']:.2f}  {row['accuracy']:.4f}  {row['macro_f1']:.4f}  "
        f"{row['hemorrhagic_recall']:.4f}  {row['hemorrhagic_f1']:.4f}  "
        f"{row['specificity']:.4f}  {row['fp']:>3d} {row['fn']:>3d}"
    )


def _format_seg_row(row):
    return (
        f"{row['threshold']:.2f}  {row['dice_all']:.4f}   {row['dice_positive']:.4f}   "
        f"{row['iou_all']:.4f}  {row['iou_positive']:.4f}  "
        f"{row['empty_fp_rate']:.4f}  {row['empty_pred_area_pct']:.4f}  "
        f"{row['positive_pred_area_pct']:.4f}"
    )


def _component_filter_cfg(ckpt_cfg, current_cfg):
    return {
        "min_component_px": int(
            ckpt_cfg.get("min_component_px", current_cfg.get("min_component_px", 16))
        ),
        "min_component_area_pct": float(
            ckpt_cfg.get("min_component_area_pct", current_cfg.get("min_component_area_pct", 0.0))
        ),
        "min_component_mean_prob": float(
            ckpt_cfg.get("min_component_mean_prob", current_cfg.get("min_component_mean_prob", 0.0))
        ),
    }


def _filter_component_mask(mask, prob, cfg):
    labeled, component_count = label_components(mask.astype(bool), structure=np.ones((3, 3)))
    kept = np.zeros_like(mask, dtype=bool)
    total_px = mask.shape[0] * mask.shape[1]
    for component_idx in range(1, component_count + 1):
        component = labeled == component_idx
        area_px = int(component.sum())
        if area_px <= 0:
            continue
        area_pct = area_px / total_px * 100
        mean_prob = float(prob[component].mean())
        if (
            area_px >= cfg["min_component_px"]
            and area_pct >= cfg["min_component_area_pct"]
            and mean_prob >= cfg["min_component_mean_prob"]
        ):
            kept[component] = True
    return kept.astype(np.float32)


def _warn_config_mismatch(name, ckpt_cfg, current_cfg, keys):
    mismatches = []
    for key in keys:
        if key in ckpt_cfg and key in current_cfg and ckpt_cfg[key] != current_cfg[key]:
            mismatches.append(f"{key}: checkpoint={ckpt_cfg[key]} current={current_cfg[key]}")
    if mismatches:
        print(f"⚠️  {name} checkpoint/config mismatch:")
        for item in mismatches:
            print(f"   - {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls_ckpt", default="./checkpoints/classifier/best_classifier.pth")
    parser.add_argument("--seg_ckpt", default="./checkpoints/segmentor/best_segmentor.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seg_batch_size", type=int, default=24)
    parser.add_argument("--no_component_filter", action="store_true")
    main(parser.parse_args())
