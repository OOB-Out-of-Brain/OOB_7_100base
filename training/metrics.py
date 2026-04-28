import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ── Classification ────────────────────────────────────────────────────────────

def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds == targets).float().mean().item()


def cls_report(preds: np.ndarray, targets: np.ndarray,
               class_names: list) -> str:
    labels = list(range(len(class_names)))
    return classification_report(
        targets,
        preds,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def conf_matrix(preds: np.ndarray, targets: np.ndarray, labels=None) -> np.ndarray:
    return confusion_matrix(targets, preds, labels=labels)


def threshold_predictions(probs: np.ndarray, threshold: float,
                          positive_idx: int = 1) -> np.ndarray:
    """이진 분류 확률을 inference threshold와 같은 방식으로 label로 바꾼다."""
    probs = np.asarray(probs)
    if probs.ndim == 2:
        positive_probs = probs[:, positive_idx]
    else:
        positive_probs = probs
    return (positive_probs >= threshold).astype(np.int64)


def binary_classification_metrics(preds: np.ndarray, targets: np.ndarray,
                                  positive_idx: int = 1) -> dict:
    """normal/hemorrhagic 이진 분류의 핵심 지표."""
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    labels = [0, positive_idx]
    cm = confusion_matrix(targets, preds, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=[positive_idx],
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "hemorrhagic_precision": float(precision),
        "hemorrhagic_recall": float(recall),
        "hemorrhagic_f1": float(f1),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(positive_probs: np.ndarray, targets: np.ndarray,
                    thresholds) -> list:
    """여러 threshold에서 분류 지표를 계산한다."""
    rows = []
    for threshold in thresholds:
        preds = threshold_predictions(positive_probs, float(threshold))
        metrics = binary_classification_metrics(preds, targets)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows


def select_threshold_row(rows: list, metric_name: str = "macro_f1",
                         min_recall: float = 0.0,
                         min_specificity: float = 0.0) -> dict:
    """recall/specificity 최소 조건을 만족하는 row 중 metric이 가장 좋은 값을 고른다."""
    eligible = [
        r for r in rows
        if r["hemorrhagic_recall"] >= min_recall and r["specificity"] >= min_specificity
    ]
    if not eligible and (min_recall > 0 or min_specificity > 0):
        print(f"  ⚠️  threshold 제약 불충족 (recall≥{min_recall}, spec≥{min_specificity}) "
              f"→ 제약 없이 최적 threshold 선택")
    pool = eligible or rows
    if not pool:
        raise ValueError("threshold rows are empty")

    def _key(row):
        return (
            row.get(metric_name, 0.0),
            row["hemorrhagic_recall"],
            row["specificity"],
            -abs(row["threshold"] - 0.5),
        )

    selected = max(pool, key=_key).copy()
    selected["metric_name"] = metric_name
    selected["selection_score"] = float(selected.get(metric_name, 0.0))
    selected["met_constraints"] = bool(eligible)
    return selected


def multiclass_classification_metrics(preds: np.ndarray, targets: np.ndarray,
                                      class_names: list) -> dict:
    """3-class classifier metrics with per-class recall/f1 for model selection."""
    preds = np.asarray(preds).astype(np.int64)
    targets = np.asarray(targets).astype(np.int64)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        zero_division=0,
    )
    metrics = {
        "accuracy": float((preds == targets).mean()) if len(targets) else 0.0,
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)),
    }
    for idx, name in enumerate(class_names):
        metrics[f"{name}_precision"] = float(precision[idx])
        metrics[f"{name}_recall"] = float(recall[idx])
        metrics[f"{name}_f1"] = float(f1[idx])
        metrics[f"{name}_support"] = int(support[idx])
    return metrics


# ── Segmentation ──────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor,
               smooth: float = 1e-6) -> float:
    """배치 평균 Dice."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return dice.mean().item()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1e-6) -> float:
    """배치 평균 IoU."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


def segmentation_stats(pred: torch.Tensor, target: torch.Tensor,
                       smooth: float = 1e-6) -> dict:
    """배치 단위 segmentation 통계. positive/empty mask를 분리해서 누적 가능하다."""
    pred = pred.float().view(pred.size(0), -1)
    target = target.float().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)

    positive = target_sum > 0
    empty = ~positive
    pixel_count = pred.size(1)
    stats = {
        "samples": int(pred.size(0)),
        "dice_sum": float(dice.sum().item()),
        "iou_sum": float(iou.sum().item()),
        "positive_count": int(positive.sum().item()),
        "empty_count": int(empty.sum().item()),
        "positive_dice_sum": float(dice[positive].sum().item()) if positive.any() else 0.0,
        "positive_iou_sum": float(iou[positive].sum().item()) if positive.any() else 0.0,
        "empty_false_positive_count": int((pred_sum[empty] > 0).sum().item()) if empty.any() else 0,
        "empty_pred_area_pct_sum": float((pred_sum[empty] / pixel_count * 100).sum().item())
        if empty.any() else 0.0,
        "positive_pred_area_pct_sum": float((pred_sum[positive] / pixel_count * 100).sum().item())
        if positive.any() else 0.0,
    }
    return stats


def merge_segmentation_stats(total: dict, batch: dict) -> dict:
    if not total:
        return batch.copy()
    for key, value in batch.items():
        total[key] = total.get(key, 0) + value
    return total


def finalize_segmentation_stats(stats: dict) -> dict:
    samples = max(1, stats.get("samples", 0))
    positive_count = stats.get("positive_count", 0)
    empty_count = stats.get("empty_count", 0)
    return {
        "dice_all": stats.get("dice_sum", 0.0) / samples,
        "iou_all": stats.get("iou_sum", 0.0) / samples,
        "dice_positive": stats.get("positive_dice_sum", 0.0) / positive_count
        if positive_count else 0.0,
        "iou_positive": stats.get("positive_iou_sum", 0.0) / positive_count
        if positive_count else 0.0,
        "empty_fp_rate": stats.get("empty_false_positive_count", 0) / empty_count
        if empty_count else 0.0,
        "empty_pred_area_pct": stats.get("empty_pred_area_pct_sum", 0.0) / empty_count
        if empty_count else 0.0,
        "positive_pred_area_pct": stats.get("positive_pred_area_pct_sum", 0.0) / positive_count
        if positive_count else 0.0,
        "samples": int(stats.get("samples", 0)),
        "positive_count": int(positive_count),
        "empty_count": int(empty_count),
        "empty_false_positive_count": int(stats.get("empty_false_positive_count", 0)),
    }


def multiclass_segmentation_stats(pred: torch.Tensor, target: torch.Tensor,
                                  num_classes: int = 3,
                                  smooth: float = 1e-6) -> dict:
    """Batch-level per-class Dice/IoU stats for class-index masks."""
    pred = pred.long()
    target = target.long()
    stats = {"samples": int(pred.size(0))}
    for cls_idx in range(num_classes):
        pred_c = (pred == cls_idx).float().view(pred.size(0), -1)
        target_c = (target == cls_idx).float().view(target.size(0), -1)
        intersection = (pred_c * target_c).sum(dim=1)
        pred_sum = pred_c.sum(dim=1)
        target_sum = target_c.sum(dim=1)
        union = pred_sum + target_sum - intersection
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)
        present = target_sum > 0
        empty = ~present
        stats[f"class_{cls_idx}_dice_sum"] = float(dice[present].sum().item()) if present.any() else 0.0
        stats[f"class_{cls_idx}_iou_sum"] = float(iou[present].sum().item()) if present.any() else 0.0
        stats[f"class_{cls_idx}_count"] = int(present.sum().item())
        stats[f"class_{cls_idx}_empty_count"] = int(empty.sum().item())
        stats[f"class_{cls_idx}_empty_fp_count"] = int((pred_sum[empty] > 0).sum().item()) if empty.any() else 0
    return stats


def finalize_multiclass_segmentation_stats(stats: dict, class_names: list) -> dict:
    out = {"samples": int(stats.get("samples", 0))}
    lesion_dices = []
    lesion_ious = []
    for cls_idx, name in enumerate(class_names):
        count = stats.get(f"class_{cls_idx}_count", 0)
        empty_count = stats.get(f"class_{cls_idx}_empty_count", 0)
        dice = stats.get(f"class_{cls_idx}_dice_sum", 0.0) / count if count else 0.0
        iou = stats.get(f"class_{cls_idx}_iou_sum", 0.0) / count if count else 0.0
        out[f"dice_{name}"] = dice
        out[f"iou_{name}"] = iou
        out[f"count_{name}"] = int(count)
        out[f"empty_fp_rate_{name}"] = (
            stats.get(f"class_{cls_idx}_empty_fp_count", 0) / empty_count
            if empty_count else 0.0
        )
        if cls_idx > 0:
            lesion_dices.append(dice)
            lesion_ious.append(iou)
    out["dice_lesion_mean"] = float(np.mean(lesion_dices)) if lesion_dices else 0.0
    out["iou_lesion_mean"] = float(np.mean(lesion_ious)) if lesion_ious else 0.0
    return out


# ── Loss functions ────────────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """Focal Loss — 어렵게 분류되는 샘플에 더 집중. 클래스 불균형에 효과적."""

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class MulticlassDiceCELoss(torch.nn.Module):
    """Dice + CrossEntropy for class-index segmentation masks."""

    def __init__(self, num_classes: int, dice_weight: float = 0.6,
                 ce_weight: float = 0.4, class_weights=None,
                 ignore_bg_in_dice: bool = True, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_bg_in_dice = ignore_bg_in_dice
        self.smooth = smooth
        self.ce = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets.long())
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(targets.long(), num_classes=self.num_classes)
        target_1h = target_1h.permute(0, 3, 1, 2).float()
        start = 1 if self.ignore_bg_in_dice else 0
        dims = (0, 2, 3)
        intersection = (probs[:, start:] * target_1h[:, start:]).sum(dim=dims)
        denominator = probs[:, start:].sum(dim=dims) + target_1h[:, start:].sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


class TverskyBCELoss(torch.nn.Module):
    """Tversky Loss + BCE. FN에 더 큰 패널티 → 작은 출혈 병변 검출 향상."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, bce_weight: float = 0.3,
                 smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha      # FP 패널티
        self.beta = beta        # FN 패널티 (beta > alpha → recall 중시)
        self.bce_w = bce_weight
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob = torch.sigmoid(logits)
        p = prob.view(prob.size(0), -1)
        t = targets.view(targets.size(0), -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_loss = (1 - tversky).mean()

        return (1 - self.bce_w) * tversky_loss + self.bce_w * bce_loss


class DiceBCELoss(torch.nn.Module):
    """Dice + BCE 조합 손실. 의료 영상 분할에 적합."""

    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice_w = dice_weight
        self.bce_w = bce_weight
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        prob = torch.sigmoid(logits)
        prob_flat = prob.view(prob.size(0), -1)
        tgt_flat = targets.view(targets.size(0), -1)
        intersection = (prob_flat * tgt_flat).sum(dim=1)
        dice_loss = 1 - (2 * intersection + 1) / (prob_flat.sum(dim=1) + tgt_flat.sum(dim=1) + 1)
        dice_loss = dice_loss.mean()

        return self.dice_w * dice_loss + self.bce_w * bce_loss
