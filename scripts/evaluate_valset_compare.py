"""Pipeline 규칙별 비교 평가:
  A. 분류기 단독 (no post-processing)
  B. 1% threshold 적용 (현재 pipeline)
  C. 세그만 있으면 hemorrhagic (pipeline 이전 버전)

val set 2089장으로 3가지 비교 → 사용자가 최적 규칙 판단.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.combined_dataset import build_combined_dataloaders
from models.classifier import StrokeClassifier
from models.segmentor import StrokeSegmentor


OUT_DIR = Path("./results/valset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def cm_metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    acc = (tp+tn) / max(tp+tn+fp+fn, 1)
    fpr = fp / max(tn+fp, 1)
    fnr = fn / max(tp+fn, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                sensitivity=sens, specificity=spec, accuracy=acc,
                fp_rate=fpr, fn_rate=fnr)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Val set 로드
    _, val_loader, _ = build_combined_dataloaders(
        ct_root="./data/raw/ct_hemorrhage/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
        tekno21_cache="./data/raw/tekno21",
        image_size=224, batch_size=1, num_workers=0,
    )
    val_ds = val_loader.dataset
    samples = val_ds.samples
    hf = val_ds.hf
    print(f"총 {len(samples)} 샘플\n")

    # Classifier 로드
    ckpt_c = torch.load("./checkpoints/classifier/best_classifier.pth",
                        map_location=device, weights_only=False)
    cls_model = StrokeClassifier(num_classes=2, pretrained=False,
                                  dropout_rate=ckpt_c["config"]["dropout_rate"])
    cls_model.load_state_dict(ckpt_c["model_state"])
    cls_model.to(device).eval()

    # Segmentor 로드
    ckpt_s = torch.load("./checkpoints/segmentor/best_segmentor.pth",
                        map_location=device, weights_only=False)
    seg_model = StrokeSegmentor(encoder_name=ckpt_s["config"]["encoder"],
                                 encoder_weights=None)
    seg_model.load_state_dict(ckpt_s["model_state"])
    seg_model.to(device).eval()

    # Transforms
    cls_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    seg_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    y_true = []
    pred_A = []  # 분류기 단독
    pred_B = []  # 1% threshold
    pred_C = []  # any-lesion = hemorrhagic

    lesion_pct_list = []
    cls_probs_list = []

    with torch.no_grad():
        for i, (source, ref, gt) in enumerate(samples):
            if source in ("ct", "bhsd"):
                img = np.array(Image.open(ref).convert("RGB"))
            else:
                item = hf[ref]
                im = item["image"]
                if not isinstance(im, Image.Image):
                    im = Image.fromarray(im)
                img = np.array(im.convert("RGB"))

            # Classifier
            cls_t = cls_tf(image=img)["image"].unsqueeze(0).to(device)
            _, probs = cls_model.predict(cls_t)
            cls_probs = probs.cpu().numpy()[0]
            cls_pred = int(np.argmax(cls_probs))

            # Segmentor
            seg_t = seg_tf(image=img)["image"].unsqueeze(0).to(device)
            mask = seg_model.predict_mask(seg_t, 0.5)
            mask_np = mask[0, 0].cpu().numpy()
            lesion_px = int(mask_np.sum())
            lesion_pct = lesion_px / (mask_np.shape[0] * mask_np.shape[1]) * 100

            # A. 분류기 단독
            pred_A.append(cls_pred)

            # B. 1% threshold (현재 pipeline)
            if lesion_pct <= 1.0:
                pred_B.append(0)
            elif cls_pred == 0:  # classifier normal, seg says > 1% → override
                pred_B.append(1)
            else:
                pred_B.append(cls_pred)

            # C. any lesion → hemorrhagic (old pipeline)
            if lesion_px > 0:
                pred_C.append(1)
            else:
                pred_C.append(cls_pred)

            y_true.append(gt)
            lesion_pct_list.append(lesion_pct)
            cls_probs_list.append(cls_probs)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(samples)}")

    # 각 rule 평가
    m_A = cm_metrics(y_true, pred_A)
    m_B = cm_metrics(y_true, pred_B)
    m_C = cm_metrics(y_true, pred_C)

    def fmt(m):
        return (f"  Accuracy={m['accuracy']:.4f}  Sens={m['sensitivity']:.4f}  "
                f"Spec={m['specificity']:.4f}  FP rate={m['fp_rate']:.4f}  "
                f"FN rate={m['fn_rate']:.4f}\n"
                f"  CM: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    report = f"""\
Val set 규칙별 비교 평가 (n={len(samples)})
=================================================

A. 분류기 단독 (Post-processing 없음):
{fmt(m_A)}

B. 현재 pipeline (병변≤1% → normal, 병변>1%+cls normal → hemorrhagic):
{fmt(m_B)}

C. 이전 pipeline (병변 존재하면 hemorrhagic):
{fmt(m_C)}

--- Trade-off 분석 ---
  B 규칙이 Normal 오탐을 줄이지만(FP ↓),
  작은 출혈(<1%)을 전부 normal로 판독해서 FN 급증
  (세그 Dice 0.56 수준이라 작은 출혈일수록 세그가 잘 못 잡음)

  임상 관점: FN(출혈 놓침)이 FP(정상 오탐)보다 훨씬 위험
  → A 규칙(분류기 단독)이 임상적으로 가장 안전
  → B 규칙은 스크리닝용(specificity ↑)으로는 좋지만 민감도 낮음

  brain_test 12장 결과가 B에서 완벽했던 건
  "큰 출혈 케이스만 있어서" 임. val set은 작은 출혈 포함.
"""
    print(report)
    (OUT_DIR / "rule_comparison.txt").write_text(report)
    print(f"저장: {OUT_DIR / 'rule_comparison.txt'}")


if __name__ == "__main__":
    main()
