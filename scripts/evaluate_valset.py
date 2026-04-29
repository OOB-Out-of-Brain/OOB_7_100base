"""학습에 쓰이지 않은 val set 2089장 상세 평가 (FP/FN 리포트 + 샘플 저장).

CQ500 다운로드 전까지의 대체 평가:
- val set은 학습 중 **검증용으로만** 쓰였고 backprop 안 됨 → 독립 평가 가능
- Patient-level split으로 leakage 없음 (patient ID 기준 분리됨)

출력 (results/valset/):
  - summary.csv (파일별 결과)
  - metrics.txt (sensitivity/specificity/FP rate/FN rate)
  - false_positives/ (정상인데 출혈이라 오탐한 샘플 이미지)
  - false_negatives/ (출혈인데 놓친 샘플 이미지)
"""

import sys, csv, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.combined_dataset import build_combined_dataloaders, CLASS_NAMES
from data.ct_hemorrhage_io import load_ct_image, ref_name
from inference.pipeline import StrokePipeline

_cfg = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))
_CT_ROOT   = _cfg["data"]["ct_hemorrhage_path"]
_TK_CACHE  = _cfg["data"]["tekno21_cache"]
_IMG_SIZE  = _cfg["classifier"]["image_size"]


OUT_DIR = Path("./results/valset")
MAX_ERROR_SAMPLES = 30  # FP/FN 각각 저장할 최대 샘플 수


def save_error_image(overlay_or_orig, out_path: Path, label: str, conf: float, lesion_pct: float):
    if overlay_or_orig is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay_or_orig)
    color = "red" if "FP" in label else "orange"
    ax.set_title(f"{label}  conf={conf:.1%}  lesion={lesion_pct:.1f}%", color=color)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_dir = OUT_DIR / "false_positives"
    fn_dir = OUT_DIR / "false_negatives"
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    print("Val set 로딩...")
    _, val_loader, _ = build_combined_dataloaders(
        ct_root=_CT_ROOT,
        tekno21_cache=_TK_CACHE,
        image_size=_IMG_SIZE, batch_size=1, num_workers=0,
    )
    val_ds = val_loader.dataset
    samples = val_ds.samples  # [(source, ref, label), ...]
    print(f"총 {len(samples)} 샘플 평가\n")

    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt="./checkpoints/segmentor/best_segmentor.pth",
    )

    tp = tn = fp = fn = 0
    fp_saved = fn_saved = 0
    summary_rows = []

    hf = val_ds.hf  # tekno21용
    for i, (source, ref, gt) in enumerate(samples):
        # 이미지 numpy로
        if source in ("ct", "bhsd"):
            img = load_ct_image(ref)
            name = ref_name(ref)
        else:  # tekno21
            item = hf[ref]
            im = item["image"]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            img = np.array(im.convert("RGB"))
            name = f"tk_{ref}.png"

        r = pipe.run(img)
        pred = 1 if r.class_name == "hemorrhagic" else 0

        if gt == 1 and pred == 1: tp += 1
        elif gt == 0 and pred == 0: tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
            if fp_saved < MAX_ERROR_SAMPLES:
                save_error_image((r.overlay_image if r.overlay_image is not None else img),
                                 fp_dir / f"fp_{fp_saved:03d}_{source}_{name}",
                                 "FALSE POSITIVE (normal → hemorrhagic)",
                                 r.confidence, r.lesion_area_pct)
                fp_saved += 1
        elif gt == 1 and pred == 0:
            fn += 1
            if fn_saved < MAX_ERROR_SAMPLES:
                save_error_image((r.overlay_image if r.overlay_image is not None else img),
                                 fn_dir / f"fn_{fn_saved:03d}_{source}_{name}",
                                 "FALSE NEGATIVE (hemorrhagic → normal)",
                                 r.confidence, r.lesion_area_pct)
                fn_saved += 1

        summary_rows.append([source, name, gt, pred,
                              f"{r.confidence:.3f}",
                              f"{r.lesion_area_pct:.2f}"])

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}  TP={tp} TN={tn} FP={fp} FN={fn}")

    # summary.csv
    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "file", "gt", "pred", "confidence", "lesion_pct"])
        w.writerows(summary_rows)

    # metrics
    total = tp + tn + fp + fn
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / max(total, 1)
    fpr = fp / max(tn + fp, 1)
    fnr = fn / max(tp + fn, 1)

    report = f"""\
Val set 평가 리포트 (총 {total} 샘플)
=============================================

분포:
  Normal      : {tn + fp}개
  Hemorrhagic : {tp + fn}개

Confusion matrix:
                예측 normal    예측 hemorrhagic
실제 normal       {tn:>6}         {fp:>6}  ← 오탐(FP)
실제 hemorrhagic  {fn:>6}         {tp:>6}  ← 누락(FN)

--- 주요 지표 ---
Sensitivity (Recall)     : {sens:.4f}  (출혈 탐지율)
Specificity              : {spec:.4f}  (정상 식별율)
PPV (Precision)          : {ppv:.4f}
NPV                      : {npv:.4f}
Accuracy                 : {acc:.4f}

--- 핵심 오탐/누락 ---
Normal 오탐율 (FP rate)     : {fpr:.4f}  ← 낮을수록 좋음
Hemorrhagic 누락율 (FN rate): {fnr:.4f}  ← 낮을수록 좋음 (임상적으로 중요)

샘플 이미지 저장:
  FP {fp_saved}개 → {fp_dir}
  FN {fn_saved}개 → {fn_dir}
"""
    print("\n" + report)
    (OUT_DIR / "metrics.txt").write_text(report)
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
