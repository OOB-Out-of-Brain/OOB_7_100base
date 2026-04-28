"""CQ500 테스트 데이터셋으로 pipeline 평가 (오탐율/누락률 중심).

전제:
  - data/raw/cq500/ 에 CQ500CT{N}/ 폴더들 + reads.csv 존재
  - 각 폴더 안에 DICOM (*.dcm) 파일 다수 (CT 슬라이스)
  - reads.csv 에 R1:ICH, R2:ICH, R3:ICH 라벨 (3명 방사선과의사 판독)

평가 전략:
  - 각 스캔(환자 1명)의 중앙 슬라이스들에 대해 pipeline 추론
  - 스캔 단위 판독: 어떤 슬라이스라도 hemorrhagic → 해당 스캔 = hemorrhagic
  - GT = 3명 readers 중 다수결 (2/3 이상이 ICH=1 이면 hemorrhagic)

출력 (results/cq500/):
  - summary.csv (스캔 ID, GT, 예측, 신뢰도, 병변 px)
  - metrics.txt (sensitivity, specificity, PPV, NPV, confusion matrix)
  - false_positives/ (실제 normal인데 hemorrhagic 오탐한 스캔 샘플)
  - false_negatives/ (실제 hemorrhagic인데 놓친 스캔 샘플)
"""

import sys, csv, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pydicom

from inference.pipeline import StrokePipeline


CQ500_DIR = Path("./data/raw/cq500")
OUT_DIR = Path("./results/cq500")


def find_scan_dirs(root: Path) -> list[Path]:
    """cq500/qctNN/CQ500CT{id} CQ500CT{id}/ 구조에서 스캔 폴더 전부 수집."""
    scans = []
    for wrapper in sorted(root.iterdir()):
        if not wrapper.is_dir() or not wrapper.name.lower().startswith("qct"):
            continue
        for scan in sorted(wrapper.iterdir()):
            if scan.is_dir() and "CQ500CT" in scan.name:
                scans.append(scan)
    return scans


def normalize_scan_name(folder_name: str) -> str:
    """'CQ500CT191 CQ500CT191' → 'CQ500-CT-191' (reads.csv 포맷)."""
    first = folder_name.split()[0]
    m = re.match(r"CQ500CT(\d+)", first)
    return f"CQ500-CT-{m.group(1)}" if m else folder_name


def apply_brain_window(hu_arr: np.ndarray, center: int = 40, width: int = 80) -> np.ndarray:
    """HU → 0-255 (brain window 40/80)."""
    lo, hi = center - width / 2, center + width / 2
    x = np.clip(hu_arr, lo, hi)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


def dicom_to_png(dcm_path: Path) -> np.ndarray:
    """DICOM → brain-windowed uint8 2D array."""
    ds = pydicom.dcmread(str(dcm_path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    hu = arr * slope + intercept
    return apply_brain_window(hu)


def parse_gt(reads_csv: Path) -> dict:
    """reads.csv 읽어서 각 스캔의 GT(다수결) 반환. 2/3 이상 ICH=1 이면 1."""
    gt = {}
    if not reads_csv.exists():
        return gt
    with open(reads_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name") or row.get("Name")
            if not name:
                continue
            cols = ["R1:ICH", "R2:ICH", "R3:ICH"]
            votes = [int(row.get(c, 0) or 0) for c in cols if c in row]
            if not votes:
                continue
            gt[name] = 1 if sum(votes) >= 2 else 0
    return gt


def evaluate_scan(pipe: StrokePipeline, scan_dir: Path,
                   max_slices: int = 12) -> dict:
    """한 스캔 평가. 중앙 ~max_slices개 슬라이스 추론 후 스캔 단위 결과."""
    dcm_files = sorted(scan_dir.rglob("*.dcm"))
    if not dcm_files:
        return {"hemorrhagic": False, "n_slices": 0, "max_conf": 0, "max_lesion_pct": 0}

    # 중앙 슬라이스 선택
    n = len(dcm_files)
    if n > max_slices:
        start = (n - max_slices) // 2
        sel = dcm_files[start:start + max_slices]
    else:
        sel = dcm_files

    hemorrhagic = False
    max_conf = 0.0
    max_lesion_pct = 0.0
    best_overlay = None

    for dcm in sel:
        try:
            u8 = dicom_to_png(dcm)
        except Exception:
            continue
        rgb = np.stack([u8] * 3, axis=-1)
        r = pipe.run(rgb)
        if r.class_name == "hemorrhagic":
            hemorrhagic = True
            if r.lesion_area_pct > max_lesion_pct:
                max_lesion_pct = r.lesion_area_pct
                max_conf = r.confidence
                best_overlay = r.overlay_image
    return {
        "hemorrhagic": hemorrhagic,
        "n_slices": len(sel),
        "max_conf": max_conf,
        "max_lesion_pct": max_lesion_pct,
        "best_overlay": best_overlay,
    }


def save_error_sample(scan_name: str, overlay, out_dir: Path):
    if overlay is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_dir / f"{scan_name}.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CQ500_DIR.exists() or not any(CQ500_DIR.iterdir()):
        print(f"❌ CQ500 데이터 없음: {CQ500_DIR}")
        print(f"   먼저 scripts/download_cq500.py 또는 수동 다운로드 후 실행")
        sys.exit(1)

    # GT
    gt_map = parse_gt(CQ500_DIR / "reads.csv")
    if not gt_map:
        print(f"❌ reads.csv 없거나 파싱 실패")
        sys.exit(1)
    print(f"GT 라벨: {len(gt_map)}개 스캔")

    # Pipeline
    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt="./checkpoints/segmentor/best_segmentor.pth",
    )

    # 스캔 폴더 목록 (cq500/qctNN/CQ500CT{id}/ 구조 탐색)
    scan_dirs = find_scan_dirs(CQ500_DIR)
    print(f"스캔 폴더: {len(scan_dirs)}개")

    # 평가
    summary_csv = OUT_DIR / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scan", "gt", "pred", "max_conf", "max_lesion_pct", "n_slices"])

        y_true, y_pred = [], []
        fp_dir = OUT_DIR / "false_positives"
        fn_dir = OUT_DIR / "false_negatives"
        fp_count, fn_count = 0, 0

        skipped = 0
        for i, scan_dir in enumerate(scan_dirs, 1):
            name = normalize_scan_name(scan_dir.name)
            if name not in gt_map:
                skipped += 1
                continue
            gt = gt_map[name]
            r = evaluate_scan(pipe, scan_dir)
            pred = 1 if r["hemorrhagic"] else 0

            y_true.append(gt)
            y_pred.append(pred)

            w.writerow([name, gt, pred, f"{r['max_conf']:.3f}",
                        f"{r['max_lesion_pct']:.2f}", r["n_slices"]])

            if gt == 0 and pred == 1 and fp_count < 20:
                save_error_sample(name, r["best_overlay"], fp_dir)
                fp_count += 1
            elif gt == 1 and pred == 0 and fn_count < 20:
                # FN은 overlay 없음(정상 예측) → 원본만
                pass

            if i % 10 == 0:
                print(f"  {i}/{len(scan_dirs)} 처리됨 (skip={skipped})...", flush=True)

    # 메트릭
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = len(y_true)

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    acc = (tp + tn) / max(total, 1)

    report = f"""\
CQ500 평가 리포트 (총 {total} 스캔)
=====================================

Confusion matrix:
                예측 normal    예측 hemorrhagic
실제 normal       {tn:>6}         {fp:>6} ← 오탐(FP)
실제 hemorrhagic  {fn:>6}         {tp:>6} ← 누락(FN)

Sensitivity (Recall) : {sens:.4f}  (출혈을 출혈로)
Specificity          : {spec:.4f}  (정상을 정상으로)
PPV (Precision)      : {ppv:.4f}
NPV                  : {npv:.4f}
Accuracy             : {acc:.4f}

Normal 오탐율(FP rate)    : {fp / max(tn + fp, 1):.4f}
Hemorrhagic 누락율(FN rate): {fn / max(tp + fn, 1):.4f}
"""
    print(report)
    (OUT_DIR / "metrics.txt").write_text(report)
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
