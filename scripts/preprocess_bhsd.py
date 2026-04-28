"""BHSD NIfTI 볼륨을 2D 슬라이스 PNG로 변환.

전처리:
  - HU raw → brain window (center=40, width=80) 클립 → [0, 255]
  - z축 슬라이싱
  - 출혈 픽셀(라벨 1~5)이 있는 슬라이스만 추출 (normal은 기존 데이터셋에서 사용)
  - 마스크: 라벨 > 0 을 binary (255), 0을 배경 (0)

출력:
  data/processed/bhsd/
    images/{patient_id}_s{slice_idx}.png       (H×W 흑백 CT)
    masks/{patient_id}_s{slice_idx}.png         (H×W binary 마스크)
    index.csv                                    (이미지-마스크 매핑)

실행:
  python scripts/preprocess_bhsd.py
"""

import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


SRC_DIR = Path("./data/raw/bhsd/label_192")
IMAGES_DIR = SRC_DIR / "images"
MASKS_DIR = SRC_DIR / "ground truths"

OUT_DIR = Path("./data/processed/bhsd")
OUT_IMG = OUT_DIR / "images"
OUT_MASK = OUT_DIR / "masks"
INDEX_CSV = OUT_DIR / "index.csv"

# 의료 표준 brain window
WINDOW_CENTER = 40
WINDOW_WIDTH = 80


def apply_brain_window(slice_hu: np.ndarray) -> np.ndarray:
    """HU 값 → 0-255 uint8 (brain window 40/80)."""
    lo = WINDOW_CENTER - WINDOW_WIDTH / 2  # 0
    hi = WINDOW_CENTER + WINDOW_WIDTH / 2  # 80
    clipped = np.clip(slice_hu, lo, hi)
    normalized = (clipped - lo) / (hi - lo) * 255.0
    return normalized.astype(np.uint8)


def process_volume(img_path: Path, mask_path: Path, pid: str) -> list:
    """(image_png_path, mask_png_path, n_hemorrhage_px) 리스트 반환."""
    img_nii = nib.load(img_path)
    mask_nii = nib.load(mask_path)

    img_arr = img_nii.get_fdata()   # (H, W, Z) 보통
    mask_arr = mask_nii.get_fdata()

    # z축이 마지막이라고 가정 (NIfTI 표준). 모양 동일 확인.
    if img_arr.shape != mask_arr.shape:
        return []

    n_slices = img_arr.shape[2]
    samples = []

    for z in range(n_slices):
        mask_slice = mask_arr[:, :, z]
        # 출혈 픽셀 (라벨 1~5) 있는지
        bin_mask = (mask_slice > 0).astype(np.uint8)
        if bin_mask.sum() == 0:
            continue  # skip non-hemorrhage slices

        img_slice = img_arr[:, :, z]
        img_u8 = apply_brain_window(img_slice)

        # NIfTI는 LPS 방향이라 시각적으로 올바르게 하려면 회전 필요
        img_u8 = np.rot90(img_u8)
        bin_mask = np.rot90(bin_mask)

        img_png = OUT_IMG / f"{pid}_s{z:03d}.png"
        mask_png = OUT_MASK / f"{pid}_s{z:03d}.png"
        Image.fromarray(img_u8).save(img_png)
        Image.fromarray(bin_mask * 255).save(mask_png)

        samples.append((img_png, mask_png, int(bin_mask.sum())))

    return samples


def main():
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    OUT_MASK.mkdir(parents=True, exist_ok=True)

    volumes = sorted(IMAGES_DIR.glob("*.nii.gz"))
    print(f"BHSD 볼륨: {len(volumes)}개\n")

    all_samples = []
    for vol_path in tqdm(volumes, desc="처리 중"):
        pid = vol_path.stem.replace(".nii", "")
        mask_path = MASKS_DIR / vol_path.name
        if not mask_path.exists():
            print(f"  ⚠️ 마스크 없음: {vol_path.name}")
            continue
        samples = process_volume(vol_path, mask_path, pid)
        all_samples.extend(samples)

    # index.csv 저장
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "lesion_px"])
        for img, msk, px in all_samples:
            writer.writerow([str(img.relative_to(OUT_DIR)),
                              str(msk.relative_to(OUT_DIR)),
                              px])

    print(f"\n완료.")
    print(f"  볼륨: {len(volumes)}")
    print(f"  출혈 슬라이스(양성): {len(all_samples)}")
    print(f"  평균 볼륨당: {len(all_samples) / max(len(volumes),1):.1f} 슬라이스")
    print(f"  출력 경로: {OUT_DIR}")


if __name__ == "__main__":
    main()
