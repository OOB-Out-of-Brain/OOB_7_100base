"""
AISD 구조와 동일한 합성 뇌 CT + 병변 마스크 데이터 생성.
실제 AISD 다운로드 전 파이프라인 검증용.

실행:
    python scripts/generate_synthetic_aisd.py --n_samples 400
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm


def make_brain_ct(size: int = 512, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r_skull = int(size * 0.44)
    r_brain = int(size * 0.40)

    Y, X = np.ogrid[:size, :size]

    # 두개골 링
    skull_mask = (X - cx)**2 + (Y - cy)**2 <= r_skull**2
    brain_mask = (X - cx)**2 + (Y - cy)**2 <= r_brain**2

    skull_ring = skull_mask & ~brain_mask
    img[skull_ring] = rng.integers(180, 220, size=3)

    # 뇌 조직 — 구역별로 다른 밀도
    brain_base = rng.integers(60, 90)
    img[brain_mask] = [brain_base, brain_base, brain_base]

    # 회백질 구조 노이즈
    noise = rng.integers(-15, 15, size=(size, size, 3))
    img = np.clip(img.astype(np.int16) + noise * brain_mask[:, :, None], 0, 255).astype(np.uint8)

    # 뇌실 (중앙 어두운 영역)
    r_vent = int(size * 0.07)
    vent_mask = (X - cx)**2 + (Y - cy)**2 <= r_vent**2
    img[vent_mask] = rng.integers(20, 40, size=3)

    return img, brain_mask


def make_ischemic_mask(size: int, brain_mask: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # 병변 중심: 뇌 내 랜덤 위치
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(0.05, 0.28) * size
    lx = int(cx + dist * np.cos(angle))
    ly = int(cy + dist * np.sin(angle))

    # 타원형 병변
    r_a = rng.integers(int(size * 0.04), int(size * 0.15))
    r_b = rng.integers(int(size * 0.03), int(size * 0.12))
    angle_rot = rng.uniform(0, np.pi)

    Y, X = np.ogrid[:size, :size]
    dx = X - lx
    dy = Y - ly
    cos_a, sin_a = np.cos(angle_rot), np.sin(angle_rot)
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a
    lesion = (x_rot / r_a)**2 + (y_rot / r_b)**2 <= 1
    lesion &= brain_mask

    # 불규칙한 경계
    from scipy.ndimage import binary_dilation, binary_erosion
    for _ in range(rng.integers(1, 4)):
        if rng.random() > 0.5:
            lesion = binary_dilation(lesion, iterations=rng.integers(1, 3))
        else:
            lesion = binary_erosion(lesion, iterations=rng.integers(1, 2))
    lesion &= brain_mask

    mask[lesion] = 255
    return mask


def main(args):
    out_root = Path("./data/raw/aisd")
    img_dir = out_root / "images"
    msk_dir = out_root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    size = 512

    print(f"합성 뇌 CT 데이터 {args.n_samples}개 생성 중...")
    for i in tqdm(range(args.n_samples)):
        img_np, brain_mask = make_brain_ct(size, rng)
        mask_np = make_ischemic_mask(size, brain_mask, rng)

        # 병변 밝기를 CT에 반영 (허혈성 → 어두운 영역)
        lesion_region = mask_np > 127
        img_np[lesion_region] = np.clip(img_np[lesion_region].astype(np.int16) - 20, 0, 255).astype(np.uint8)

        fname = f"synth_{i:04d}.png"
        Image.fromarray(img_np).save(img_dir / fname)
        Image.fromarray(mask_np).save(msk_dir / fname)

    print(f"\n완료!")
    print(f"  이미지: {img_dir}  ({args.n_samples}개)")
    print(f"  마스크: {msk_dir}  ({args.n_samples}개)")
    print("\n분할 학습 시작: python training/train_segmentor.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=400)
    args = parser.parse_args()
    main(args)
