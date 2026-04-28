"""모든 데이터셋 다운로드/준비 통합 스크립트.

준비하는 데이터셋:
  1. tekno21         : HuggingFace 자동 다운로드
  2. CT Hemorrhage   : PhysioNet에서 zip 직접 다운로드
  3. BHSD            : HuggingFace (별도 스크립트 호출)
  4. AISD (synthetic): 로컬 생성 (별도 스크립트)

실행:
    python scripts/download_data.py
"""

import sys, os, zipfile, subprocess, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


TEKNO21_CACHE   = Path("./data/raw/tekno21")
CT_HEM_DIR      = Path("./data/raw/ct_hemorrhage")
CT_HEM_UNPACKED = CT_HEM_DIR / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
CT_HEM_ZIP_URL  = "https://physionet.org/static/published-projects/ct-ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0.zip"

AISD_DIR        = Path("./data/raw/aisd")

BHSD_DIR        = Path("./data/raw/bhsd/label_192")
BHSD_PROCESSED  = Path("./data/processed/bhsd")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()


# ── 1. tekno21 ───────────────────────────────────────────────────────────────
def check_tekno21() -> bool:
    print("\n[1] tekno21 (HuggingFace)")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "BTX24/tekno21-brain-stroke-dataset-multi",
            split="train",
            cache_dir=str(TEKNO21_CACHE),
        )
        print(f"  ✅ tekno21: {len(ds)}개 슬라이스")
        return True
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return False


# ── 2. CT Hemorrhage (PhysioNet) ─────────────────────────────────────────────
def check_ct_hemorrhage() -> bool:
    print("\n[2] CT Hemorrhage (PhysioNet)")
    csv_path = CT_HEM_UNPACKED / "hemorrhage_diagnosis.csv"
    if csv_path.exists():
        print(f"  ✅ 이미 있음: {CT_HEM_UNPACKED}")
        return True

    zip_path = CT_HEM_DIR / "ct_hemorrhage.zip"
    CT_HEM_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"  다운로드 중 (약 1.2GB, 시간 걸림)...")
        print(f"    URL: {CT_HEM_ZIP_URL}")
        try:
            urllib.request.urlretrieve(CT_HEM_ZIP_URL, zip_path, reporthook=_progress)
            print()
        except Exception as e:
            print(f"\n  ❌ 다운로드 실패: {e}")
            print(f"  수동 다운로드: https://physionet.org/content/ct-ich/1.0.0/")
            return False

    print("  압축 해제 중...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CT_HEM_DIR)
    print(f"  ✅ 완료: {CT_HEM_UNPACKED}")
    return True


# ── 3. AISD (synthetic) ──────────────────────────────────────────────────────
def check_aisd() -> bool:
    print("\n[3] AISD (synthetic)")
    images_dir = AISD_DIR / "images"
    masks_dir = AISD_DIR / "masks"
    if images_dir.exists() and len(list(images_dir.glob("*.png"))) > 0:
        print(f"  ✅ 이미 있음 ({len(list(images_dir.glob('*.png')))}개 이미지)")
        return True

    gen_script = Path(__file__).parent / "generate_synthetic_aisd.py"
    if gen_script.exists():
        print(f"  synthetic AISD 생성 중 (generate_synthetic_aisd.py)...")
        try:
            subprocess.run([sys.executable, str(gen_script)], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 생성 실패: {e}")
    print(f"""
  ⚠️ 합성 AISD 생성 스크립트 실패 또는 없음.
  실제 AISD는 https://github.com/GriffinLiang/AISD 에서 수동 요청.
""")
    return False


# ── 4. BHSD ──────────────────────────────────────────────────────────────────
def check_bhsd() -> bool:
    print("\n[4] BHSD (HuggingFace)")
    if BHSD_DIR.exists() and (BHSD_DIR / "images").exists():
        n = len(list((BHSD_DIR / "images").glob("*.nii.gz")))
        print(f"  ✅ 원본 이미 있음 ({n}개 볼륨)")
    else:
        print("  원본 다운로드 → scripts/download_bhsd.py 실행...")
        script = Path(__file__).parent / "download_bhsd.py"
        try:
            subprocess.run([sys.executable, str(script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 실패: {e}")
            return False

    # 전처리된 PNG 슬라이스 확인
    if BHSD_PROCESSED.exists() and (BHSD_PROCESSED / "index.csv").exists():
        n_img = len(list((BHSD_PROCESSED / "images").glob("*.png")))
        print(f"  ✅ 전처리 완료 ({n_img}개 슬라이스)")
    else:
        print("  2D 슬라이스 전처리 → scripts/preprocess_bhsd.py 실행...")
        script = Path(__file__).parent / "preprocess_bhsd.py"
        try:
            subprocess.run([sys.executable, str(script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 전처리 실패: {e}")
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  뇌졸중 AI 프로젝트 — 전체 데이터셋 준비")
    print("=" * 60)

    results = {
        "tekno21":       check_tekno21(),
        "CT_hemorrhage": check_ct_hemorrhage(),
        "AISD":          check_aisd(),
        "BHSD":          check_bhsd(),
    }

    print("\n" + "=" * 60)
    print("  요약")
    print("=" * 60)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    if all(results.values()):
        print("\n모든 데이터 준비 완료. 학습 시작:")
        print("  python training/train_classifier.py")
        print("  python training/train_segmentor.py")
    else:
        print("\n일부 데이터셋 실패. 위 안내 확인 후 재실행하세요.")


if __name__ == "__main__":
    main()
