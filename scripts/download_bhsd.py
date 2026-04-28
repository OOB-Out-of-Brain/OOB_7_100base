"""BHSD (Brain Hemorrhage Segmentation Dataset) 다운로드.

출처: https://github.com/White65534/BHSD (HuggingFace mirror)
라이선스: NC-ND (비상업적 사용만).

출력:
  data/raw/bhsd/label_192/
    imagesTr/*.nii.gz   (train CT volumes)
    labelsTr/*.nii.gz   (train 세그멘테이션 마스크)
    imagesTs/*.nii.gz   (test CT volumes)
    labelsTs/*.nii.gz   (test 마스크)
  data/raw/bhsd/dataset.json
"""

import sys, urllib.request, zipfile
from pathlib import Path

BHSD_DIR = Path("./data/raw/bhsd")
ZIP_URL = "https://huggingface.co/datasets/WuBiao/BHSD/resolve/main/label_192.zip"
DATASET_JSON_URL = "https://raw.githubusercontent.com/White65534/BHSD/main/dataset.json"


def _report(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:5.1f}%  {mb:.1f}/{total_mb:.1f} MB")
        sys.stdout.flush()


def download(url: str, out_path: Path):
    if out_path.exists():
        print(f"  이미 존재: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  다운로드: {url}")
    print(f"    → {out_path}")
    urllib.request.urlretrieve(url, out_path, reporthook=_report)
    print()


def main():
    BHSD_DIR.mkdir(parents=True, exist_ok=True)

    # dataset.json
    print("[1] dataset.json")
    download(DATASET_JSON_URL, BHSD_DIR / "dataset.json")

    # label_192.zip
    print("\n[2] label_192.zip (192 volumes with pixel-level masks)")
    zip_path = BHSD_DIR / "label_192.zip"
    download(ZIP_URL, zip_path)

    # Unzip
    extract_dir = BHSD_DIR / "label_192"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print("\n[3] 압축 해제 중...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BHSD_DIR)
        print(f"  완료: {extract_dir}")
    else:
        print(f"\n[3] 이미 압축 해제됨: {extract_dir}")

    # 구조 확인
    print("\n[4] 구조 확인")
    for sub in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        p = extract_dir / sub
        if p.exists():
            n = len(list(p.glob("*.nii*")))
            print(f"  {sub}: {n}개")
        else:
            print(f"  {sub}: (없음)")


if __name__ == "__main__":
    main()
