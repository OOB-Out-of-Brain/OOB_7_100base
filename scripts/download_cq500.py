"""CQ500 (qure.ai) 자동 다운로드 — 테스트 전용.

⚠️ CQ500 라이선스: CC BY-NC-SA 4.0 (학습 금지, 연구/평가 목적만).

다운로드 방법 (자동 선택):
  1. aria2c + Academic Torrents   (최고, 28GB, reads.csv 포함)
  2. Kaggle API (crawford/qureai-headct, 40GB, reads.csv 별도 필요)

실행:
    python scripts/download_cq500.py
    python scripts/download_cq500.py --method torrent
    python scripts/download_cq500.py --method kaggle

필요 도구 (자동 감지):
  - aria2c: `brew install aria2` 또는 `apt install aria2`
  - kaggle:  `pip install kaggle` 후 ~/.kaggle/kaggle.json 설정
"""

import argparse, subprocess, shutil, sys, os
from pathlib import Path


CQ500_DIR = Path("./data/raw/cq500")

TORRENT_URL = "https://academictorrents.com/download/47e9d8aab761e75fd0a81982fa62bddf3a173831.torrent"
MAGNET = ("magnet:?xt=urn:btih:47e9d8aab761e75fd0a81982fa62bddf3a173831"
          "&dn=CQ500&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce")

KAGGLE_DATASET = "crawford/qureai-headct"


def has(tool: str) -> bool:
    return shutil.which(tool) is not None


def download_torrent():
    """aria2c로 Academic Torrents에서 CQ500 받기 (reads.csv 포함)."""
    if not has("aria2c"):
        print("  ❌ aria2c 없음. 설치:")
        print("     macOS:  brew install aria2")
        print("     Ubuntu: sudo apt install aria2")
        return False

    CQ500_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n🌀 aria2c로 CQ500 torrent 다운로드 시작 (~28GB, 몇 시간 소요)")
    print(f"   저장: {CQ500_DIR}\n")

    cmd = [
        "aria2c",
        "--dir", str(CQ500_DIR),
        "--seed-time=0",
        "--max-connection-per-server=8",
        "--split=8",
        "--bt-enable-lpd=true",
        "--enable-dht=true",
        "--summary-interval=10",
        MAGNET,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ torrent 다운 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ aria2c 실패: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  중단됨 (이어받기 가능: 같은 명령 재실행)")
        return False


def download_kaggle():
    """Kaggle API 로 CQ500 받기. reads.csv는 별도 필요."""
    if not has("kaggle"):
        print("  ❌ kaggle CLI 없음:  pip install kaggle")
        return False
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"  ❌ 인증 없음. {kaggle_json} 에 Kaggle API token 필요")
        print("     https://www.kaggle.com/settings/account → Create New Token")
        return False

    CQ500_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Kaggle {KAGGLE_DATASET} 다운로드 (~40GB, 몇 시간)")

    cmd = ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
           "-p", str(CQ500_DIR), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Kaggle 다운 완료")
        print("⚠️  reads.csv(라벨)는 Kaggle 미러에 없음 — torrent 버전 권장")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Kaggle 실패: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["auto", "torrent", "kaggle"], default="auto")
    args = p.parse_args()

    print("=" * 60)
    print("  CQ500 외부 테스트 데이터셋 다운로드 (테스트 전용)")
    print("=" * 60)

    # 이미 있으면 스킵
    reads_csv = CQ500_DIR / "reads.csv"
    qct_any = list(CQ500_DIR.glob("*/Unknown Study"))[:1] if CQ500_DIR.exists() else []
    if reads_csv.exists() and qct_any:
        n = len([d for d in CQ500_DIR.iterdir() if d.is_dir() and d.name.startswith("CQ500CT")])
        print(f"✅ 이미 준비됨: {n}개 스캔 + reads.csv")
        print(f"   평가:  python scripts/evaluate_cq500.py")
        return

    # 방법 선택
    method = args.method
    if method == "auto":
        if has("aria2c"):
            method = "torrent"
        elif has("kaggle") and (Path.home() / ".kaggle" / "kaggle.json").exists():
            method = "kaggle"
        else:
            print("""
❌ 자동 다운로드 도구 없음. 둘 중 하나 설치하세요:

  [옵션 1 — 추천] Academic Torrents (reads.csv 포함, 무인증)
     macOS:   brew install aria2
     Ubuntu:  sudo apt install aria2

  [옵션 2] Kaggle (40GB, Kaggle 계정 필요, reads.csv 별도)
     pip install kaggle
     https://www.kaggle.com/settings/account → API token 생성
     → ~/.kaggle/kaggle.json 에 저장

  이후 재실행: python scripts/download_cq500.py
""")
            sys.exit(1)

    print(f"\n방법: {method}")
    ok = download_torrent() if method == "torrent" else download_kaggle()
    if ok:
        print("\n이제 평가 실행:  python scripts/evaluate_cq500.py")


if __name__ == "__main__":
    main()
