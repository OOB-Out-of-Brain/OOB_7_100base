# 학습 실행 가이드 (HOW TO TRAIN)

뇌 CT 영상의 **출혈 여부(2-class)** 분류 + 병변 분할 파이프라인입니다.
분류기: EfficientNet-B2 · 분할기: U-Net (ResNet34 encoder).
Apple Silicon(MPS) / CUDA / CPU 자동 감지.

---

## 1. 레포 클론 & 가상환경

```bash
git clone <이 레포 URL>
cd OOB_test_5

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.10+ 권장**, PyTorch 2.0 이상 필요.

---

## 2. 데이터셋 자동 다운로드

```bash
python scripts/download_data.py
```

이 스크립트가 4개 데이터셋을 준비합니다:

| 데이터셋 | 출처 | 용량 | 방식 |
|---|---|---|---|
| tekno21 | HuggingFace (`BTX24/tekno21-brain-stroke-dataset-multi`) | ~560MB | 자동 |
| CT Hemorrhage | PhysioNet (`ct-ich v1.0.0`) | ~1.2GB | 자동 (zip) |
| AISD (synthetic) | 로컬 생성 | ~109MB | 자동 |
| BHSD | HuggingFace (`WuBiao/BHSD`) | ~1.4GB | 자동 |

총 **약 3.3GB** 다운로드 + 전처리됩니다.

**개별 실행도 가능:**

```bash
python scripts/download_bhsd.py       # BHSD 원본만
python scripts/preprocess_bhsd.py     # BHSD NIfTI → 2D PNG
python scripts/generate_synthetic_aisd.py   # AISD 합성
```

**주의:** BHSD는 3D NIfTI 볼륨이라 2D 슬라이스로 자를 때 **brain window (center 40, width 80)** 를 적용하고, 라벨 1~5(EDH/IPH/IVH/SAH/SDH)은 모두 **이진 `1=hemorrhagic`** 으로 통합됩니다 (subtype 정보는 버림).

---

## 3. 학습

### 3-1. 분류기 (EfficientNet-B2, 2-class)

```bash
python training/train_classifier.py
```

- 데이터: CT Hemorrhage + tekno21 + BHSD 통합 (~8600장)
- 설정: 50 epoch, batch 16, lr 1e-3, image 224×224
- 체크포인트: `checkpoints/classifier/best_classifier.pth`
- 옵션: `--epochs 30 --batch_size 8 --lr 1e-4`

### 3-2. 분할기 (U-Net + ResNet34, binary mask)

```bash
python training/train_segmentor.py
```

- 데이터: CT Hemorrhage 마스크 + BHSD 마스크 통합 (~2600장)
- 설정: 80 epoch, batch 8, lr 1e-4, image 256×256
- early stopping patience 15
- 체크포인트: `checkpoints/segmentor/best_segmentor.pth`

**디바이스 자동 선택 순서:** MPS(Apple Silicon) → CUDA → CPU.

**예상 시간 (MacBook M 시리즈 MPS 기준):**
- 분류기 50 epoch: 약 1.5~2시간
- 분할기 80 epoch(early stopping): 약 20~40분

---

## 4. 추론 (학습 끝난 후)

```bash
python demo.py --image path/to/ct.jpg
python demo.py --image input.jpg --output results/my_result.png
```

결과 이미지는 기본값 `results/{파일명}_result.png` 에 저장.

---

## 5. 폴더 구조

```
OOB_test_5/
  config.yaml                # 하이퍼파라미터 + 데이터 경로
  requirements.txt
  demo.py                    # 추론 데모
  HOWTRAIN.md                # 이 문서

  data/
    raw/                     # (gitignore: 다운로드로 받음)
      tekno21/
      ct_hemorrhage/.../Patients_CT/
      aisd/{images,masks}/
      bhsd/label_192/{images,ground truths}/
    processed/bhsd/          # (gitignore) 2D 슬라이스 PNG
    combined_dataset.py      # 분류기 통합 로더 (CT+tekno21+BHSD)
    ct_hemorrhage_dataset.py # 분할기 통합 로더 (CT+BHSD)

  models/
    classifier.py            # EfficientNet-B2 wrapper
    segmentor.py             # U-Net (smp)

  training/
    train_classifier.py
    train_segmentor.py
    metrics.py               # Dice, IoU, DiceBCE

  inference/
    pipeline.py              # StrokePipeline (classifier → segmentor → overlay)
    visualization.py

  scripts/
    download_data.py         # 전체 데이터 통합 다운로드 (메인)
    download_bhsd.py
    preprocess_bhsd.py
    generate_synthetic_aisd.py
    validate.py              # 학습 후 리포트 생성

  checkpoints/               # (gitignore) 학습 결과
  results/                   # 추론 결과 (사용자 생성)
  archive/                   # 이전 실험 결과 보관
```

---

## 6. 라벨 구성

**이진 분류 (2-class)** — 모든 데이터셋을 `normal(0)` / `hemorrhagic(1)` 로 통합:

| 원본 데이터셋 | 원본 라벨 | 매핑 |
|---|---|---|
| CT Hemorrhage | `No_Hemorrhage=1` | → 0 (normal) |
| CT Hemorrhage | `No_Hemorrhage=0` | → 1 (hemorrhagic) |
| tekno21 | `İnme Yok` (정상) | → 0 (normal) |
| tekno21 | `Kanama` (출혈) | → 1 (hemorrhagic) |
| tekno21 | `iskemi` (허혈) | **제외** (2-class 범위 밖) |
| BHSD | 라벨 1~5 (EDH/IPH/IVH/SAH/SDH) | → 1 (hemorrhagic, **subtype 통합**) |
| BHSD | 라벨 0 (배경) | 스킵 (출혈 슬라이스만 사용) |

Normal 샘플은 BHSD에 없으므로 CT Hemorrhage + tekno21 에서만 가져옵니다.

---

## 7. 트러블슈팅

- **"datasets 모듈 없음"** → `pip install -r requirements.txt`
- **"MPS 디바이스 없음" (macOS Intel/Linux)** → 자동으로 CUDA 또는 CPU로 동작
- **PhysioNet zip 다운로드 실패** → 브라우저에서 https://physionet.org/content/ct-ich/1.0.0/ 방문해 수동 다운로드 후 `data/raw/ct_hemorrhage/` 에 압축 해제
- **BHSD 다운로드 느림** → HuggingFace CDN 상태 확인, 또는 `huggingface-cli login` 으로 인증
- **학습 OOM (GPU 메모리 부족)** → `config.yaml` 에서 `batch_size` 절반으로 감소

---

## 8. 검증 리포트

학습 완료 후 상세 검증 리포트 생성:

```bash
python scripts/validate.py
```

- 분류기: precision, recall, F1, 혼동 행렬
- 분할기: Dice, IoU
