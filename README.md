# Brain Stroke AI — CT 출혈 분석 (A 규칙 / 분류기 단독)

⭐ **이 버전(OOB_test_7)은 가장 추천되는 최종 버전입니다.**
분류기 단독으로 판독 → **Val set 2089장 기준 Accuracy 96.03%**.

- **Classifier**: EfficientNet-B2 (2-class)
- **Segmentor**: U-Net + ResNet34 encoder (마스크는 **시각화 용도로만** 사용)
- **Device**: Apple Silicon MPS / NVIDIA CUDA / CPU 자동 감지
- **Pipeline 규칙**: 분류기 softmax 출력 그대로 사용 (post-processing 없음)

---

## 1. Quick Start

```bash
git clone https://github.com/OOB-Out-of-Brain/OOB_test_7.git
cd OOB_test_7

python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py       # 학습용 4개 데이터셋 ~3.3GB 자동
python training/train_classifier.py   # 분류기 학습 (~1.5~2시간)
python training/train_segmentor.py    # 분할기 학습 (~30분)

python demo.py --image path/to/ct.jpg # 추론
```

---

## 2. 데이터셋 가이드

### 2-1. 학습용 (필수) — `download_data.py` 한 번

| 데이터셋 | 출처 | 용량 | 자동 | 역할 |
|---|---|---|---|---|
| tekno21 | HuggingFace `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | ✅ | 분류 |
| CT Hemorrhage | PhysioNet `ct-ich v1.0.0` | ~1.2GB | ✅ | 분류 + 분할 |
| AISD (synthetic) | 로컬 생성 | ~110MB | ✅ | 분할 보조 |
| BHSD | HuggingFace `WuBiao/BHSD` | ~1.4GB | ✅ | 분류 + 분할 (subtype 5종 → binary) |

**BHSD 처리**: 3D NIfTI → brain window (center 40, width 80 HU) → 2D PNG 슬라이스.
라벨 5종(EDH/IPH/IVH/SAH/SDH)은 전부 `1=hemorrhagic` 으로 통합.

### 2-2. 외부 테스트셋 (선택, **학습 금지**)

| 데이터셋 | 출처 | 용량 | 자동 | 용도 |
|---|---|---|---|---|
| **CQ500** | qure.ai | ~28GB | ✅ **완전 자동** (aria2c 설치 후) | 외부 일반화 테스트 |

**한 줄로 받고 바로 평가:**

```bash
brew install aria2                        # 최초 1회 (macOS). Ubuntu는 sudo apt install aria2
python scripts/download_cq500.py          # Academic Torrents 자동 (28GB, reads.csv 포함)
python scripts/evaluate_cq500.py          # 491 스캔 평가 → results/cq500/ 에 FP/FN 리포트
```

aria2c 없으면 Kaggle 대안 (40GB, `pip install kaggle` + `~/.kaggle/kaggle.json` 필요):
```bash
python scripts/download_cq500.py --method kaggle
```

**라이선스**: CC BY-NC-SA 4.0 — 연구/평가 용도만, 학습 금지.

---

## 3. 테스트 실행 방법

### 3-1. 단일 이미지

```bash
python demo.py --image path/to/ct.jpg
# → results/{파일명}_result.png
```

### 3-2. 폴더 배치 (대량)

```bash
python scripts/run_batch_test.py \
    --input-dir /path/to/images \
    --output-dir results/my_test/
```

**출력**: 각 이미지 결과 PNG + 터미널 요약 (normal/hemorrhagic, 신뢰도, 병변 크기).

### 3-3. Val set 전체 상세 평가 (2089장)

```bash
python scripts/evaluate_valset.py
# → results/valset/metrics.txt + summary.csv + FP/FN 샘플
```

### 3-4. 3가지 Pipeline 규칙 비교

```bash
python scripts/evaluate_valset_compare.py
# → results/valset/rule_comparison.txt
```

### 3-5. 전체 2089장 4폴더 분류 저장

```bash
python scripts/save_all_valset_results.py
# → results/valset_all/{correct_normal,correct_hemorrhagic,false_positives,false_negatives}/*.png
```

### 3-6. 외부 CQ500 평가 (데이터 수동 다운로드 후)

```bash
python scripts/evaluate_cq500.py
# → results/cq500/metrics.txt + FP/FN 샘플
```

---

## 4. Pipeline 판독 규칙 (A 규칙, 이 버전)

```
[이미지] → [분류기(EfficientNet-B2)] → softmax 확률 → 최종 판독
       ↓ (선택, 시각화용)
       [세그멘터(U-Net)] → 병변 마스크 → overlay 이미지
```

분류기 판단을 **그대로 신뢰**. 세그멘터는 "어디가 출혈인가?" 만 시각화.

### 성능 (val set 2089장)

| 지표 | A 규칙 (이 버전) | B 규칙 (OOB_test_6, 1% threshold) |
|---|---:|---:|
| **Accuracy** | **96.03%** | 72.00% |
| Sensitivity (출혈 탐지) | **93.32%** | 28.02% ⚠️ |
| Specificity (정상 식별) | 97.64% | **98.09%** |
| FP (오탐) | 31 | **25** |
| FN (누락, 임상 위험) | **52** | 560 ⚠️ |

**왜 A 규칙이 더 좋은가?**
- 분류기는 8621장(CT+tekno21+BHSD)으로 학습되어 자체 Acc 96%
- 세그멘터 Dice 0.56 수준이라 **작은 출혈 마스크를 못 잡음**
- 1% 규칙을 쓰면 분류기가 맞게 예측해도 pipeline 이 뒤집어 normal 만들어 **FN 급증**
- 시각화는 세그 결과 그대로 사용하면 되므로 **분류와 분리하는 게 맞음**

---

## 5. 폴더 구조

```
OOB_test_7/
├─ README.md                  # 이 문서 (A 규칙)
├─ HOWTRAIN.md                # 학습 상세 가이드
├─ config.yaml                # 하이퍼파라미터
├─ demo.py                    # 단일 이미지 추론
│
├─ data/
│  ├─ combined_dataset.py       # 분류기 로더
│  ├─ ct_hemorrhage_dataset.py  # 분할기 로더
│  └─ raw/, processed/          # gitignore (download_data.py 로 받음)
│
├─ models/{classifier,segmentor}.py
├─ training/{train_classifier,train_segmentor,metrics}.py
├─ inference/
│  ├─ pipeline.py              # ⭐ 분류기 단독 판독 (post-processing 없음)
│  └─ visualization.py
│
├─ scripts/
│  ├─ download_data.py         # 학습용 4개 데이터셋
│  ├─ download_bhsd.py / preprocess_bhsd.py
│  ├─ generate_synthetic_aisd.py
│  ├─ download_cq500.py        # CQ500 시도 + 수동 안내
│  ├─ evaluate_cq500.py        # 외부 테스트 평가
│  ├─ run_batch_test.py        # 폴더 배치
│  ├─ evaluate_valset.py       # val set 상세
│  ├─ evaluate_valset_compare.py  # 규칙 비교
│  └─ save_all_valset_results.py  # 2089장 4폴더 분류
│
├─ checkpoints/               # 학습 결과 (gitignore)
└─ results/                   # 추론 결과
```

---

## 6. 예상 소요 시간 (MacBook M 시리즈)

| 작업 | 시간 |
|---|---|
| `download_data.py` (3.3GB) | 30분~1시간 |
| `train_classifier.py` (50 epoch) | ~1.5~2시간 |
| `train_segmentor.py` (early stop) | ~20~40분 |
| 단일 추론 | 1~2초 |
| Val set 전체 평가 (2089장) | ~10분 |

---

## 7. 트러블슈팅

- **PhysioNet zip 실패** → https://physionet.org/content/ct-ich/1.0.0/ 수동 다운로드 후 `data/raw/ct_hemorrhage/` 에 압축 해제
- **BHSD 느림** → `huggingface-cli login`
- **CQ500 자동 실패** → 이메일 등록 방식이라 정상 (수동 필요)
- **GPU OOM** → `config.yaml` 의 `batch_size` 절반
- **MPS 미지원** → 자동 CPU 전환 (느림)

---

## 8. 관련 레포

- **`OOB_test_6`** : 1% threshold 규칙 버전 (스크리닝용, specificity ↑)
- **`OOB_test_7`** (이 레포) : 분류기 단독 (정확도 ↑, 권장)
- 이전 실험: `OOB_test_1~5`
