# OOB_7_100base — Brain CT Triage Assist + LLM Report

OOB_test_7 epoch100 체크포인트를 베이스로, CT 기반 정상/뇌출혈 이진 분류, 병변 분할 오버레이, **LLaMA 3.2 Vision 11B triage 보조 리포트**를 결합한 응급실 초기 판단 보조 파이프라인.

> **연구/교육 목적 전용 — 임상 진단 도구가 아닙니다.**

## 현재 작업 기준

- 로컬 기준 작업 폴더는 `/Users/pke03/OOB_7_100base`입니다.
- `OOB_test_7_epoch100` 및 `/Users/pke03/OOB_test_7_epoch100-main`은 baseline 참고용으로만 사용합니다.
- 실행 전 확인:

```bash
cd /Users/pke03/OOB_7_100base
./venv/bin/python scripts/check_repo_context.py
```

---

## 베이스라인 성능 (val set 2,089장)

| 지표 | 값 |
|------|---:|
| Accuracy | 95.88% |
| Sensitivity (출혈 탐지율) | **92.80%** |
| Specificity (정상 식별율) | **97.71%** |
| Precision (PPV) | 96.01% |
| FP (정상 → 출혈 오탐) | 30 |
| FN (출혈 → 정상 누락) | 56 |

---

## 주요 개선 사항 (vs OOB_test_7)

| 항목 | 변경 내용 |
|------|----------|
| **LLM 보조 리포트** | LLaMA 3.2 Vision 11B (Ollama) — CT 오버레이 + 파이프라인 수치 → triage 보조 요약 |
| **분류기 학습** | 2-stage freeze→unfreeze, OneCycleLR, FocalLoss, AMP, gradient clipping |
| **세그멘터 학습** | TverskyBCELoss (FN 패널티 강화), AMP, positive/empty Dice 분리 리포트 |
| **체크포인트 선택** | threshold sweep → macro F1 + recall constraint 기반 자동 선택 |
| **모델 백본** | EfficientNet-B2 → **EfficientNet-B4**, 240px/320px |
| **추론 파이프라인** | config-driven threshold, decision_source, connected component 필터링 |
| **라이브 모니터** | `--log` argparse 파라미터화 (하드코딩 경로 제거) |

> M4 Pro 24GB MPS 안정 기본값은 classifier batch 32입니다. batch 64는 backward 단계에서 MPS private-pool OOM이 날 수 있습니다.

---

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/OOB-Out-of-Brain/OOB_7_100base.git
cd OOB_7_100base

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 데이터 다운로드

```bash
python scripts/download_data.py      # tekno21 + CT Hemorrhage + BHSD + AISD (~3.3GB)
```

### 3. 학습

```bash
# 분류기 (2-stage, ~1.5~2시간 / M4 Pro)
python training/train_classifier.py

# 세그멘터 (~30~50분 / M4 Pro)
python training/train_segmentor.py

# 실시간 모니터링 (별도 터미널)
python scripts/live_monitor.py --log logs/classifier_train.log
```

### 4. 추론

```bash
# 기본 추론
python demo.py --image path/to/ct.jpg

# LLM 보조 리포트 포함 (기본 fast 모드)
python demo.py --image path/to/ct.jpg --llm --llm-mode fast

# 가장 빠른 LLM fallback: 이미지 없이 pipeline 수치만 전달
python demo.py --image path/to/ct.jpg --llm --llm-mode text-only

# LLM 리포트 파일 저장
python demo.py --image path/to/ct.jpg --llm --llm-save
```

---

## LLaMA 3.2 Vision 11B 설정

```bash
# Ollama 설치 (최초 1회)
brew install ollama          # macOS
# curl -fsSL https://ollama.com/install.sh | sh  # Linux

# 모델 다운로드 (~7GB)
ollama pull llama3.2-vision:11b

# 서버 실행 (학습/추론 전에 실행해 두기)
ollama serve
```

### 단일 이미지 LLM 보조 리포트

```bash
python demo.py --image ct.jpg --llm --llm-mode fast
```

```
============================================================
  LLM 보조 리포트 (llama3.2-vision:11b, mode=fast)
  파이프라인 입력: HEMORRHAGIC (94.3%)
  이미지 입력: 사용
  응답 시간: 8.2s
────────────────────────────────────────────────────────────
1. Observation: The CT image shows a hyperdense region ...
2. Consistency: The automated classification aligns with ...
3. Uncertainty: The lesion boundary near the ...
────────────────────────────────────────────────────────────
  ⚠  연구/교육 목적 전용 — 임상 진단이 아닙니다.
```

### LLM 속도 모드

| 모드 | 용도 | 특징 |
|------|------|------|
| `fast` | 기본 데모/웹 응답 | 384px JPEG, 짧은 prompt/output, timeout 45s |
| `balanced` | 품질/속도 균형 | 512px JPEG, 중간 길이 리포트 |
| `detailed` | 발표/보고서용 | 더 큰 이미지와 긴 응답, 느림 |
| `text-only` | 가장 빠른 fallback | 이미지 없이 pipeline 수치만 LLM에 전달 |

### 배치 LLM 보조 리포트

```bash
python scripts/run_llm_report.py \
    --image_dir test_samples/ \
    --output_dir results/llm/ \
    --mode fast

# 출력: results/llm/{이미지명}.llm.txt + llm_summary.csv
```

---

## 파이프라인 구조

```
CT 이미지 (PNG/JPG)
      │
      ▼
┌─────────────────────────────────┐
│  StrokePipeline                 │
│  ┌──────────────────────────┐   │
│  │ EfficientNet-B4 분류기   │   │
│  │ → normal / hemorrhagic   │   │
│  │ → threshold sweep 선택   │   │
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │ U-Net + EfficientNet-B4  │   │
│  │ → 병변 마스크 + 오버레이  │   │
│  │ → connected component    │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
      │ PipelineResult
      ▼
┌─────────────────────────────────┐
│  LLMReporter (선택)             │
│  LLaMA 3.2 Vision 11B (Ollama) │
│  → triage 보조 자연어 요약       │
└─────────────────────────────────┘
      │
      ▼
시각화 PNG + (LLM 리포트 TXT)
```

---

## 평가 스크립트

```bash
# Val set 상세 평가 (FP/FN 샘플 저장)
python scripts/evaluate_valset.py
# → results/valset/metrics.txt + false_positives/ + false_negatives/

# 파이프라인 규칙 A/B/C 비교
python scripts/evaluate_valset_compare.py

# Val set 전체 4폴더 분류 저장 (TN/TP/FP/FN)
python scripts/save_all_valset_results.py

# 외부 CQ500 평가 (~28GB, aria2c 필요)
brew install aria2
python scripts/download_cq500.py
python scripts/evaluate_cq500.py
```

---

## 폴더 구조

```
OOB_7_100base/
├── demo.py                        # 단일 이미지 추론 (--llm 옵션)
├── config.yaml                    # 하이퍼파라미터 (EfficientNet-B4, AMP 등)
├── CLAUDE.md                      # AI 에이전트 프로젝트 가이드
│
├── data/
│   ├── combined_dataset.py        # 분류기 데이터로더
│   ├── ct_hemorrhage_dataset.py   # 세그멘터 데이터로더
│   └── raw/, processed/           # .gitignore (download_data.py로 받기)
│
├── models/
│   ├── classifier.py              # EfficientNet-B4, freeze/unfreeze, threshold
│   └── segmentor.py               # U-Net + EfficientNet-B4 encoder
│
├── training/
│   ├── train_classifier.py        # 2-stage, FocalLoss, OneCycleLR, AMP
│   ├── train_segmentor.py         # TverskyBCELoss, AMP, positive/empty Dice
│   ├── metrics.py                 # threshold_sweep, segmentation_stats, loss 함수
│   └── runtime.py                 # AMP helper, BatchProgressLogger, ETA
│
├── inference/
│   ├── pipeline.py                # StrokePipeline (분류 + 세그 + component 필터)
│   ├── visualization.py           # 오버레이 시각화
│   └── llm_reporter.py            # LLaMA 3.2 Vision triage 보조 리포터
│
├── scripts/
│   ├── download_data.py           # 학습용 데이터셋 자동 다운로드
│   ├── run_llm_report.py          # 배치 LLM 보조 리포트
│   ├── evaluate_valset.py         # Val set 상세 평가
│   ├── evaluate_valset_compare.py # 규칙 A/B/C 비교
│   ├── save_all_valset_results.py # 전체 2089장 4폴더 분류
│   ├── evaluate_cq500.py          # 외부 CQ500 평가
│   ├── validate.py                # threshold sweep + segmentation 검증
│   └── live_monitor.py            # 실시간 학습 모니터 (--log 파라미터)
│
├── checkpoints/                   # 학습된 모델 (.gitignore)
└── results/                       # 추론/평가 결과
    └── valset/metrics.txt         # 베이스라인 val set 결과
```

---

## 모델 선정 이유

### 분류기: EfficientNet-B4

| 후보 | 파라미터 | ImageNet Top-1 | 선택 이유 |
|------|:--------:|:--------------:|-----------|
| ResNet-50 | 25M | 76.1% | 기준선. 수용 영역이 좁아 전체 맥락 파악 부족 |
| EfficientNet-B2 | 9M | 80.1% | 베이스라인 모델 (ver7_100epoch) — 빠르지만 CQ500 특이도 48% |
| **EfficientNet-B4** | **19M** | **83.0%** | **채택** — B2 대비 +2.9%p 정확도, 240px 입력으로 세부 출혈 패턴 포착 |
| EfficientNet-B7 | 66M | 84.4% | 과적합 위험 ↑, M4 Pro 24GB에서 배치 64 불가 |
| ViT-B/16 | 86M | 81.8% | 데이터 수(~8K장)가 ViT 사전학습 fine-tune 최소선 미달 |

**선택 근거:**
- **CQ500 특이도 개선 목표(48% → 65%)**: B4의 더 깊은 feature 계층이 도메인 간 windowing 차이를 더 잘 분리
- **2-stage freeze→unfreeze**: B4 encoder를 처음 5 epoch 동결 후 차등 LR(×0.1)로 fine-tune → 의료 도메인에 안전한 전이학습
- **FocalLoss(γ=2)**: 정상 슬라이스가 출혈보다 많은 클래스 불균형 보정
- **threshold sweep**: argmax 대신 macro F1 + recall 제약으로 최적 결정 경계 선택

---

### 세그멘터: U-Net + EfficientNet-B4 Encoder

| 후보 | 특징 | 선택 이유 |
|------|------|-----------|
| U-Net + ResNet-34 | 경량, 빠름 | 베이스라인 (Dice 0.4665). 출혈 경계 해상도 부족 |
| **U-Net + EfficientNet-B4** | **skip connection 5단계** | **채택** — 분류기와 backbone 공유로 일관된 feature 공간 |
| DeepLabV3+ | Atrous convolution | 뇌 CT처럼 작고 분산된 병변에 U-Net이 더 적합 |
| Attention U-Net | 추가 파라미터 | 병변 크기 분포 다양 → 일반 U-Net 먼저 검증 후 고려 |

**선택 근거:**
- **TverskyBCELoss(α=0.3, β=0.7)**: 출혈 병변 미탐지(FN)에 더 큰 패널티 → 작은 병변 탐지율 향상
- **분류기와 동일 backbone(EfficientNet-B4)**: 전이학습 가중치 재사용, 도메인 적응 일관성
- **connected component 필터링**: 노이즈성 소형 위양성 영역 후처리 제거

---

## 학습 설정 (config.yaml 주요 값)

| 항목 | 분류기 | 세그멘터 |
|------|--------|---------|
| 백본 | EfficientNet-B4 | EfficientNet-B4 |
| 입력 크기 | 240×240 | 320×320 |
| 배치 크기 | 64 | 24 |
| 학습률 | 0.001 | 0.0001 |
| Loss | FocalLoss (γ=2) | TverskyBCELoss (α=0.3, β=0.7) |
| 스케줄러 | OneCycleLR | CosineAnnealingLR |
| AMP | ✅ | ✅ |
| Gradient Clip | 1.0 | 1.0 |

---

## 데이터셋

| 데이터셋 | 출처 | 크기 | 용도 |
|----------|------|------|------|
| tekno21 | HuggingFace `BTX24/tekno21-brain-stroke-dataset-multi` | ~560MB | 분류 |
| CT Hemorrhage | PhysioNet `ct-ich v1.3.1` | ~1.2GB | 분류 + 세그 |
| AISD (synthetic) | 로컬 생성 | ~110MB | 세그 보조 |
| BHSD | HuggingFace `WuBiao/BHSD` | ~1.4GB | 분류 + 세그 |
| CQ500 | qure.ai (CC BY-NC-SA 4.0) | ~28GB | 외부 평가 전용 |

---

## 예상 소요 시간 (MacBook M4 Pro)

| 작업 | 시간 |
|------|------|
| 데이터 다운로드 | 30분~1시간 |
| 분류기 학습 (50 epoch) | ~1.5~2시간 |
| 세그멘터 학습 (early stop) | ~30~50분 |
| 단일 추론 (파이프라인) | ~1~2초 |
| 단일 추론 (파이프라인 + LLM fast) | 환경에 따라 수 초~10초대 |
| Val set 전체 평가 (2089장) | ~10분 |

---

## 관련 레포

| 레포 | 설명 |
|------|------|
| [OOB_test_7_epoch100](https://github.com/OOB-Out-of-Brain/OOB_test_7_epoch100) | 이 레포의 베이스 (epoch100, 2-class) |
| [OOB_test_12](https://github.com/OOB-Out-of-Brain/OOB_test_12) | 2-class 개선 파이프라인 |
