# OOB_7_100base — Shared Agent Rules

This repository is a Python medical-image research pipeline for brain CT hemorrhage classification and lesion segmentation. It is forked from OOB_test_7_epoch100 (2-class binary baseline, epoch 100 checkpoint, val accuracy 95.9%) and extended with improved training infrastructure.

## Canonical Repository Boundary

- Canonical local worktree: `/Users/pke03/OOB_7_100base`.
- Do all code edits, training, validation, inference smoke checks, logs, checkpoint writes, and result generation in this repository only.
- Treat `/Users/pke03/OOB_test_7_epoch100-main` and the upstream `OOB_test_7_epoch100` project as read-only baseline/reference material. Do not patch files, install dependencies, write logs, generate outputs, or run new training there unless the user explicitly asks for that repo by path.
- If a traceback, terminal prompt, config path, or script output mentions `/Users/pke03/OOB_test_7_epoch100-main`, stop and switch back to `/Users/pke03/OOB_7_100base` before making changes.
- Start local commands with `cd /Users/pke03/OOB_7_100base` and prefer `./venv/bin/python` over a Python interpreter from another OOB checkout.
- Run `./venv/bin/python scripts/check_repo_context.py` before debugging data-loading, training, validation, or checkpoint issues.

## Project Direction

- Product direction: **emergency-room CT triage assist platform**. The model pipeline is one part of a workflow that uploads CT images, returns binary hemorrhage triage, visualizes lesion regions, and optionally generates a concise LLM support report.
- Base model direction: **2-class binary** (normal / hemorrhagic) inherited from OOB_test_7.
- The training infrastructure (AMP, 2-stage freeze/unfreeze, FocalLoss, threshold sweep, TverskyBCELoss) has been ported from the 3-class local development branch.
- Future direction may expand to 3-class (normal / ischemic / hemorrhagic) — do not assume 2-class is permanent.
- Treat the goal as a reliable training and inference pipeline, not just a higher accuracy number.
- Optimize for: sensitivity (hemorrhagic recall), specificity, macro F1, precision, Dice, IoU, false-positive lesion area, and threshold behavior together.
- Do not present outputs as clinical diagnosis. This is research/development tooling.
- Use wording such as "triage support", "initial-priority assist", "follow-up verification reminder", and "specialist review recommended"; avoid definitive treatment recommendations.
- Do not accept random-weight or missing-checkpoint outputs as meaningful results.

## Baseline Performance (OOB_test_7 epoch100, val 2089 samples)

| Metric | Value |
|--------|-------|
| Sensitivity (recall) | 92.80% |
| Specificity | 97.71% |
| Precision (PPV) | 96.01% |
| Accuracy | 95.88% |
| FP rate (normal→hem) | 2.29% |
| FN rate (hem→normal) | 7.20% |

## Current Stack

- Python with PyTorch, torchvision, timm, segmentation-models-pytorch, Albumentations, scikit-learn, pandas, nibabel, matplotlib, and Hugging Face datasets.
- Classifier: EfficientNet wrapper in `models/classifier.py`; default EfficientNet-B4, 2-class.
- Segmentor: U-Net with EfficientNet-B4 encoder in `models/segmentor.py`.
- Training scripts: `training/train_classifier.py` and `training/train_segmentor.py`.
- Inference: `demo.py` → `inference/pipeline.py` → `inference/visualization.py`.
- Current local optimization target: MacBook Pro M4 Pro, 12-core CPU, 16-core GPU, 24GB unified memory.
- Default stable profile: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95`, `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85`, classifier batch 32 at 240px, segmentor batch 24 at 320px. Classifier batch 64 can exceed the MPS private-pool limit on M4 Pro 24GB during backward.

## Improved Infrastructure (ported from local dev branch)

- `training/runtime.py`: AMP autocast helper, device cache clearing, DataLoader runtime tuning, BatchProgressLogger, ETA formatting.
- `training/metrics.py`: `threshold_predictions`, `threshold_sweep`, `select_threshold_row`, `binary_classification_metrics`, `segmentation_stats` (positive/empty split), `FocalLoss`, `TverskyBCELoss`, `MulticlassDiceCELoss`.
- `training/train_classifier.py`: 2-stage backbone freeze→unfreeze, OneCycleLR, FocalLoss, AMP, gradient clipping, threshold sweep checkpoint selection.
- `training/train_segmentor.py`: TverskyBCELoss, AMP, gradient clipping, positive/empty Dice split reporting.
- `inference/pipeline.py`: config-driven image_size/threshold from checkpoint, `decision_source`/`override_reason` fields, connected component filtering, `lesion_prob_map`.
- `scripts/live_monitor.py`: argparse `--log` parameter (no more hardcoded paths).
- `scripts/validate.py`: threshold sweep + positive/empty segmentation stats.

## Change Safety Rules

- Preserve existing implemented behavior by default. Do not remove, hide, or replace a working feature unless the user explicitly asks for it.
- Before deleting operational features (resume support, live monitoring, progress display, ETA/finish-time estimates, logs, checkpoint snapshots, evaluation reports), ask for confirmation.
- When a change intentionally removes behavior, call it out in the final report with the file path and reason.

## Data And Path Rules

- 2-class label map: `0 = normal`, `1 = hemorrhagic`.
- tekno21: `Kanama → hemorrhagic (1)`, `İnme Yok → normal (0)`. iskemi (허혈) slices are excluded in 2-class mode.
- CT Hemorrhage: `No_Hemorrhage=1 → normal (0)`, `No_Hemorrhage=0 → hemorrhagic (1)`.
- BHSD (5 hemorrhage subtypes) → all map to hemorrhagic (1).
- CQ500: external evaluation only. Do not add to training.
- Keep raw and processed CT paths separate in config.yaml.
- Split CT and BHSD by patient/volume, not by slice, to avoid leakage.
- **CT Hemorrhage CSV 파일명**: v1.3.1 데이터셋은 `hemorrhage_diagnosis_raw_ct.csv`를 사용한다. v1.0.0은 `hemorrhage_diagnosis.csv`를 사용한다. `_collect_ct()` 및 `_collect_samples()` 구현 시 반드시 `hemorrhage_diagnosis_raw_ct.csv` 를 먼저 시도하고 없으면 `hemorrhage_diagnosis.csv`로 fallback하는 패턴을 사용해야 한다. 하드코딩 금지.
- Current local CT Hemorrhage data is v1.3.1 NIfTI layout: `ct_scans/*.nii`, `masks/*.nii`, and `hemorrhage_diagnosis_raw_ct.csv`. Do not assume the legacy `Patients_CT/<patient>/brain/*.jpg` layout is present.

## Training Rules

- Prefer sensitivity (hemorrhagic recall) with specificity constraint over raw accuracy.
- Select classifier checkpoints with threshold-aware validation metrics. The saved checkpoint carries the selected threshold.
- Report at least confusion matrix, sensitivity, specificity, PPV, accuracy for classification.
- Report Dice and IoU split into all-slice and positive-mask metrics, with empty-mask false-positive rate for segmentation.
- Report metrics from the saved best checkpoint, not the last epoch.
- Use validation threshold sweeps instead of treating 0.5 as a permanent constant.

## Inference Rules

- Keep classifier confidence separate from final decision confidence.
- If segmentation overrides a normal classifier result, record an override reason.
- The inference pipeline exposes `lesion_prob_map`, raw/filtered lesion area, component counts, and max component mean probability.

## LLM Integration (LLaMA 3.2 Vision 11B)

- LLM 연동 모듈: `inference/llm_reporter.py` (Ollama 로컬 서버 기반)
- 배치 스크립트: `scripts/run_llm_report.py`
- 역할: PipelineResult + CT 오버레이 이미지를 받아 응급실 triage 보조용 자연어 요약 생성 (연구/교육 목적)
- 모델: `llama3.2-vision:11b` (M4 Pro 24GB에서 Q4 quantization으로 동작, ~7GB)
- 기본 모드: `fast` (384px JPEG, 짧은 prompt/output, timeout 45s). `balanced`, `detailed`, `text_only` 모드를 지원한다.
- LLM은 항상 선택 기능이다. LLM timeout/error가 발생해도 classifier/segmentor 결과와 시각화는 정상 반환되어야 한다.
- 속도가 중요한 데모/웹 플로우에서는 `--llm-mode fast` 또는 `--llm-mode text-only` / `--llm-no-image`를 우선 사용한다.
- 설정:
  ```bash
  brew install ollama
  ollama serve
  ollama pull llama3.2-vision:11b
  ```
- 단일 이미지: `python demo.py --image ct.png --llm --llm-mode fast`
- 배치: `python scripts/run_llm_report.py --image_dir test_samples/ --mode fast`
- LLM은 파이프라인 판단을 대체하지 않음 — 분류/세그 결과에 대한 보조 소견만 제공
- LLM 오류 시 파이프라인 결과는 정상 반환되며 리포트만 생략됨

## Handoff Notes

- Before running long training or downloads, ask the user unless they explicitly requested it.
- Safe quick checks:
  - `./venv/bin/python scripts/check_repo_context.py`
  - `./venv/bin/python -m py_compile demo.py data/*.py models/*.py training/*.py inference/*.py scripts/*.py`
  - `./venv/bin/python demo.py --image test_samples/EDH1.jpg --output /tmp/oob7_demo.png`
- Default smoke commands:
  - `./venv/bin/python training/train_classifier.py --epochs 1 --batch_size 16`
  - `./venv/bin/python training/train_segmentor.py --epochs 1 --batch_size 4`
  - `./venv/bin/python scripts/validate.py`

## Reporting Rule

- Start reports with the final result summary.
- Use concise tables for metrics or before/after comparisons.
- Include commands run and whether training, download, or only smoke checks were performed.
