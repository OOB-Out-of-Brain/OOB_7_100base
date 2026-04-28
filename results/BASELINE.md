# 베이스라인 성능 (ver7_100epoch 체크포인트)

체크포인트 출처: `OOB_test_2class-ver7_100epoch`  
모델: EfficientNet-B2 분류기 + U-Net/ResNet34 세그멘터  
판독 규칙: 분류기 단독 (A 규칙, post-processing 없음)

---

## 3개 테스트셋 결과

| 테스트셋 | 규모 | Accuracy | Sensitivity | Specificity | FP | FN |
|----------|------|:--------:|:-----------:|:-----------:|:--:|:--:|
| brain_test | 12장 | **100.00%** | 100.00% | 100.00% | 0 | 0 |
| Val set | 2,089장 | **95.88%** | 92.80% | 97.71% | 30 | 56 |
| CQ500 (외부) | 491 스캔 | **64.15%** | 85.99% | 48.24% | 147 | 29 |

---

## 체크포인트 상세

| 항목 | 분류기 | 세그멘터 |
|------|--------|---------|
| 파일 | `checkpoints/classifier/best_classifier.pth` | `checkpoints/segmentor/best_segmentor.pth` |
| 백본 | EfficientNet-B2 | U-Net + ResNet34 |
| 입력 크기 | 224×224 | 256×256 |
| Val 지표 | acc=0.9582 | dice=0.4665 |
| 클래스 | normal / hemorrhagic | binary mask |

---

## 개선 목표 (OOB_7_100base 재학습 기준)

| 지표 | 베이스라인 | 목표 |
|------|:----------:|:----:|
| Val Sensitivity | 92.80% | ≥ 93% |
| Val Specificity | 97.71% | ≥ 97% |
| CQ500 Specificity | **48.24%** ⚠️ | ≥ 65% |
| CQ500 Sensitivity | 85.99% | ≥ 85% (유지) |
| Segmentor Dice | 0.4665 | ≥ 0.55 |

### CQ500 Specificity 48% 원인 분석

- 정상 스캔 284개 중 147개(51.8%)를 출혈로 오탐
- 원인: 학습 데이터(tekno21+CT Hemorrhage) vs CQ500 도메인 불일치
  - DICOM 촬영 프로토콜 차이 (병원별 windowing)
  - 스캔 레벨 집계 규칙 (12 슬라이스 중 1개라도 hemorrhagic → 스캔 전체 hemorrhagic)
- 개선 방향: EfficientNet-B4 + FocalLoss + threshold sweep으로 FP 억제
