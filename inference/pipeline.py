"""
3단계 추론 파이프라인:
  1. 분류 (Normal / Hemorrhagic)
  2. 병변 분할 (U-Net)
  3. 시각화 결과 반환
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label as label_components

from models.classifier import StrokeClassifier
from models.segmentor import StrokeSegmentor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PipelineResult:
    # 분류 결과
    class_idx: int
    class_name: str
    confidence: float
    class_probs: dict = field(default_factory=dict)
    classifier_class_idx: int = 0
    classifier_class_name: str = ""
    classifier_confidence: float = 0.0
    decision_source: str = "classifier"
    override_reason: Optional[str] = None

    # 분할 결과
    lesion_prob_map: Optional[np.ndarray] = None  # H×W float32 probability
    lesion_mask: Optional[np.ndarray] = None   # H×W binary float32
    raw_lesion_area_px: int = 0
    raw_lesion_area_pct: float = 0.0
    lesion_area_px: int = 0
    lesion_area_pct: float = 0.0
    lesion_component_count: int = 0
    kept_component_count: int = 0
    max_component_area_px: int = 0
    max_component_mean_prob: float = 0.0
    segmentation_confidence: float = 0.0

    # 시각화
    overlay_image: Optional[np.ndarray] = None  # H×W×3 uint8

    def __str__(self):
        lines = [
            f"분류 결과  : {self.class_name.upper()} (신뢰도 {self.confidence:.1%})",
            f"클래스 확률: " + " | ".join(
                f"{k}={v:.3f}" for k, v in self.class_probs.items()
            ),
        ]
        if self.lesion_mask is not None:
            lines.append(f"병변 면적  : {self.lesion_area_px}px ({self.lesion_area_pct:.1f}%)")
            lines.append(
                f"병변 컴포넌트: {self.kept_component_count}/{self.lesion_component_count} "
                f"(max_prob={self.max_component_mean_prob:.3f})"
            )
        if self.decision_source != "classifier":
            lines.append(f"판단 근거  : {self.decision_source} ({self.override_reason})")
        return "\n".join(lines)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class StrokePipeline:
    def __init__(self,
                 classifier_ckpt: str,
                 segmentor_ckpt: str,
                 cls_image_size: Optional[int] = None,
                 seg_image_size: Optional[int] = None,
                 seg_threshold: Optional[float] = None,
                 min_component_px: Optional[int] = None,
                 min_component_area_pct: Optional[float] = None,
                 min_component_mean_prob: Optional[float] = None,
                 override_min_area_pct: Optional[float] = None,
                 device: Optional[torch.device] = None):

        self.device = device or _get_device()

        self.classifier = self._load_classifier(classifier_ckpt)
        self.segmentor = self._load_segmentor(segmentor_ckpt)
        self.cls_size = cls_image_size or self.classifier_cfg.get("image_size", 240)
        self.seg_size = seg_image_size or self.segmentor_cfg.get("image_size", 320)
        self.seg_threshold = float(
            seg_threshold if seg_threshold is not None
            else self.segmentor_cfg.get("seg_threshold", 0.5)
        )
        self.min_component_px = int(
            min_component_px if min_component_px is not None
            else self.segmentor_cfg.get("min_component_px", 16)
        )
        self.min_component_area_pct = float(
            min_component_area_pct if min_component_area_pct is not None
            else self.segmentor_cfg.get("min_component_area_pct", 0.0)
        )
        self.min_component_mean_prob = float(
            min_component_mean_prob if min_component_mean_prob is not None
            else self.segmentor_cfg.get("min_component_mean_prob", 0.0)
        )
        self.override_min_area_pct = float(
            override_min_area_pct if override_min_area_pct is not None
            else self.segmentor_cfg.get("override_min_area_pct", 0.5)
        )

        self.cls_transform = A.Compose([
            A.Resize(self.cls_size, self.cls_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
        self.seg_transform = A.Compose([
            A.Resize(self.seg_size, self.seg_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _load_classifier(self, ckpt_path: str) -> StrokeClassifier:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.classifier_cfg = cfg
        self.class_names = ckpt.get("class_names", cfg.get("class_names", ["normal", "hemorrhagic"]))
        self.cls_threshold = cfg.get("cls_threshold", 0.5)
        model = StrokeClassifier(
            model_name=cfg.get("model_name", "efficientnet_b4"),
            num_classes=len(self.class_names),
            pretrained=False,
            dropout_rate=cfg.get("dropout_rate", 0.3),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model

    def _load_segmentor(self, ckpt_path: str) -> StrokeSegmentor:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        self.segmentor_cfg = cfg
        model = StrokeSegmentor(
            encoder_name=cfg.get("encoder", "resnet34"),
            encoder_weights=None,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def run(self, image_input, make_overlay: bool = True) -> PipelineResult:
        """
        image_input: 파일 경로(str/Path), PIL.Image, 또는 np.ndarray (H×W×3 uint8)
        make_overlay: False면 대량 배치 추론에서 시각화 생성 비용을 건너뜁니다.
        """
        orig_np = self._load_image(image_input)

        # ── 1단계: 분류 ──────────────────────────────────────────────────────
        cls_tensor = self.cls_transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
        pred_idx, probs = self.classifier.predict(cls_tensor, threshold=self.cls_threshold)

        pred_idx = pred_idx.item()
        probs_np = probs.cpu().numpy()[0]
        class_name = self.class_names[pred_idx]

        result = PipelineResult(
            class_idx=pred_idx,
            class_name=class_name,
            confidence=float(probs_np[pred_idx]),
            class_probs={name: float(p) for name, p in zip(self.class_names, probs_np)},
            classifier_class_idx=pred_idx,
            classifier_class_name=class_name,
            classifier_confidence=float(probs_np[pred_idx]),
        )

        # ── 2단계: 세그멘테이션 (항상 실행) ────────────────────────────────
        seg_tensor = self.seg_transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
        prob_tensor = self.segmentor.predict_proba(seg_tensor)
        prob_resized = self._resize_prob(
            prob_tensor[0, 0].cpu().numpy(),
            target_h=orig_np.shape[0],
            target_w=orig_np.shape[1],
        )
        raw_mask = (prob_resized >= self.seg_threshold).astype(np.float32)
        total_px = orig_np.shape[0] * orig_np.shape[1]
        raw_lesion_px = int(raw_mask.sum())
        raw_lesion_pct = raw_lesion_px / total_px * 100

        mask_resized, component_summary = self._filter_components(raw_mask, prob_resized)
        lesion_px = int(mask_resized.sum())
        lesion_pct = lesion_px / total_px * 100

        result.lesion_prob_map = prob_resized
        result.raw_lesion_area_px = raw_lesion_px
        result.raw_lesion_area_pct = raw_lesion_pct
        result.lesion_component_count = component_summary["component_count"]
        result.kept_component_count = component_summary["kept_count"]
        result.max_component_area_px = component_summary["max_area_px"]
        result.max_component_mean_prob = component_summary["max_mean_prob"]
        result.segmentation_confidence = component_summary["max_mean_prob"]

        # 세그멘테이션 필터를 통과한 병변 면적이 기준 이상이면 hemorrhagic으로 override.
        if (
            lesion_pct >= self.override_min_area_pct
            and class_name == "normal"
            and "hemorrhagic" in self.class_names
        ):
            hemorrhagic_idx = self.class_names.index("hemorrhagic")
            result.class_idx = hemorrhagic_idx
            result.class_name = "hemorrhagic"
            result.confidence = float(result.class_probs.get("hemorrhagic", result.confidence))
            result.decision_source = "segmentation_override"
            result.override_reason = (
                f"filtered_lesion_area_pct={lesion_pct:.2f}% >= "
                f"{self.override_min_area_pct:.2f}%, "
                f"max_component_mean_prob={result.max_component_mean_prob:.3f}"
            )

        if lesion_px > 0:
            result.lesion_mask = mask_resized
            result.lesion_area_px = lesion_px
            result.lesion_area_pct = lesion_pct

        # ── 3단계: 시각화 ────────────────────────────────────────────────────
        if make_overlay:
            from inference.visualization import visualize_result
            result.overlay_image = visualize_result(orig_np, result)

        return result

    def _load_image(self, inp) -> np.ndarray:
        if isinstance(inp, np.ndarray):
            return inp if inp.ndim == 3 else np.stack([inp] * 3, axis=-1)
        img = Image.open(inp).convert("RGB") if not isinstance(inp, Image.Image) else inp.convert("RGB")
        return np.array(img)

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        import cv2
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return (resized > 0.5).astype(np.float32)

    @staticmethod
    def _resize_prob(prob: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        import cv2
        resized = cv2.resize(prob, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(resized, 0.0, 1.0).astype(np.float32)

    def _filter_components(self, mask: np.ndarray, prob: np.ndarray):
        labeled, component_count = label_components(mask.astype(bool), structure=np.ones((3, 3)))
        kept = np.zeros_like(mask, dtype=bool)
        total_px = mask.shape[0] * mask.shape[1]
        max_area_px = 0
        max_mean_prob = 0.0
        kept_count = 0

        for component_idx in range(1, component_count + 1):
            component = labeled == component_idx
            area_px = int(component.sum())
            if area_px <= 0:
                continue
            area_pct = area_px / total_px * 100
            mean_prob = float(prob[component].mean())
            if area_px > max_area_px:
                max_area_px = area_px
            if mean_prob > max_mean_prob:
                max_mean_prob = mean_prob
            keep = (
                area_px >= self.min_component_px
                and area_pct >= self.min_component_area_pct
                and mean_prob >= self.min_component_mean_prob
            )
            if keep:
                kept[component] = True
                kept_count += 1

        summary = {
            "component_count": int(component_count),
            "kept_count": int(kept_count),
            "max_area_px": int(max_area_px),
            "max_mean_prob": float(max_mean_prob),
        }
        return kept.astype(np.float32), summary
