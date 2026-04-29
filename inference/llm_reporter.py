"""
LLaMA 3.2 Vision 11B 기반 CT 판독 리포트 생성 (Ollama 로컬 서버)

사용 전 설정:
    brew install ollama
    ollama serve          # 터미널 1
    ollama pull llama3.2-vision:11b   # 터미널 2 (약 7GB)

사용법:
    from inference.llm_reporter import LLMReporter
    reporter = LLMReporter()
    report = reporter.analyze(pipeline_result, overlay_image_np)
    print(report)
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_MODEL = "llama3.2-vision:11b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODE = "fast"

MODE_DEFAULTS = {
    "fast": {
        "include_image": True,
        "max_side": 384,
        "jpeg_quality": 70,
        "num_predict": 220,
        "temperature": 0.2,
        "timeout": 45,
    },
    "balanced": {
        "include_image": True,
        "max_side": 512,
        "jpeg_quality": 80,
        "num_predict": 384,
        "temperature": 0.25,
        "timeout": 90,
    },
    "detailed": {
        "include_image": True,
        "max_side": 768,
        "jpeg_quality": 85,
        "num_predict": 700,
        "temperature": 0.3,
        "timeout": 180,
    },
    "text_only": {
        "include_image": False,
        "max_side": 0,
        "jpeg_quality": 0,
        "num_predict": 220,
        "temperature": 0.2,
        "timeout": 45,
    },
}

SYSTEM_PROMPT = (
    "You support a research-grade brain CT hemorrhage triage-assist pipeline. "
    "You are not making clinical diagnoses. Use the automated model outputs "
    "and optional overlay image only to summarize triage-support information. "
    "Do not recommend definitive treatment. Always state that qualified medical "
    "professionals make final clinical decisions."
)


@dataclass
class LLMReport:
    model: str
    mode: str
    raw_response: str
    elapsed_sec: float
    pipeline_class: str
    pipeline_confidence: float
    image_used: bool = True
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"[LLM 오류] {self.error}"
        lines = [
            "─" * 60,
            f"  LLM 보조 리포트 ({self.model}, mode={self.mode})",
            f"  파이프라인 입력: {self.pipeline_class.upper()} ({self.pipeline_confidence:.1%})",
            f"  이미지 입력: {'사용' if self.image_used else '미사용(text-only)'}",
            f"  응답 시간: {self.elapsed_sec:.1f}s",
            "─" * 60,
            self.raw_response,
            "─" * 60,
            "  ⚠  연구/교육 목적 전용 — 임상 진단이 아닙니다.",
        ]
        return "\n".join(lines)


def _mode_settings(mode: str, overrides: Optional[dict] = None) -> dict:
    if mode not in MODE_DEFAULTS:
        choices = ", ".join(sorted(MODE_DEFAULTS))
        raise ValueError(f"Unknown LLM mode: {mode}. Choose one of: {choices}")
    settings = dict(MODE_DEFAULTS[mode])
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})
    return settings


def _ndarray_to_jpeg_b64(image_np: np.ndarray, max_side: int = 384, quality: int = 70) -> str:
    img = Image.fromarray(image_np.astype(np.uint8))
    w, h = img.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _result_lines(result) -> list[str]:
    lines = [
        "## Automated Pipeline Results",
        f"- Classification: **{result.class_name.upper()}** "
        f"(confidence: {result.confidence:.1%})",
    ]

    if result.class_probs:
        prob_str = " | ".join(f"{k}={v:.3f}" for k, v in result.class_probs.items())
        lines.append(f"- Class probabilities: {prob_str}")

    if result.lesion_area_px:
        lines.append(
            f"- Lesion area: {result.lesion_area_px}px "
            f"({result.lesion_area_pct:.1f}% of brain region)"
        )
        lines.append(
            f"- Connected components: {result.kept_component_count}/"
            f"{result.lesion_component_count} kept"
        )
        lines.append(
            f"- Max component mean probability: {result.max_component_mean_prob:.3f}"
        )

    if result.decision_source != "classifier":
        lines.append(
            f"- Decision override: {result.decision_source} — {result.override_reason}"
        )
    return lines


def _build_user_prompt(result, mode: str, image_used: bool) -> str:
    lines = _result_lines(result)
    visual_context = (
        "An overlay image is attached. Red indicates the detected lesion area."
        if image_used
        else "No image is attached. Use only the structured pipeline values."
    )
    lines += [
        "",
        "## Your Task",
        visual_context,
    ]

    if mode in ("fast", "text_only"):
        lines += [
            "Write in Korean. Return exactly 3 short bullets:",
            "1. Triage summary for emergency workflow",
            "2. Evidence from pipeline values and overlay if available",
            "3. Limitation / next-check reminder",
            "Keep it under 120 Korean words. This is not a clinical diagnosis.",
        ]
    elif mode == "balanced":
        lines += [
            "Write in Korean. Provide 4 concise sections:",
            "1. Observation",
            "2. Pipeline consistency",
            "3. Triage-support note",
            "4. Limitation",
            "Keep it concise. This is research/education tooling, not a clinical report.",
        ]
    else:
        lines += [
            "Write in Korean. Provide a structured report:",
            "1. Key observation",
            "2. Model evidence",
            "3. Triage-support interpretation",
            "4. Uncertainty and possible false positive/false negative factors",
            "5. Recommended verification by qualified clinicians",
            "This is research/education tooling, not a clinical report.",
        ]
    return "\n".join(lines)


class LLMReporter:
    """LLaMA 3.2 Vision 11B를 Ollama 로컬 서버로 호출하는 CT triage 보조 리포터."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        mode: str = DEFAULT_MODE,
        timeout: Optional[int] = None,
        max_side: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        include_image: Optional[bool] = None,
    ):
        self.model = model
        self.host = host
        overrides = {
            "timeout": timeout,
            "max_side": max_side,
            "jpeg_quality": jpeg_quality,
            "num_predict": num_predict,
            "temperature": temperature,
            "include_image": include_image,
        }
        self.mode = mode
        self.settings = _mode_settings(mode, overrides)
        self.timeout = int(self.settings["timeout"])
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host, timeout=self.timeout)
            except ImportError as e:
                raise ImportError(
                    "ollama 패키지가 없습니다: pip install ollama\n"
                    "Ollama 서버도 필요합니다: https://ollama.com"
                ) from e
        return self._client

    def is_available(self) -> bool:
        """Ollama 서버와 모델이 준비됐는지 확인."""
        try:
            client = self._get_client()
            models = client.list()
            names = [
                getattr(m, "model", None) or getattr(m, "name", None)
                or (m.get("model", "") if isinstance(m, dict) else "")
                for m in models.models
            ]
            return any(self.model.split(":")[0] in (n or "") for n in names)
        except Exception:
            return False

    def analyze(
        self,
        result,
        overlay_image: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
        include_image: Optional[bool] = None,
    ) -> LLMReport:
        """
        PipelineResult와 이미지를 받아 LLM triage 보조 리포트를 반환.

        overlay_image: pipeline.result.overlay_image (병변 오버레이 포함)
        original_image: 원본 CT NumPy array (overlay가 없을 때 사용)
        """
        image_np = overlay_image if overlay_image is not None else original_image
        if image_np is None and hasattr(result, "overlay_image"):
            image_np = result.overlay_image
        should_use_image = bool(self.settings["include_image"] if include_image is None else include_image)
        should_use_image = should_use_image and image_np is not None

        t0 = time.time()
        try:
            client = self._get_client()
            prompt = _build_user_prompt(result, self.mode, should_use_image)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if should_use_image:
                image_b64 = _ndarray_to_jpeg_b64(
                    image_np,
                    max_side=int(self.settings["max_side"]),
                    quality=int(self.settings["jpeg_quality"]),
                )
                messages.append({
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                })
            else:
                messages.append({"role": "user", "content": prompt})

            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": float(self.settings["temperature"]),
                    "num_predict": int(self.settings["num_predict"]),
                },
            )
            raw = response.message.content.strip()
            elapsed = time.time() - t0

            return LLMReport(
                model=self.model,
                mode=self.mode,
                raw_response=raw,
                elapsed_sec=elapsed,
                pipeline_class=result.class_name,
                pipeline_confidence=result.confidence,
                image_used=should_use_image,
            )

        except Exception as e:
            return LLMReport(
                model=self.model,
                mode=self.mode,
                raw_response="",
                elapsed_sec=time.time() - t0,
                pipeline_class=getattr(result, "class_name", "unknown"),
                pipeline_confidence=getattr(result, "confidence", 0.0),
                image_used=should_use_image,
                error=str(e),
            )

    def batch_analyze(self, results_with_images: list) -> list[LLMReport]:
        """
        [(result, overlay_np), ...] 리스트를 순차 처리.
        대용량 배치는 Ollama가 단일 스레드이므로 병렬 불필요.
        """
        reports = []
        for i, item in enumerate(results_with_images):
            result, overlay = item if len(item) == 2 else (item[0], None)
            print(f"  LLM 분석 [{i+1}/{len(results_with_images)}] "
                  f"{result.class_name} ({result.confidence:.1%})...", end=" ", flush=True)
            report = self.analyze(result, overlay_image=overlay)
            print(f"{report.elapsed_sec:.1f}s")
            reports.append(report)
        return reports
