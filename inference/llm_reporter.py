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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_MODEL = "llama3.2-vision:11b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are an AI assistant supporting a research-grade brain CT hemorrhage "
    "detection pipeline. You are NOT making clinical diagnoses — this is a "
    "research and educational tool only. Provide a concise, structured analysis "
    "based on the automated pipeline output and the CT image overlay. "
    "Always note that final clinical decisions must be made by qualified "
    "medical professionals."
)


@dataclass
class LLMReport:
    model: str
    raw_response: str
    elapsed_sec: float
    pipeline_class: str
    pipeline_confidence: float
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.error:
            return f"[LLM 오류] {self.error}"
        lines = [
            "─" * 60,
            f"  LLM 판독 ({self.model})",
            f"  파이프라인 입력: {self.pipeline_class.upper()} ({self.pipeline_confidence:.1%})",
            f"  응답 시간: {self.elapsed_sec:.1f}s",
            "─" * 60,
            self.raw_response,
            "─" * 60,
            "  ⚠  연구/교육 목적 전용 — 임상 진단이 아닙니다.",
        ]
        return "\n".join(lines)


def _ndarray_to_jpeg_b64(image_np: np.ndarray, max_side: int = 512) -> str:
    img = Image.fromarray(image_np.astype(np.uint8))
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _build_user_prompt(result) -> str:
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

    lines += [
        "",
        "## Your Task",
        "The image above is a brain CT scan with lesion overlay (red = detected lesion area).",
        "Please provide a brief structured response covering:",
        "1. **Observation**: What do you see in the CT image?",
        "2. **Consistency**: Does the image appearance match the pipeline classification?",
        "3. **Uncertainty**: Any areas of concern or ambiguity?",
        "",
        "Keep it concise (4–6 sentences). This is a research tool, not a clinical report.",
    ]
    return "\n".join(lines)


class LLMReporter:
    """LLaMA 3.2 Vision 11B를 Ollama 로컬 서버로 호출하는 CT 판독 리포터."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
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
            names = [m.model for m in models.models]
            return any(self.model.split(":")[0] in n for n in names)
        except Exception:
            return False

    def analyze(
        self,
        result,
        overlay_image: Optional[np.ndarray] = None,
        original_image: Optional[np.ndarray] = None,
    ) -> LLMReport:
        """
        PipelineResult와 이미지를 받아 LLM 판독 리포트를 반환.

        overlay_image: pipeline.result.overlay_image (병변 오버레이 포함)
        original_image: 원본 CT NumPy array (overlay가 없을 때 사용)
        """
        image_np = overlay_image if overlay_image is not None else original_image
        if image_np is None and hasattr(result, "overlay_image"):
            image_np = result.overlay_image

        t0 = time.time()
        try:
            client = self._get_client()
            prompt = _build_user_prompt(result)

            messages = []
            if image_np is not None:
                image_b64 = _ndarray_to_jpeg_b64(image_np)
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
                options={"temperature": 0.3, "num_predict": 512},
            )
            raw = response.message.content.strip()
            elapsed = time.time() - t0

            return LLMReport(
                model=self.model,
                raw_response=raw,
                elapsed_sec=elapsed,
                pipeline_class=result.class_name,
                pipeline_confidence=result.confidence,
            )

        except Exception as e:
            return LLMReport(
                model=self.model,
                raw_response="",
                elapsed_sec=time.time() - t0,
                pipeline_class=getattr(result, "class_name", "unknown"),
                pipeline_confidence=getattr(result, "confidence", 0.0),
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
