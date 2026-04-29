"""
배치 CT 이미지에 LLaMA 3.2 Vision triage 보조 리포트 적용.

사용법:
    python scripts/run_llm_report.py --image_dir test_samples/
    python scripts/run_llm_report.py --image_dir test_samples/ --output_dir results/llm/
    python scripts/run_llm_report.py --image_dir test_samples/ --model llama3.2-vision:11b
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import yaml

from inference.pipeline import StrokePipeline
from inference.llm_reporter import LLMReporter
from inference.visualization import save_visualization

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _load_config() -> dict:
    config_path = Path("config.yaml")
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return yaml.safe_load(f) or {}


def _normalize_llm_mode(mode: str) -> str:
    return mode.replace("-", "_")


def _llm_arg(args, name: str, mode_settings: dict):
    value = getattr(args, name)
    return mode_settings.get(name) if value is None else value


def main(args):
    cfg = _load_config()
    llm_cfg = cfg.get("llm", {})
    mode = _normalize_llm_mode(args.mode or llm_cfg.get("mode", "fast"))
    mode_settings = dict(llm_cfg.get("modes", {}).get(mode, {}))
    include_image = False if args.no_image else mode_settings.get("include_image")

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"이미지 없음: {image_dir}")
        sys.exit(1)

    print(f"\n이미지 {len(images)}개 발견: {image_dir}")

    cls_ckpt = args.cls_ckpt or "checkpoints/classifier/best_classifier.pth"
    seg_ckpt = args.seg_ckpt or "checkpoints/segmentor/best_segmentor.pth"

    for ckpt in [cls_ckpt, seg_ckpt]:
        if not Path(ckpt).exists():
            print(f"체크포인트 없음: {ckpt}")
            sys.exit(1)

    pipeline = StrokePipeline(classifier_ckpt=cls_ckpt, segmentor_ckpt=seg_ckpt)

    reporter = LLMReporter(
        model=args.model or llm_cfg.get("model", "llama3.2-vision:11b"),
        host=args.host or llm_cfg.get("host", "http://localhost:11434"),
        mode=mode,
        timeout=_llm_arg(args, "timeout", mode_settings),
        max_side=_llm_arg(args, "max_side", mode_settings),
        jpeg_quality=_llm_arg(args, "jpeg_quality", mode_settings),
        num_predict=_llm_arg(args, "num_predict", mode_settings),
        temperature=_llm_arg(args, "temperature", mode_settings),
        include_image=include_image,
    )
    if not reporter.is_available():
        print(f"Ollama 서버 또는 모델({reporter.model})을 찾을 수 없습니다.")
        print("  ollama serve && ollama pull", reporter.model)
        sys.exit(1)

    print(f"LLM 모델: {reporter.model}  mode={mode}\n")

    summary_rows = []
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name}")

        orig_np = np.array(Image.open(img_path).convert("RGB"))
        result = pipeline.run(str(img_path))

        vis_path = output_dir / f"{img_path.stem}_vis.png"
        save_visualization(orig_np, result, str(vis_path))

        print(f"  파이프라인: {result.class_name.upper()} ({result.confidence:.1%})", end=" ")
        report = reporter.analyze(result, overlay_image=result.overlay_image, original_image=orig_np)
        print(f"→ LLM {report.elapsed_sec:.1f}s")

        report_path = output_dir / f"{img_path.stem}.llm.txt"
        report_path.write_text(str(report), encoding="utf-8")

        summary_rows.append({
            "image": img_path.name,
            "pipeline_class": result.class_name,
            "pipeline_confidence": f"{result.confidence:.4f}",
            "lesion_area_pct": f"{result.lesion_area_pct:.2f}",
            "decision_source": result.decision_source,
            "llm_mode": report.mode,
            "llm_image_used": str(report.image_used),
            "llm_elapsed_sec": f"{report.elapsed_sec:.1f}",
            "llm_error": report.error or "",
            "llm_response_preview": report.raw_response[:120].replace("\n", " "),
        })

    csv_path = output_dir / "llm_summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n완료: {len(images)}개 처리")
    print(f"  결과 디렉토리: {output_dir}")
    print(f"  요약 CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="배치 CT LLM triage 보조 리포트")
    parser.add_argument("--image_dir", required=True, help="CT 이미지 디렉토리")
    parser.add_argument("--output_dir", default="results/llm", help="결과 저장 경로")
    parser.add_argument("--cls_ckpt", default=None)
    parser.add_argument("--seg_ckpt", default=None)
    parser.add_argument("--model", default=None, help="Ollama 모델명 (기본: config.yaml)")
    parser.add_argument("--host", default=None, help="Ollama 서버 주소 (기본: config.yaml)")
    parser.add_argument("--mode", default=None, help="fast, balanced, detailed, text-only")
    parser.add_argument("--no-image", action="store_true", help="이미지 없이 pipeline 수치만 LLM에 전달")
    parser.add_argument("--timeout", type=int, default=None, help="LLM timeout 초")
    parser.add_argument("--max-side", type=int, default=None, help="LLM 입력 이미지 최대 변 길이")
    parser.add_argument("--jpeg-quality", type=int, default=None, help="LLM 입력 JPEG 품질")
    parser.add_argument("--num-predict", type=int, default=None, help="LLM 최대 출력 토큰")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature")
    args = parser.parse_args()
    main(args)
