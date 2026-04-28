"""
배치 CT 이미지에 LLaMA 3.2 Vision 판독 적용.

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

from inference.pipeline import StrokePipeline
from inference.llm_reporter import LLMReporter
from inference.visualization import save_visualization

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def main(args):
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

    reporter = LLMReporter(model=args.model, host=args.host)
    if not reporter.is_available():
        print(f"Ollama 서버 또는 모델({args.model})을 찾을 수 없습니다.")
        print("  ollama serve && ollama pull", args.model)
        sys.exit(1)

    print(f"LLM 모델: {args.model}\n")

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
    parser = argparse.ArgumentParser(description="배치 CT LLM 판독")
    parser.add_argument("--image_dir", required=True, help="CT 이미지 디렉토리")
    parser.add_argument("--output_dir", default="results/llm", help="결과 저장 경로")
    parser.add_argument("--cls_ckpt", default=None)
    parser.add_argument("--seg_ckpt", default=None)
    parser.add_argument("--model", default="llama3.2-vision:11b", help="Ollama 모델명")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama 서버 주소")
    args = parser.parse_args()
    main(args)
