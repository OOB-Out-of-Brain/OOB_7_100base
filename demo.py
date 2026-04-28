"""
뇌 CT 출혈 분석 데모.

사용법:
    python demo.py --image path/to/ct.png
    python demo.py --image path/to/ct.png --output results/result.png
    python demo.py --image path/to/ct.png --llm
    python demo.py --image path/to/ct.png --llm --llm-model llama3.2-vision:11b
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from inference.pipeline import StrokePipeline
from inference.visualization import save_visualization


def main(args):
    cls_ckpt = args.cls_ckpt or "checkpoints/classifier/best_classifier.pth"
    seg_ckpt = args.seg_ckpt or "checkpoints/segmentor/best_segmentor.pth"

    for ckpt in [cls_ckpt, seg_ckpt]:
        if not Path(ckpt).exists():
            print(f"체크포인트 없음: {ckpt}")
            print("  먼저 학습을 완료하세요:")
            print("  python training/train_classifier.py")
            print("  python training/train_segmentor.py")
            sys.exit(1)

    if not Path(args.image).exists():
        print(f"이미지 없음: {args.image}")
        sys.exit(1)

    print(f"\n이미지: {args.image}")
    print("모델 로딩 중...")

    pipeline = StrokePipeline(
        classifier_ckpt=cls_ckpt,
        segmentor_ckpt=seg_ckpt,
    )

    print("추론 실행 중...\n")
    result = pipeline.run(args.image)
    orig_np = np.array(Image.open(args.image).convert("RGB"))

    print("=" * 60)
    print(result)
    print("=" * 60)

    output_path = args.output or f"results/{Path(args.image).stem}_result.png"
    save_visualization(orig_np, result, output_path)
    print(f"\n시각화 저장: {output_path}")

    if args.llm:
        print("\nLLM 판독 실행 중...")
        from inference.llm_reporter import LLMReporter
        reporter = LLMReporter(
            model=args.llm_model,
            host=args.llm_host,
        )
        if not reporter.is_available():
            print(f"  Ollama 서버 또는 모델({args.llm_model})을 찾을 수 없습니다.")
            print("  설정 방법:")
            print("    brew install ollama")
            print("    ollama serve")
            print(f"    ollama pull {args.llm_model}")
        else:
            report = reporter.analyze(result, overlay_image=result.overlay_image, original_image=orig_np)
            print(report)

            if args.llm_save:
                report_path = Path(output_path).with_suffix(".llm.txt")
                report_path.write_text(str(report), encoding="utf-8")
                print(f"LLM 리포트 저장: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뇌 CT 출혈 분석 데모")
    parser.add_argument("--image", required=True, help="CT 이미지 경로")
    parser.add_argument("--output", default=None, help="시각화 PNG 저장 경로")
    parser.add_argument("--cls_ckpt", default=None, help="분류 모델 체크포인트")
    parser.add_argument("--seg_ckpt", default=None, help="분할 모델 체크포인트")

    llm = parser.add_argument_group("LLM 옵션")
    llm.add_argument("--llm", action="store_true", help="LLM 판독 활성화 (LLaMA 3.2 Vision)")
    llm.add_argument("--llm-model", default="llama3.2-vision:11b", help="Ollama 모델명")
    llm.add_argument("--llm-host", default="http://localhost:11434", help="Ollama 서버 주소")
    llm.add_argument("--llm-save", action="store_true", help="LLM 리포트를 .llm.txt로 저장")

    args = parser.parse_args()
    main(args)
