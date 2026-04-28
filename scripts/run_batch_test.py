"""폴더 내 모든 CT 이미지를 배치 추론 + 결과 저장.

사용법:
    python scripts/run_batch_test.py --input-dir /path/to/images --output-dir results/
    python scripts/run_batch_test.py --input-dir data/test/ --output-dir results/test_run/
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from inference.pipeline import StrokePipeline


def main(args):
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("모델 로딩...")
    pipe = StrokePipeline(
        classifier_ckpt=args.cls_ckpt,
        segmentor_ckpt=args.seg_ckpt,
    )

    images = sorted([
        p for p in in_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        and not p.name.startswith("result_")
    ])
    if not images:
        print(f"❌ 이미지 없음: {in_dir}")
        return
    print(f"이미지 {len(images)}개 추론 → {out_dir}\n")

    # 헤더
    print(f"{'파일':<30} {'결과':<14} {'신뢰도':>8} {'병변':>14}")
    print("-" * 70)

    summary = {"normal": 0, "hemorrhagic": 0}
    for img_path in images:
        r = pipe.run(img_path)
        summary[r.class_name] = summary.get(r.class_name, 0) + 1

        print(f"{img_path.name:<30} {r.class_name:<14} {r.confidence:>7.1%} "
              f"{r.lesion_area_px:>7}px ({r.lesion_area_pct:.1f}%)")

        # 결과 이미지 저장
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        orig = np.array(Image.open(img_path).convert("RGB"))
        axes[0].imshow(orig); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(r.overlay_image if r.overlay_image is not None else orig)
        color = "green" if r.class_name == "normal" else "red"
        axes[1].set_title(f"{r.class_name.upper()} ({r.confidence:.1%})", color=color)
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{img_path.stem}_result.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("\n" + "=" * 70)
    print(f"  총 {len(images)}개 | normal {summary.get('normal', 0)}개 | "
          f"hemorrhagic {summary.get('hemorrhagic', 0)}개")
    print(f"  저장: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="테스트 이미지 폴더")
    p.add_argument("--output-dir", required=True, help="결과 저장 폴더")
    p.add_argument("--cls-ckpt", default="./checkpoints/classifier/best_classifier.pth")
    p.add_argument("--seg-ckpt", default="./checkpoints/segmentor/best_segmentor.pth")
    args = p.parse_args()
    main(args)
