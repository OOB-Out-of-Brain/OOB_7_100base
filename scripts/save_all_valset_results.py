"""Val set 2089장 전체 결과를 4개 폴더로 분류 저장.

출력 (results/valset_all/):
  correct_normal/       실제 정상 & 정상 예측 (맞춤, TN)
  correct_hemorrhagic/  실제 출혈 & 출혈 예측 (맞춤, TP)
  false_positives/      실제 정상 but 출혈 예측 (오탐, FP)
  false_negatives/      실제 출혈 but 정상 예측 (누락, FN)

파일명: {idx:04d}_{source}_conf={:.2f}_lesion={:.1f}%.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from data.combined_dataset import build_combined_dataloaders
from data.ct_hemorrhage_io import load_ct_image, ref_name
from inference.pipeline import StrokePipeline


OUT_DIR = Path("./results/valset_all")
SUBDIRS = {
    ("tn",): OUT_DIR / "correct_normal",
    ("tp",): OUT_DIR / "correct_hemorrhagic",
    ("fp",): OUT_DIR / "false_positives",
    ("fn",): OUT_DIR / "false_negatives",
}
for p in SUBDIRS.values():
    p.mkdir(parents=True, exist_ok=True)

_CFG = yaml.safe_load(open(Path(__file__).parent.parent / "config.yaml"))


def save_panel(img, overlay, out_path: Path, title: str, title_color: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay if overlay is not None else img)
    axes[1].set_title(title, color=title_color, fontsize=10)
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close()


def main():
    print("Val set 로딩...")
    _, val_loader, _ = build_combined_dataloaders(
        ct_root=_CFG["data"]["ct_hemorrhage_path"],
        tekno21_cache=_CFG["data"]["tekno21_cache"],
        image_size=_CFG["classifier"]["image_size"], batch_size=1, num_workers=0,
    )
    val_ds = val_loader.dataset
    samples = val_ds.samples
    hf = val_ds.hf

    pipe = StrokePipeline(
        classifier_ckpt="./checkpoints/classifier/best_classifier.pth",
        segmentor_ckpt="./checkpoints/segmentor/best_segmentor.pth",
    )
    print(f"총 {len(samples)} 샘플 추론 + 저장\n")

    counts = {"tn": 0, "tp": 0, "fp": 0, "fn": 0}

    for i, (source, ref, gt) in enumerate(samples):
        if source in ("ct", "bhsd"):
            img = load_ct_image(ref)
            short = ref_name(ref)[:20]
        else:
            item = hf[ref]
            im = item["image"]
            if not isinstance(im, Image.Image):
                im = Image.fromarray(im)
            img = np.array(im.convert("RGB"))
            short = f"tk{ref}"

        r = pipe.run(img)
        pred = 1 if r.class_name == "hemorrhagic" else 0

        if gt == 0 and pred == 0:
            bucket = "tn"; label = "정상 → 정상 (맞춤)"; color = "green"
        elif gt == 1 and pred == 1:
            bucket = "tp"; label = "출혈 → 출혈 (맞춤)"; color = "green"
        elif gt == 0 and pred == 1:
            bucket = "fp"; label = "⚠️ 오탐 (정상 → 출혈)"; color = "red"
        else:
            bucket = "fn"; label = "🚨 누락 (출혈 → 정상)"; color = "darkred"

        counts[bucket] += 1

        out_name = f"{i:04d}_{source}_{short}_conf{r.confidence:.2f}_les{r.lesion_area_pct:.1f}.png"
        out_path = SUBDIRS[(bucket,)] / out_name

        title = f"{label}\nconf={r.confidence:.1%}  lesion={r.lesion_area_pct:.1f}%"
        save_panel(img, r.overlay_image, out_path, title, color)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}  TN={counts['tn']} TP={counts['tp']} "
                  f"FP={counts['fp']} FN={counts['fn']}")

    print(f"\n=== 최종 분포 ===")
    print(f"  correct_normal      (TN, 정상 맞춤) : {counts['tn']}장")
    print(f"  correct_hemorrhagic (TP, 출혈 맞춤) : {counts['tp']}장")
    print(f"  false_positives     (FP, 오탐)     : {counts['fp']}장")
    print(f"  false_negatives     (FN, 누락)     : {counts['fn']}장")
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
