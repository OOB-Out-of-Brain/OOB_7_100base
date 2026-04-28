import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import cv2

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "DejaVu Sans"


CLASS_COLORS = {
    "normal":      (100, 220, 100),
    "ischemic":    (255, 100, 100),
    "hemorrhagic": (255, 165,  50),
}

_BAR_PALETTE = ["#4caf50", "#ff9800", "#f44336", "#2196f3", "#9c27b0"]


def visualize_result(orig_np: np.ndarray, result, alpha: float = 0.45) -> np.ndarray:
    fig = _build_figure(orig_np, result, alpha)
    overlay_np = _fig_to_numpy(fig)
    plt.close(fig)
    return overlay_np


def save_visualization(orig_np: np.ndarray, result, output_path: str,
                       alpha: float = 0.45, dpi: int = 150):
    fig = _build_figure(orig_np, result, alpha)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _build_figure(orig_np: np.ndarray, result, alpha: float):
    has_mask = result.lesion_mask is not None
    ncols = 3 if has_mask else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), facecolor="black")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08, wspace=0.12)

    color_rgb = CLASS_COLORS.get(result.class_name, (200, 200, 200))
    color_norm = [c / 255 for c in color_rgb]

    # ── Panel 1: Original CT ──────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(orig_np)
    ax.set_title("Original CT", color="white", fontsize=13, pad=8)
    ax.axis("off")

    # ── Panel 2: Classification bar chart ─────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    probs = list(result.class_probs.values())
    labels_en = [k.capitalize() for k in result.class_probs.keys()]
    bar_colors = _BAR_PALETTE[:len(probs)]
    bars = ax.barh(labels_en, probs, color=bar_colors, height=0.5)
    for bar, p in zip(bars, probs):
        ax.text(min(p + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", color="white", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Probability", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title(
        f"Result: {result.class_name.upper()}  ({result.confidence:.1%})",
        color=color_norm, fontsize=13, fontweight="bold", pad=8,
    )

    # ── Panel 3: Lesion overlay (허혈성 / 출혈성) ─────────────────────────────
    if has_mask:
        ax = axes[2]
        overlay = _make_overlay(orig_np, result.lesion_mask, color_rgb, alpha)
        ax.imshow(overlay)
        _draw_lesion_contour(ax, result.lesion_mask, result.class_name)
        ax.set_title(
            f"Lesion Overlay  ({result.lesion_area_pct:.1f}% of image)",
            color="white", fontsize=13, pad=8,
        )
        label = f"{result.class_name.capitalize()} Lesion"
        patch = mpatches.Patch(color=color_norm, label=label)
        ax.legend(handles=[patch], loc="lower right",
                  facecolor="#222", edgecolor="#555",
                  labelcolor="white", fontsize=10)
        ax.axis("off")

    fig.suptitle("Brain CT Stroke Analysis",
                 color="white", fontsize=15, fontweight="bold", y=0.97)
    return fig


def _make_overlay(image: np.ndarray, mask: np.ndarray,
                  color_rgb: tuple, alpha: float) -> np.ndarray:
    overlay = image.copy().astype(np.float32)
    lesion_pixels = mask > 0.5
    color_arr = np.array(color_rgb, dtype=np.float32)
    overlay[lesion_pixels] = (1 - alpha) * overlay[lesion_pixels] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


CONTOUR_COLORS = {
    "ischemic":    "yellow",
    "hemorrhagic": "cyan",
}


def _draw_lesion_contour(ax, mask: np.ndarray, class_name: str = "ischemic"):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_color = CONTOUR_COLORS.get(class_name, "yellow")
    for cnt in contours:
        pts = cnt[:, 0, :]
        ax.plot(
            np.append(pts[:, 0], pts[0, 0]),
            np.append(pts[:, 1], pts[0, 1]),
            color=contour_color, linewidth=1.2, alpha=0.85,
        )


def _fig_to_numpy(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()
