"""
학습 진행 모니터링 그래프 (세그멘테이션)
실행: python scripts/monitor_training.py
"""
import re
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

SEG_LOG = "/private/tmp/claude-501/-Users-bari-brain-stroke-ai/6b56275a-b1ae-4ede-8e23-7a21d5428975/tasks/buqzn31j6.output"
OUTPUT  = "results/training_monitor.png"
TOTAL_EPOCHS = 80


def parse_log(log_path):
    epochs, train_loss, train_dice = [], [], []
    val_loss, val_dice, val_iou = [], [], []

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+Train loss=([\d.]+)\s+dice=([\d.]+)\s+\|\s+Val loss=([\d.]+)\s+dice=([\d.]+)\s+iou=([\d.]+)"
    )
    try:
        text = Path(log_path).read_text(errors="ignore")
        for m in pattern.finditer(text):
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            train_dice.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))
            val_dice.append(float(m.group(5)))
            val_iou.append(float(m.group(6)))
    except FileNotFoundError:
        pass
    return epochs, train_loss, train_dice, val_loss, val_dice, val_iou


def draw(epochs, train_loss, train_dice, val_loss, val_dice, val_iou):
    fig = plt.figure(figsize=(14, 8), facecolor="#111")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    current = epochs[-1] if epochs else 0
    pct = current / TOTAL_EPOCHS * 100

    fig.suptitle(
        f"Segmentation Training  —  Epoch {current}/{TOTAL_EPOCHS}  ({pct:.0f}%)",
        color="white", fontsize=15, fontweight="bold", y=0.97
    )

    # ── 진행률 바 ─────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor("#1a1a1a")
    ax0.barh([0], [pct], color="#4caf50", height=0.5)
    ax0.barh([0], [100 - pct], left=[pct], color="#333", height=0.5)
    ax0.set_xlim(0, 100)
    ax0.set_yticks([])
    ax0.set_xlabel("Progress (%)", color="white")
    ax0.tick_params(colors="white")
    ax0.text(min(pct + 1, 95), 0, f"{pct:.0f}%  (epoch {current}/{TOTAL_EPOCHS})",
             va="center", color="white", fontsize=12, fontweight="bold")
    for sp in ax0.spines.values():
        sp.set_edgecolor("#444")

    if not epochs:
        plt.savefig(OUTPUT, dpi=120, bbox_inches="tight", facecolor="#111")
        plt.close(fig)
        return

    best_dice = max(val_dice)
    best_epoch = epochs[val_dice.index(best_dice)]

    # ── Dice Score ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor("#1a1a1a")
    ax1.plot(epochs, train_dice, color="#4caf50", linewidth=1.8, label="Train Dice")
    ax1.plot(epochs, val_dice,   color="#ff9800", linewidth=1.8, label="Val Dice")
    ax1.axvline(best_epoch, color="#ff9800", linestyle="--", alpha=0.5, linewidth=1)
    ax1.annotate(f"Best {best_dice:.3f}", xy=(best_epoch, best_dice),
                 xytext=(best_epoch + 1, best_dice - 0.03),
                 color="#ff9800", fontsize=9)
    ax1.set_title("Dice Score", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="white"); ax1.set_ylabel("Dice", color="white")
    ax1.tick_params(colors="white"); ax1.legend(facecolor="#222", labelcolor="white", fontsize=9)
    ax1.set_ylim(0, 1)
    for sp in ax1.spines.values(): sp.set_edgecolor("#444")

    # ── Loss ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor("#1a1a1a")
    ax2.plot(epochs, train_loss, color="#2196f3", linewidth=1.8, label="Train Loss")
    ax2.plot(epochs, val_loss,   color="#f44336", linewidth=1.8, label="Val Loss")
    ax2.set_title("Loss", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="white"); ax2.set_ylabel("Loss", color="white")
    ax2.tick_params(colors="white"); ax2.legend(facecolor="#222", labelcolor="white", fontsize=9)
    for sp in ax2.spines.values(): sp.set_edgecolor("#444")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=120, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
    print(f"저장: {OUTPUT}  |  Epoch {current}/{TOTAL_EPOCHS}  |  Best Dice: {best_dice:.4f} (ep{best_epoch})")


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"모니터링 시작 (갱신 주기: {interval}초) — Ctrl+C로 종료")
    while True:
        data = parse_log(SEG_LOG)
        draw(*data)
        time.sleep(interval)
