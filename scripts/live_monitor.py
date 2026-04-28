"""
실시간 학습 모니터링

실행 예:
  python scripts/live_monitor.py --log logs/segmentor_3class_train.log
  python scripts/live_monitor.py --log logs/classifier_3class_train.log
"""

import argparse
import re
import time
from pathlib import Path

from rich import box
from rich.console import Group
from rich.live import Live
from rich.table import Table
from rich.text import Text


DEFAULT_LOG = "logs/segmentor_3class_train.log"
DEFAULT_TOTAL_EPOCHS = 80

PHASE_PATTERN = re.compile(
    r"\[Phase\s+(\d+)\]\s+(.+?)\s+—\s+(\d+)\s+epochs",
    re.IGNORECASE,
)

PROGRESS_PATTERN = re.compile(
    r"\[Progress\]\s+phase=([a-z_]+)\s+split=(train|eval)\s+"
    r"epoch=(\d+)/(\d+)\s+batch=(\d+)/(\d+)\s+pct=([-\d.]+)\s+elapsed=([0-9hms ]+)",
    re.IGNORECASE,
)

ETA_PATTERN = re.compile(
    r"->\s+epoch_time=([^|]+)\|\s+recent_avg=([^|]+)\|\s+remaining=([^|]+)\|\s+finish~(.+)",
    re.IGNORECASE,
)


SEG3_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
    r"train loss=([-\d.]+)\s+dice_lesion_mean=([-\d.]+)\s+\|\s+"
    r"val loss=([-\d.]+)\s+dice_lesion_mean=([-\d.]+)\s+\|\s+"
    r"dice_isc=([-\d.]+)\s+dice_hem=([-\d.]+)\s+"
    r"iou_isc=([-\d.]+)\s+iou_hem=([-\d.]+)\s+"
    r"empty_fp_isc=([-\d.]+)\s+empty_fp_hem=([-\d.]+)",
    re.IGNORECASE,
)

SEG2_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+train loss=([-\d.]+)\s+dice_pos=([-\d.]+)"
    r"\s+\|\s+val loss=([-\d.]+)\s+dice_positive=([-\d.]+)"
    r"\s+\|\s+dice_all=([-\d.]+)\s+dice_pos=([-\d.]+)"
    r"\s+iou_all=([-\d.]+)\s+iou_pos=([-\d.]+)\s+empty_fp=([-\d.]+)",
    re.IGNORECASE,
)

CLS3_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
    r"train loss=([-\d.]+)\s+acc=([-\d.]+)\s+\|\s+"
    r"val loss=([-\d.]+)\s+score=([-\d.]+)\s+([a-z0-9_]+)=([-\d.]+)\s+\|\s+"
    r"acc=([-\d.]+)\s+macro_f1=([-\d.]+)\s+"
    r"isc_rec=([-\d.]+)\s+hem_rec=([-\d.]+)\s+lesion_rec=([-\d.]+)",
    re.IGNORECASE,
)

CLS2_PATTERN = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+train loss=([-\d.]+)\s+acc=([-\d.]+)\s+\|\s+"
    r"val loss=([-\d.]+)\s+score=([-\d.]+)\s+\S+\s+\|\s+"
    r"acc=([-\d.]+)\s+macro_f1=([-\d.]+)\s+hem_rec=([-\d.]+)\s+hem_f1=([-\d.]+)\s+spec=([-\d.]+)",
    re.IGNORECASE,
)


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _iter_log_lines(text: str):
    for raw_line in text.splitlines():
        for chunk in raw_line.split("\r"):
            line = chunk.strip()
            if line:
                yield line


def _valid_metric_row(row: dict) -> bool:
    numeric_keys = [
        "tl", "train_main", "vl", "val_main",
        "dice_isc", "dice_hem", "iou_isc", "iou_hem",
        "empty_fp_isc", "empty_fp_hem",
        "val_acc", "macro_f1", "isc_rec", "hem_rec", "lesion_rec",
    ]
    for key in numeric_keys:
        value = row.get(key)
        if value is None:
            continue
        if value != value:  # NaN
            return False
    if row["tl"] < 0 or row["vl"] < 0:
        return False
    for key in ("train_main", "val_main", "dice_isc", "dice_hem", "iou_isc", "iou_hem"):
        value = row.get(key)
        if value is not None and not (0.0 <= value <= 1.0):
            return False
    for key in ("val_acc", "macro_f1", "isc_rec", "hem_rec", "lesion_rec"):
        value = row.get(key)
        if value is not None and not (0.0 <= value <= 1.0):
            return False
    for key in ("empty_fp_isc", "empty_fp_hem"):
        value = row.get(key)
        if value is not None and not (0.0 <= value <= 1.0):
            return False
    return True


def parse_monitor_state(log_path: str):
    rows = []
    state = {
        "rows": [],
        "phase": None,
        "progress": None,
        "eta": None,
    }
    try:
        text = Path(log_path).read_text(errors="ignore")
    except FileNotFoundError:
        return state

    for line in _iter_log_lines(text):
        phase_match = PHASE_PATTERN.search(line)
        if phase_match:
            state["phase"] = {
                "index": int(phase_match.group(1)),
                "label": phase_match.group(2).strip(),
                "epochs": int(phase_match.group(3)),
            }
            continue

        progress_match = PROGRESS_PATTERN.search(line)
        if progress_match:
            state["progress"] = {
                "phase": progress_match.group(1).strip(),
                "split": progress_match.group(2).strip(),
                "epoch": int(progress_match.group(3)),
                "total_epochs": int(progress_match.group(4)),
                "batch": int(progress_match.group(5)),
                "total_batches": int(progress_match.group(6)),
                "pct": _safe_float(progress_match.group(7)),
                "elapsed": progress_match.group(8).strip(),
            }
            continue

        eta_match = ETA_PATTERN.search(line)
        if eta_match:
            state["eta"] = {
                "epoch_time": eta_match.group(1).strip(),
                "recent_avg": eta_match.group(2).strip(),
                "remaining": eta_match.group(3).strip(),
                "finish_at": eta_match.group(4).strip(),
            }
            continue

        mc2 = CLS2_PATTERN.search(line)
        if mc2:
            row = {
                "kind": "cls2",
                "epoch": int(mc2.group(1)),
                "total": int(mc2.group(2)),
                "tl": _safe_float(mc2.group(3)),
                "train_main": _safe_float(mc2.group(4)),
                "vl": _safe_float(mc2.group(5)),
                "val_main": _safe_float(mc2.group(6)),
                "val_acc": _safe_float(mc2.group(7)),
                "macro_f1": _safe_float(mc2.group(8)),
                "hem_rec": _safe_float(mc2.group(9)),
                "hem_f1": _safe_float(mc2.group(10)),
                "spec": _safe_float(mc2.group(11)),
            }
            if _valid_metric_row(row):
                rows.append(row)
            continue

        mc = CLS3_PATTERN.search(line)
        if mc:
            row = {
                "kind": "cls3",
                "epoch": int(mc.group(1)),
                "total": int(mc.group(2)),
                "tl": _safe_float(mc.group(3)),
                "train_main": _safe_float(mc.group(4)),
                "vl": _safe_float(mc.group(5)),
                "val_main": _safe_float(mc.group(6)),
                "selection_metric": mc.group(7),
                "selection_metric_value": _safe_float(mc.group(8)),
                "val_acc": _safe_float(mc.group(9)),
                "macro_f1": _safe_float(mc.group(10)),
                "isc_rec": _safe_float(mc.group(11)),
                "hem_rec": _safe_float(mc.group(12)),
                "lesion_rec": _safe_float(mc.group(13)),
            }
            if _valid_metric_row(row):
                rows.append(row)
            continue

        m3 = SEG3_PATTERN.search(line)
        if m3:
            row = {
                "kind": "seg3",
                "epoch": int(m3.group(1)),
                "total": int(m3.group(2)),
                "tl": _safe_float(m3.group(3)),
                "train_main": _safe_float(m3.group(4)),
                "vl": _safe_float(m3.group(5)),
                "val_main": _safe_float(m3.group(6)),
                "dice_isc": _safe_float(m3.group(7)),
                "dice_hem": _safe_float(m3.group(8)),
                "iou_isc": _safe_float(m3.group(9)),
                "iou_hem": _safe_float(m3.group(10)),
                "empty_fp_isc": _safe_float(m3.group(11)),
                "empty_fp_hem": _safe_float(m3.group(12)),
            }
            if _valid_metric_row(row):
                rows.append(row)
            continue

        m2 = SEG2_PATTERN.search(line)
        if m2:
            row = {
                "kind": "seg2",
                "epoch": int(m2.group(1)),
                "total": int(m2.group(2)),
                "tl": _safe_float(m2.group(3)),
                "train_main": _safe_float(m2.group(4)),
                "vl": _safe_float(m2.group(5)),
                "val_main": _safe_float(m2.group(6)),
                "dice_isc": _safe_float(m2.group(7)),
                "dice_hem": _safe_float(m2.group(8)),
                "iou_isc": _safe_float(m2.group(9)),
                "iou_hem": _safe_float(m2.group(10)),
                "empty_fp_isc": 0.0,
                "empty_fp_hem": _safe_float(m2.group(11)),
            }
            if _valid_metric_row(row):
                rows.append(row)

    deduped = {}
    for row in rows:
        deduped[row["epoch"]] = row
    state["rows"] = [deduped[epoch] for epoch in sorted(deduped)]
    return state


def parse_seg(log_path: str):
    return parse_monitor_state(log_path)["rows"]


def render(state, total_epochs=DEFAULT_TOTAL_EPOCHS):
    if isinstance(state, list):
        rows = state
        phase_state = None
        progress_state = None
        eta_state = None
    else:
        rows = state.get("rows", [])
        phase_state = state.get("phase")
        progress_state = state.get("progress")
        eta_state = state.get("eta")

    if rows:
        current = rows[-1]["epoch"]
        total_epochs = rows[-1].get("total", total_epochs)
        pct = current / max(total_epochs, 1) * 100
    elif progress_state:
        current = progress_state["epoch"]
        total_epochs = progress_state["total_epochs"]
        completed = max(0.0, current - 1 + progress_state["pct"] / 100.0)
        pct = completed / max(total_epochs, 1) * 100
    else:
        current = 0
        pct = 0.0
    best_main = max((r["val_main"] for r in rows), default=0.0)
    best_ep = next((r["epoch"] for r in rows if r["val_main"] == best_main), 0)
    bar_len = 40
    filled = int(bar_len * pct / 100)
    kind = rows[-1]["kind"] if rows else "seg3"
    if kind in ("cls2", "cls3"):
        metric_label = "Val Score"
    else:
        metric_label = "Val Dice Lesion" if kind == "seg3" else "Val Dice Positive"

    prog = Text()
    prog.append("\n  Epoch  ", style="bold")
    prog.append(f"{current:3d} / {total_epochs}", style="cyan bold")
    prog.append("   [", style="dim")
    prog.append("█" * filled, style="green bold")
    prog.append("░" * (bar_len - filled), style="dim")
    prog.append("]", style="dim")
    prog.append(f"  {pct:.0f}%\n", style="bold white")
    prog.append(f"  Best {metric_label}: ", style="")
    prog.append(f"{best_main:.4f}", style="bold green")
    prog.append(f"  (epoch {best_ep})\n", style="dim")

    phase_label = phase_state["label"] if phase_state else "-"
    if progress_state:
        phase_text = progress_state["phase"]
        if phase_state and progress_state["phase"] not in phase_label.lower():
            phase_text = f"{progress_state['phase']} ({phase_label})"
        split_text = progress_state["split"]
        prog.append("  Status: ", style="")
        prog.append(phase_text, style="bold cyan")
        prog.append(" / ", style="dim")
        prog.append(split_text, style="bold yellow")
        prog.append(" / ", style="dim")
        prog.append(
            f"batch {progress_state['batch']}/{progress_state['total_batches']} "
            f"({progress_state['pct']:.0f}%)",
            style="bold white",
        )
        prog.append(f" / elapsed {progress_state['elapsed']}\n", style="dim")
    else:
        prog.append("  Status: ", style="")
        prog.append(phase_label, style="bold cyan")
        prog.append("\n", style="")

    if eta_state:
        prog.append("  ETA: ", style="")
        prog.append(eta_state["remaining"], style="bold green")
        prog.append(" remaining / finish~", style="dim")
        prog.append(f"{eta_state['finish_at']}\n", style="bold white")
    elif progress_state:
        prog.append("  ETA: ", style="")
        prog.append("첫 epoch 완료 후 계산됩니다\n", style="dim")

    table = Table(box=box.SIMPLE_HEAVY, border_style="blue", header_style="bold cyan", show_edge=True)

    if kind == "cls2":
        table.add_column("Epoch", justify="right", style="cyan", width=7)
        table.add_column("Train Loss", justify="right", width=11)
        table.add_column("Train Acc", justify="right", width=10)
        table.add_column("Val Loss", justify="right", width=10)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Val Acc", justify="right", width=8)
        table.add_column("Macro F1", justify="right", width=9)
        table.add_column("HEM Rec", justify="right", width=9)
        table.add_column("Spec", justify="right", width=8)
        table.add_column(" ", justify="center", width=3)

        for row in rows[-20:]:
            is_best = row["epoch"] == best_ep
            d_color = "green" if row["val_main"] >= 0.7 else "yellow" if row["val_main"] >= 0.5 else "red"
            table.add_row(
                f"[bold cyan]{row['epoch']}[/bold cyan]" if is_best else str(row["epoch"]),
                f"{row['tl']:.4f}",
                f"{row['train_main']:.4f}",
                f"{row['vl']:.4f}",
                f"[{d_color} bold]{row['val_main']:.4f}[/{d_color} bold]",
                f"{row['val_acc']:.4f}",
                f"{row['macro_f1']:.4f}",
                f"{row['hem_rec']:.4f}",
                f"{row['spec']:.4f}",
                "[green]★[/green]" if is_best else "",
            )
    elif kind == "cls3":
        table.add_column("Epoch", justify="right", style="cyan", width=7)
        table.add_column("Train Loss", justify="right", width=11)
        table.add_column("Train Acc", justify="right", width=10)
        table.add_column("Val Loss", justify="right", width=10)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Val Acc", justify="right", width=8)
        table.add_column("Macro F1", justify="right", width=9)
        table.add_column("ISC Rec", justify="right", width=8)
        table.add_column("HEM Rec", justify="right", width=8)
        table.add_column(" ", justify="center", width=3)

        for row in rows[-20:]:
            is_best = row["epoch"] == best_ep
            d_color = "green" if row["val_main"] >= 0.5 else "yellow" if row["val_main"] >= 0.3 else "red"
            table.add_row(
                f"[bold cyan]{row['epoch']}[/bold cyan]" if is_best else str(row["epoch"]),
                f"{row['tl']:.4f}",
                f"{row['train_main']:.4f}",
                f"{row['vl']:.4f}",
                f"[{d_color} bold]{row['val_main']:.4f}[/{d_color} bold]",
                f"{row['val_acc']:.4f}",
                f"{row['macro_f1']:.4f}",
                f"{row['isc_rec']:.4f}",
                f"{row['hem_rec']:.4f}",
                "[green]★[/green]" if is_best else "",
            )
    else:
        table.add_column("Epoch", justify="right", style="cyan", width=7)
        table.add_column("Train Loss", justify="right", width=11)
        table.add_column("Train Main", justify="right", width=10)
        table.add_column("Val Loss", justify="right", width=10)
        table.add_column("Val Main", justify="right", width=9)
        table.add_column("Dice ISC", justify="right", width=9)
        table.add_column("Dice HEM", justify="right", width=9)
        table.add_column("IoU HEM", justify="right", width=8)
        table.add_column("FP HEM", justify="right", width=8)
        table.add_column(" ", justify="center", width=3)

        for row in rows[-20:]:
            is_best = row["epoch"] == best_ep
            d_color = "green" if row["val_main"] >= 0.6 else "yellow" if row["val_main"] >= 0.4 else "red"
            table.add_row(
                f"[bold cyan]{row['epoch']}[/bold cyan]" if is_best else str(row["epoch"]),
                f"{row['tl']:.4f}",
                f"{row['train_main']:.4f}",
                f"{row['vl']:.4f}",
                f"[{d_color} bold]{row['val_main']:.4f}[/{d_color} bold]",
                f"{row['dice_isc']:.4f}",
                f"{row['dice_hem']:.4f}",
                f"{row['iou_hem']:.4f}",
                f"{row['empty_fp_hem']:.4f}",
                "[green]★[/green]" if is_best else "",
            )

    return Group(prog, table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=DEFAULT_LOG, help="학습 로그 파일")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL_EPOCHS, help="전체 epoch 수")
    parser.add_argument("--interval", type=float, default=3.0, help="갱신 주기(초)")
    parser.add_argument("--no-alt-screen", action="store_true", help="일반 터미널 출력 모드로 표시")
    args = parser.parse_args()

    print(f"  실시간 모니터링: {args.log}  (Ctrl+C 종료)\n")
    if not Path(args.log).exists():
        print("  로그 파일이 아직 없습니다. 학습 출력을 tee로 저장하면 자동으로 표시됩니다.")
        print(f"  예: python training/train_segmentor_3class.py 2>&1 | tee {args.log}")
        print(f"      python training/train_classifier_3class.py 2>&1 | tee {args.log}\n")
    with Live(
        render([], args.total),
        refresh_per_second=1,
        screen=not args.no_alt_screen,
        vertical_overflow="crop",
    ) as live:
        while True:
            live.update(render(parse_monitor_state(args.log), args.total))
            time.sleep(args.interval)
