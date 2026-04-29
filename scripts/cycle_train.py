"""
2-class 사이클 훈련 + Rich 실시간 모니터링
분류기(train_classifier.py) → 분할기(train_segmentor.py) 순서로 실행합니다.

실행:
  python scripts/cycle_train.py                  # 체크포인트 있으면 이어서/처음 선택
  python scripts/cycle_train.py --resume         # 항상 이어서
  python scripts/cycle_train.py --no-resume      # 항상 처음부터
  python scripts/cycle_train.py --cls-epochs 50 --seg-epochs 80
  python scripts/cycle_train.py --cycles 2

옵션:
  --cls-epochs N     분류기 에포크 수
  --seg-epochs N     분할기 에포크 수
  --cls-batch N      분류기 배치 크기
  --seg-batch N      분할기 배치 크기
  --cycles N         반복 횟수 (0=무한)
  --resume           체크포인트에서 이어서 (프롬프트 없음)
  --no-resume        처음부터 새로 시작 (프롬프트 없음)
  --no-caffeinate    macOS 절전 방지 비활성화
  --no-alt-screen    일반 터미널 출력 모드
  --interval FLOAT   모니터 갱신 주기(초, 기본 3.0)
"""

import argparse
import datetime
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from scripts.live_monitor import parse_monitor_state, render as monitor_render

PYTHON = str(ROOT / "venv" / "bin" / "python")
LOG_DIR = ROOT / "logs" / "cycle_train"
console = Console()

_CHECKPOINTS: list[tuple[str, Path, Path]] = [
    (
        "분류기",
        ROOT / "checkpoints" / "classifier" / "last_classifier.pth",
        ROOT / "checkpoints" / "classifier" / "best_classifier.pth",
    ),
    (
        "분할기",
        ROOT / "checkpoints" / "segmentor" / "last_segmentor.pth",
        ROOT / "checkpoints" / "segmentor" / "best_segmentor.pth",
    ),
]

_stop_event = threading.Event()


def _read_ckpt_info(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        epoch  = ckpt.get("epoch", "?")
        score  = ckpt.get("val_score", ckpt.get("val_dice", ckpt.get("val_metrics", {}).get("dice_positive", "?")))
        metric = ckpt.get("selection_metric", "?")
        phase  = ckpt.get("phase", "")
        return {"epoch": epoch, "score": score, "metric": metric, "phase": phase, "mtime": mtime}
    except Exception:
        return {"epoch": "?", "score": "?", "metric": "?", "phase": "", "mtime": mtime}


def _scan_checkpoints() -> list[dict]:
    found = []
    for label, last_path, best_path in _CHECKPOINTS:
        last_info = _read_ckpt_info(last_path)
        best_info = _read_ckpt_info(best_path)
        if last_info or best_info:
            found.append({
                "label": label,
                "last": last_info,
                "best": best_info,
                "last_path": last_path,
                "best_path": best_path,
            })
    return found


def _prompt_resume_mode() -> bool:
    found = _scan_checkpoints()
    if not found:
        console.print("[dim]  저장된 체크포인트 없음 → 처음부터 시작합니다[/dim]\n")
        return False

    table = Table(box=box.SIMPLE_HEAVY, border_style="cyan", header_style="bold cyan", show_edge=True)
    table.add_column("모델",       style="bold", width=7)
    table.add_column("종류",       width=6)
    table.add_column("Epoch",     justify="right", width=7)
    table.add_column("Best Score", justify="right", width=12)
    table.add_column("선택 지표",  width=16)
    table.add_column("단계",       width=8)
    table.add_column("저장 시각",  width=17)

    for item in found:
        for kind, info in [("last", item["last"]), ("best", item["best"])]:
            if info is None:
                continue
            score_str = f"{info['score']:.4f}" if isinstance(info["score"], float) else str(info["score"])
            table.add_row(
                item["label"] if kind == "last" else "",
                f"[yellow]{kind}[/yellow]" if kind == "last" else f"[green]{kind}[/green]",
                str(info["epoch"]),
                f"[green bold]{score_str}[/green bold]" if kind == "best" else score_str,
                str(info["metric"]),
                str(info["phase"]),
                f"[dim]{info['mtime']}[/dim]",
            )

    console.print(
        Panel(
            Group(Text("\n  저장된 체크포인트를 발견했습니다.\n", style="bold"), table),
            title="[bold cyan]학습 재개 선택[/bold cyan]",
            border_style="cyan",
        )
    )

    while True:
        try:
            answer = console.input(
                "  [bold]이어서 학습하시겠습니까?[/bold]\n"
                "  [cyan][1][/cyan] 이어서 학습  "
                "  [yellow][2][/yellow] 처음부터 새로 시작\n"
                "  선택 [dim](Enter = 이어서)[/dim]: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]중단됨[/yellow]")
            sys.exit(0)

        if answer in ("", "1"):
            console.print("  [green]→ 이어서 학습합니다[/green]\n")
            return True
        if answer == "2":
            console.print("  [yellow]→ 처음부터 새로 시작합니다[/yellow]\n")
            return False
        console.print("  [red]1 또는 2를 입력하세요[/red]")


def _install_signal_handler():
    def _handler(sig, frame):
        console.print("\n[yellow]중지 신호 수신 — 현재 단계 완료 후 종료합니다...[/yellow]")
        _stop_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _start_caffeinate() -> "subprocess.Popen | None":
    if platform.system() != "Darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if not caffeinate:
        return None
    return subprocess.Popen(
        [caffeinate, "-i", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _make_header(label: str, cycle: int, max_cycles: int, started_at: str) -> Text:
    t = Text()
    t.append("\n  사이클 ", style="dim")
    t.append(f"#{cycle}", style="bold cyan")
    if max_cycles > 0:
        t.append(f"/{max_cycles}", style="dim")
    t.append("  —  ", style="dim")
    t.append(label, style="bold yellow")
    t.append(f"\n  시작: {started_at}\n", style="dim")
    return t


def _make_idle_panel(message: str) -> Panel:
    return Panel(
        Text(f"\n  {message}\n", style="dim"),
        title="[bold blue]학습 모니터[/bold blue]",
        border_style="blue",
    )


def _run_step(cmd, log_path, label, cycle, max_cycles, interval, use_alt_screen) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = _make_header(label, cycle, max_cycles, started_at)
    return_codes: list[int] = []

    def _reader(proc, fh):
        try:
            for raw in proc.stdout:
                fh.write(raw)
                fh.flush()
        except Exception:
            pass
        proc.wait()
        return_codes.append(proc.returncode)

    with open(log_path, "w", encoding="utf-8", buffering=1) as log_fh:
        log_fh.write(f"# {label}  started {started_at}\n# cmd: {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        reader_thread = threading.Thread(target=_reader, args=(proc, log_fh), daemon=True)
        reader_thread.start()

        with Live(
            _make_idle_panel("학습 시작 대기 중..."),
            console=console,
            refresh_per_second=1,
            screen=use_alt_screen,
            vertical_overflow="crop",
        ) as live:
            while proc.poll() is None:
                state = parse_monitor_state(str(log_path))
                if state["rows"] or state.get("progress") or state.get("eta"):
                    live.update(
                        Panel(
                            Group(header, monitor_render(state)),
                            title=f"[bold blue]학습 모니터  —  {label}[/bold blue]",
                            border_style="blue",
                        )
                    )
                else:
                    live.update(_make_idle_panel(f"{label} 진행 중... (로그 대기)"))

                if _stop_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(interval)

            state = parse_monitor_state(str(log_path))
            if state["rows"] or state.get("progress") or state.get("eta"):
                live.update(
                    Panel(
                        Group(header, monitor_render(state)),
                        title=f"[bold blue]학습 모니터  —  {label} (완료)[/bold blue]",
                        border_style="green",
                    )
                )

    reader_thread.join(timeout=10)
    return return_codes[0] if return_codes else (proc.returncode or 0)


def _print_summary(cycle: int, cls_log: Path, seg_log: Path):
    console.print(Rule(f"사이클 #{cycle} 완료", style="green"))
    for label, log_path in [("분류기", cls_log), ("분할기", seg_log)]:
        state = parse_monitor_state(str(log_path))
        rows = state.get("rows", [])
        if rows:
            best = max(rows, key=lambda r: r.get("val_main", 0.0))
            console.print(
                f"  {label}  best val_main=[bold green]{best['val_main']:.4f}[/bold green]"
                f"  epoch {best['epoch']}/{best['total']}"
            )
        else:
            console.print(f"  {label}  (메트릭 없음)")
    console.print()


def main(args):
    _install_signal_handler()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    caf_proc = None
    if not args.no_caffeinate:
        caf_proc = _start_caffeinate()
        if caf_proc:
            console.print(f"[dim]caffeinate 시작 (PID {caf_proc.pid})[/dim]")

    try:
        if args.resume:
            use_resume = True
        elif args.no_resume:
            use_resume = False
        else:
            use_resume = _prompt_resume_mode()

        cls_extra: list[str] = []
        seg_extra: list[str] = []
        if args.cls_epochs:
            cls_extra += ["--epochs", str(args.cls_epochs)]
        if args.cls_batch:
            cls_extra += ["--batch_size", str(args.cls_batch)]
        if args.seg_epochs:
            seg_extra += ["--epochs", str(args.seg_epochs)]
        if args.seg_batch:
            seg_extra += ["--batch_size", str(args.seg_batch)]
        if use_resume:
            # ⚠ --resume은 현재 예약된 플래그로 실제 재개 로직 미구현.
            # training 스크립트가 플래그를 수신하나 무시하며 항상 처음부터 학습함.
            console.print("[yellow]⚠ resume 기능 미구현 — 처음부터 학습합니다.[/yellow]")
            cls_extra.append("--resume")
            seg_extra.append("--resume")

        cls_cmd = [PYTHON, "training/train_classifier.py"] + cls_extra
        seg_cmd = [PYTHON, "training/train_segmentor.py"] + seg_extra

        max_cycles = args.cycles
        cycle = 0

        resume_label = "[green]이어서[/green]" if use_resume else "[yellow]처음부터[/yellow]"
        console.rule("[bold cyan]2-class 사이클 훈련 + 실시간 모니터[/bold cyan]", style="cyan")
        console.print(f"  학습 모드  : {resume_label}")
        console.print(f"  최대 사이클: [cyan]{'무한' if max_cycles == 0 else max_cycles}[/cyan]")
        console.print(f"  로그 폴더  : [dim]{LOG_DIR}[/dim]")
        console.print(f"  분류기 cmd : [dim]{' '.join(cls_cmd)}[/dim]")
        console.print(f"  분할기 cmd : [dim]{' '.join(seg_cmd)}[/dim]\n")

        while not _stop_event.is_set():
            cycle += 1
            console.rule(
                f"[bold yellow]사이클 #{cycle}"
                + (f"/{max_cycles}" if max_cycles > 0 else "")
                + "  시작[/bold yellow]",
                style="yellow",
            )

            cls_log = LOG_DIR / f"{run_ts}_cls_cycle{cycle}.log"
            if not _stop_event.is_set():
                rc = _run_step(
                    cls_cmd, cls_log,
                    label=f"분류기  사이클#{cycle}",
                    cycle=cycle, max_cycles=max_cycles,
                    interval=args.interval,
                    use_alt_screen=not args.no_alt_screen,
                )
                if rc != 0:
                    console.print(f"[yellow]  분류기 종료 코드 {rc} — 분할기로 계속합니다[/yellow]")

            seg_log = LOG_DIR / f"{run_ts}_seg_cycle{cycle}.log"
            if not _stop_event.is_set():
                rc = _run_step(
                    seg_cmd, seg_log,
                    label=f"분할기  사이클#{cycle}",
                    cycle=cycle, max_cycles=max_cycles,
                    interval=args.interval,
                    use_alt_screen=not args.no_alt_screen,
                )
                if rc != 0:
                    console.print(f"[yellow]  분할기 종료 코드 {rc}[/yellow]")

            _print_summary(cycle, cls_log, seg_log)

            if max_cycles > 0 and cycle >= max_cycles:
                console.print(f"[green]지정된 사이클 수({max_cycles})에 도달했습니다. 종료합니다.[/green]")
                break

        console.rule("[bold green]사이클 훈련 종료[/bold green]", style="green")
        console.print(f"  완료된 사이클: {cycle}  |  로그: {LOG_DIR}\n")

    finally:
        if caf_proc and caf_proc.poll() is None:
            caf_proc.terminate()
            console.print("[dim]caffeinate 종료[/dim]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2-class 사이클 훈련 + Rich 실시간 모니터")
    parser.add_argument("--cls-epochs", type=int, default=None)
    parser.add_argument("--seg-epochs", type=int, default=None)
    parser.add_argument("--cls-batch",  type=int, default=None)
    parser.add_argument("--seg-batch",  type=int, default=None)
    parser.add_argument("--cycles",     type=int, default=1)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume",    action="store_true")
    resume_group.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-caffeinate", action="store_true")
    parser.add_argument("--no-alt-screen", action="store_true")
    parser.add_argument("--interval",   type=float, default=3.0)
    main(parser.parse_args())
