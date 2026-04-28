import datetime as dt
import os
import platform
import sys
import time
import warnings

import torch


def configure_torch_runtime(cfg: dict) -> None:
    """Apply runtime knobs that are safe after torch import."""
    perf = cfg.get("performance", {})

    threads = perf.get("torch_num_threads")
    if threads:
        torch.set_num_threads(int(threads))

    interop_threads = perf.get("torch_num_interop_threads")
    if interop_threads:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            pass

    precision = perf.get("float32_matmul_precision")
    if precision and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def runtime_summary(device: torch.device) -> str:
    parts = [
        f"python={platform.python_version()}",
        f"torch={torch.__version__}",
        f"threads={torch.get_num_threads()}",
        f"interop={torch.get_num_interop_threads()}",
        f"mps_high={os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '-')}",
        f"mps_low={os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO', '-')}",
    ]
    if device.type == "mps" and hasattr(torch.mps, "recommended_max_memory"):
        try:
            gb = torch.mps.recommended_max_memory() / 1024**3
            parts.append(f"mps_recommended={gb:.1f}GB")
        except RuntimeError as exc:
            parts.append(f"mps_recommended=unavailable({exc})")
    return "  ".join(parts)


def suppress_noisy_runtime_warnings() -> None:
    """Silence environment-specific warnings that clutter long training logs."""
    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*A new version of Albumentations is available:.*",
        category=UserWarning,
    )


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_dataloader_runtime(device: torch.device,
                               num_workers: int,
                               prefetch_factor: int) -> dict:
    """Choose stable DataLoader settings for the current runtime."""
    workers = max(0, int(num_workers))
    prefetch = max(1, int(prefetch_factor))
    persistent_workers = workers > 0
    adjusted = False

    if device.type == "mps" and platform.system() == "Darwin":
        capped_workers = min(workers, 4)
        capped_prefetch = min(prefetch, 2)
        if capped_workers != workers or capped_prefetch != prefetch or persistent_workers:
            adjusted = True
        workers = capped_workers
        prefetch = capped_prefetch
        persistent_workers = False

    return {
        "num_workers": workers,
        "prefetch_factor": prefetch,
        "persistent_workers": persistent_workers,
        "adjusted": adjusted,
    }


def should_show_progress() -> bool:
    """Render tqdm bars only on an interactive terminal."""
    return bool(sys.stdout.isatty())


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def epoch_eta_message(epoch: int,
                      total_epochs: int,
                      epoch_seconds: float,
                      epoch_history: list,
                      recent_window: int = 5) -> str:
    history = list(epoch_history) if epoch_history else [float(epoch_seconds)]
    recent = history[-min(len(history), recent_window):]
    recent_avg = sum(recent) / max(1, len(recent))
    remaining_epochs = max(0, int(total_epochs) - int(epoch))
    remaining_seconds = recent_avg * remaining_epochs
    eta_at = dt.datetime.now() + dt.timedelta(seconds=remaining_seconds)
    return (
        f"    -> epoch_time={format_duration(epoch_seconds)} | "
        f"recent_avg={format_duration(recent_avg)} | "
        f"remaining={format_duration(remaining_seconds)} | "
        f"finish~{eta_at.strftime('%Y-%m-%d %H:%M')}"
    )


class BatchProgressLogger:
    """Write low-frequency progress lines for tee/no-TTY runs."""

    def __init__(self,
                 phase: str,
                 split: str,
                 epoch: int,
                 total_epochs: int,
                 total_steps: int,
                 enabled: bool,
                 report_points: int = 10):
        self.phase = str(phase)
        self.split = str(split)
        self.epoch = int(epoch)
        self.total_epochs = int(total_epochs)
        self.total_steps = max(1, int(total_steps))
        self.enabled = bool(enabled)
        self.report_every = max(1, self.total_steps // max(1, int(report_points)))
        self.started_at = time.time()
        self._last_reported_step = 0

    def update(self, step: int) -> None:
        if not self.enabled:
            return

        step = max(1, min(int(step), self.total_steps))
        should_report = (
            step == 1
            or step == self.total_steps
            or (step - self._last_reported_step) >= self.report_every
        )
        if not should_report:
            return

        self._last_reported_step = step
        pct = step / self.total_steps * 100.0
        elapsed = format_duration(time.time() - self.started_at)
        print(
            f"    [Progress] phase={self.phase} split={self.split} "
            f"epoch={self.epoch}/{self.total_epochs} "
            f"batch={step}/{self.total_steps} pct={pct:.0f} elapsed={elapsed}",
            flush=True,
        )
