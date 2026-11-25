"""Diagnostics and evaluation metrics for representation quality (RankMe, logging).

Strictly follows docs/TRAIN_CONTRACT.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np


class ProgressBar:
    """Zero-dependency terminal progress bar with dynamic ETA, step rates, and metric tracking."""

    def __init__(
        self,
        total: int,
        prefix: str = "",
        bar_len: int = 24,
        log_every: int = 5,
        is_tty: Optional[bool] = None,
        log_file: Optional[str] = None,
        process_tag: Optional[str] = None,
    ):
        self.total = max(1, total)
        self.prefix = prefix
        self.bar_len = bar_len
        self.log_every = max(1, log_every)
        self.is_tty = sys.stdout.isatty() if is_tty is None else is_tty
        self.log_file = log_file
        self.pid = os.getpid()
        self.process_tag = process_tag or f"PID {self.pid}"
        self.start_time = time.time()
        self._last_line_len = 0

    @staticmethod
    def _format_time(seconds: float) -> str:
        s = int(round(seconds))
        if s < 60:
            return f"{s:02d}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m:02d}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h:d}h{m:02d}m"

    def _append_to_file(self, line: str) -> None:
        """Append log line to log_file if path is specified."""
        if not self.log_file:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self.log_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line.strip() + "\n")
        except Exception:
            pass

    def update(self, current: int, metrics: Optional[Dict[str, Any]] = None):
        """Update progress bar display with current step and metrics dict."""
        is_periodic = (current % self.log_every == 0) or (current == self.total)
        if not is_periodic and not self.is_tty:
            return

        now = time.time()
        elapsed = now - self.start_time
        rate = elapsed / max(current, 1)
        remaining = max(0.0, (self.total - current) * rate)

        time_str = f"{rate:5.2f}s/step" if rate >= 1.0 else f"{1.0 / max(rate, 1e-4):5.1f}step/s"
        eta_str = self._format_time(remaining)
        elap_str = self._format_time(elapsed)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        metric_parts = []
        if metrics and is_periodic:
            for k, v in metrics.items():
                if isinstance(v, (float, np.floating, jax.Array)):
                    val = float(v)
                    if abs(val) < 1e-3 and val != 0:
                        metric_parts.append(f"{k}: {val:.2e}")
                    else:
                        metric_parts.append(f"{k}: {val:.4f}")
                else:
                    metric_parts.append(f"{k}: {v}")
        metrics_str = " | ".join(metric_parts)

        fraction = min(1.0, current / self.total)
        step_w = len(str(self.total))
        step_str = f"{current:0{step_w}d}/{self.total:0{step_w}d}"

        filled_len = int(round(self.bar_len * fraction))
        if 0 < filled_len < self.bar_len:
            bar = "=" * (filled_len - 1) + ">"
        else:
            bar = "=" * filled_len
        bar = bar.ljust(self.bar_len, "-")

        is_periodic = (current % self.log_every == 0) or (current == self.total)

        if self.is_tty:
            status = f"\r{self.prefix} [{bar}] {step_str} ({fraction * 100:4.1f}%) | {time_str} | ETA: {eta_str} | {metrics_str}"
            cols = shutil.get_terminal_size((100, 20)).columns
            clear_width = max(self._last_line_len, cols - 1)

            if is_periodic and current < self.total:
                # Clear active bar line and print permanent mini-log line
                sys.stdout.write("\r" + " " * clear_width + "\r")
                mini_log = f"[{now_ts}] [{self.process_tag}] [{self.prefix.strip()} | Step {step_str}] {metrics_str} | {time_str} | Elapsed: {elap_str} | ETA: {eta_str}"
                sys.stdout.write(mini_log + "\n")
                self._append_to_file(mini_log)
                sys.stdout.write(status[: cols - 1])
                sys.stdout.flush()
                self._last_line_len = len(status[: cols - 1])
            elif current == self.total:
                # Finished: clear bar line and print final summary mini-log
                sys.stdout.write("\r" + " " * clear_width + "\r")
                mini_log = f"[{now_ts}] [{self.process_tag}] [{self.prefix.strip()} | Step {step_str}] {metrics_str} | {time_str} | Elapsed: {elap_str}"
                sys.stdout.write(mini_log + "\n")
                self._append_to_file(mini_log)
                sys.stdout.flush()
                self._last_line_len = 0
            else:
                sys.stdout.write(status[: cols - 1])
                sys.stdout.flush()
                self._last_line_len = len(status[: cols - 1])
        else:
            # Non-TTY mode (e.g. redirected output, CI, background log)
            if is_periodic or current == 1:
                line = f"[{now_ts}] [{self.process_tag}] [{self.prefix.strip()} | Step {step_str}] ({fraction * 100:4.1f}%) | {time_str} | Elapsed: {elap_str} | ETA: {eta_str} | {metrics_str}"
                print(line, flush=True)
                self._append_to_file(line)



def compute_rankme(embeddings: np.ndarray | jax.Array, eps: float = 1e-7) -> float:
    """Compute RankMe effective rank metric on uncentered embeddings.

    RankMe (Garrido et al., NeurIPS 2023) measures the continuous effective rank
    of a representation matrix without thresholding singular values:
      p_k = sigma_k / (sum_j sigma_j + eps)
      RankMe = exp(-sum_k p_k * log(p_k + eps))

    Note: As specified in the RankMe paper and docs/TRAIN_CONTRACT.md, the singular
    value decomposition is computed directly on the UNCENTERED matrix (do NOT subtract
    the mean), preserving the origin and full feature energy.

    Args:
        embeddings: Latent representations of shape (N, D) or (..., D).
        eps: Small epsilon for numerical stability.

    Returns:
        Continuous effective dimensionality in [1.0, D].
    """
    z = jnp.asarray(embeddings, dtype=jnp.float32)
    if z.ndim > 2:
        z = z.reshape((-1, z.shape[-1]))

    # Compute singular values on uncentered embedding matrix
    s = jnp.linalg.svd(z, compute_uv=False)
    p = s / (jnp.sum(s) + eps)
    entropy = -jnp.sum(p * jnp.log(p + eps))
    return float(jnp.exp(entropy))


def append_metrics_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a training/validation metric record to a JSON Lines (jsonl) file.

    Handles converting JAX/NumPy scalar and array values to JSON-serializable types.

    Args:
        path: Path to target .jsonl file.
        record: Dictionary of metrics for the current step/epoch.
    """
    def _serialize(v: Any) -> Any:
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, (np.ndarray, jax.Array)):
            return v.tolist()
        return v

    clean_record = {k: _serialize(v) for k, v in record.items()}
    if "timestamp" not in clean_record:
        clean_record["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "pid" not in clean_record:
        clean_record["pid"] = os.getpid()

    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_record) + "\n")
