"""Foundation utilities: logging, seeding, GPU/environment detection, and
memory-safe JSONL/YAML I/O.

This module has zero internal (src/) imports and is the base of the
dependency graph — every other module may import from here, this module
imports from nothing else in the project.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

import yaml

_LOGGER_CONFIGURED = False


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the root project logger once; safe to call multiple times."""
    global _LOGGER_CONFIGURED
    root = logging.getLogger("telecom_t2c")
    if not _LOGGER_CONFIGURED:
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
        if log_file is not None:
            ensure_dir(log_file.parent)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(handler.formatter)
            root.addHandler(file_handler)
        root.propagate = False
        _LOGGER_CONFIGURED = True
    else:
        root.setLevel(level)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the project's root logger namespace."""
    if not _LOGGER_CONFIGURED:
        setup_logging()
    return logging.getLogger(f"telecom_t2c.{name}")


def set_seed(seed: int) -> None:
    """Seed python's random, numpy, and torch (CPU + all CUDA devices) if available."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


GpuFamily = Literal["A100", "L4", "T4", "OTHER", "CPU"]


@dataclass
class GPUInfo:
    name: str
    family: GpuFamily
    vram_gb: float
    bf16_supported: bool
    compute_capability: tuple[int, int]


def _classify_gpu_family(name: str) -> GpuFamily:
    upper = name.upper()
    if "A100" in upper:
        return "A100"
    if "L4" in upper and "L40" not in upper:
        return "L4"
    if "T4" in upper:
        return "T4"
    return "OTHER"


def detect_gpu() -> GPUInfo:
    """Detect the active CUDA GPU and classify it into a known family.

    Returns a GPUInfo with family="CPU" if no GPU is available, rather than
    raising — callers decide whether that's fatal for their use case.
    """
    try:
        import torch
    except ImportError:
        return GPUInfo(name="none (torch not installed)", family="CPU", vram_gb=0.0,
                        bf16_supported=False, compute_capability=(0, 0))

    if not torch.cuda.is_available():
        return GPUInfo(name="none (CUDA unavailable)", family="CPU", vram_gb=0.0,
                        bf16_supported=False, compute_capability=(0, 0))

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    return GPUInfo(
        name=name,
        family=_classify_gpu_family(name),
        vram_gb=vram_gb,
        bf16_supported=bf16_supported,
        compute_capability=(props.major, props.minor),
    )


@dataclass
class GPUStats:
    utilization_pct: Optional[float]
    memory_used_gb: float
    memory_reserved_gb: float
    memory_total_gb: float


def get_gpu_stats() -> Optional[GPUStats]:
    """Snapshot current GPU memory usage (+ utilization if nvidia-ml-py is installed).

    Returns None if no GPU is available — callers (e.g. GPUCallback) should
    skip logging rather than fail.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None

    memory_used_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
    memory_reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
    memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    utilization_pct: Optional[float] = None
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization_pct = float(util.gpu)
        pynvml.nvmlShutdown()
    except Exception:
        utilization_pct = None

    return GPUStats(
        utilization_pct=utilization_pct,
        memory_used_gb=memory_used_gb,
        memory_reserved_gb=memory_reserved_gb,
        memory_total_gb=memory_total_gb,
    )


def read_jsonl(path: Path) -> Iterator[dict]:
    """Stream a JSONL file line-by-line as dicts.

    Never materializes the full file in memory — required for the real
    dataset (train_sft_batched.jsonl is ~544MB / 91k lines). Lines that fail
    to parse as JSON are skipped with a logged warning rather than raising,
    so a single corrupted line doesn't abort the whole read.
    """
    logger = get_logger("utils")
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping corrupted JSON at %s:%d — %s", path, line_number, exc)
                continue


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dicts to a JSONL file, one JSON object per line."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def count_jsonl_lines(path: Path) -> int:
    """Fast non-parsing line count, for progress/stat display on large files."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_yaml(path: Path) -> dict:
    """Read a YAML file into a dict. Raises FileNotFoundError / yaml.YAMLError as-is
    (callers such as config.load_config wrap these into actionable errors)."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist; return it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_git_hash(repo_dir: Optional[Path] = None) -> str:
    """Return the current git commit hash, or "unknown" if unavailable.

    Gracefully degrades (never raises) when git isn't installed, the
    directory isn't a git repo, or there are no commits yet — this project
    directory may not always be a fully-initialized repo with history.
    """
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass
    return "unknown"


def is_colab() -> bool:
    """True when running inside a Google Colab kernel."""
    return "google.colab" in sys.modules


def mount_google_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive if running in Colab; no-op (returns None) otherwise.

    Never raises — Drive sync is a convenience feature, not a hard
    requirement, so failures here should degrade gracefully.
    """
    logger = get_logger("utils")
    if not is_colab():
        logger.info("Not running in Colab — skipping Google Drive mount.")
        return None
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount(mount_point)
        return Path(mount_point)
    except Exception as exc:  # pragma: no cover - Colab-only path
        logger.warning("Google Drive mount failed: %s", exc)
        return None


def human_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. '543.2 MB')."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def timestamp_run_id() -> str:
    """Return a sortable run-id timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def disable_unused_transformers_backends() -> None:
    """Force transformers' is_torchaudio_available()/is_torchao_available() to
    always return False, regardless of actual package presence.

    This project is text-only (never touches torchaudio) and always uses
    bitsandbytes 4-bit QLoRA via Unsloth (never TorchAO quantization) — so
    transformers never legitimately needs either package. Both have caused
    confirmed import failures on Colab images where the installed build is
    present-but-broken relative to the installed torch build (a circular
    import inside torchaudio's own CUDA-version check; a torch op signature
    mismatch inside torchao/dtypes/nf4tensor.py) — and both availability
    checks only confirm the package is *present*, not that importing it
    actually succeeds, so a broken install crashes unrelated code paths
    (nearly any Auto* class transitively imports transformers/modeling_utils.py,
    which unconditionally imports both quantizer modules).

    pip-uninstalling both (see the notebook's Install section) is a first
    line of defense but isn't reliable alone: both checks are `@lru_cache`d,
    so if either gets called even once earlier in the same kernel process —
    before a later uninstall takes effect, or across cells without an
    intervening restart — the cached result never re-checks reality again
    for the rest of that process; a subsequent uninstall can leave the flag
    stuck at a stale True. Patching the check itself, called as early as
    possible (right after Install, before anything else touches
    transformers), sidesteps both the "present but broken" and "stale cache"
    failure modes at once, and is safe precisely because this project never
    needs either package for anything. No-ops silently if transformers isn't
    installed yet; safe to call multiple times.
    """
    try:
        import transformers.utils as _t_utils
        import transformers.utils.import_utils as _t_import_utils
    except Exception:
        return

    def _always_false(*_args: object, **_kwargs: object) -> bool:
        return False

    for name in ("is_torchaudio_available", "is_torchao_available"):
        setattr(_t_import_utils, name, _always_false)
        setattr(_t_utils, name, _always_false)
