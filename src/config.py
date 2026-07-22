"""Single source of truth for all tunables.

configs/experiment.yaml is the only file a user should need to edit between
runs; every other module in src/ receives its settings as an ExperimentConfig
instance rather than parsing YAML itself.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from src import utils

logger = utils.get_logger("config")

_RUN_DIR_RE = re.compile(r"^run_(\d{8}_\d{6})$")
_REQUIRED_DATA_FIELDS = ("train_path", "val_path")


class ConfigError(Exception):
    """Raised when configs/experiment.yaml is missing, malformed, or missing required fields."""


@dataclass
class IdentityConfig:
    experiment_name: str = "telecom_t2c_gemma4"
    dataset_version: str = "phase1"
    lora_version: str = "v1"
    generator_version: str = "unknown"
    validator_version: str = "unknown"
    seed: int = 42


@dataclass
class DataConfig:
    train_path: str = ""
    val_path: str = ""
    golden_path: Optional[str] = None
    eval_source: str = "val"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = 256
    max_seq_length: int = 1536


@dataclass
class ModelConfig:
    base_model: str = "google/gemma-4-12B-it"
    continue_adapter: Optional[str] = None
    hf_token_env_var: str = "HF_TOKEN"


@dataclass
class LoraConfigSection:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None
    lora_bias: str = "none"


@dataclass
class TrainingConfig:
    epochs: float = 1.0
    learning_rate: float = 1e-4
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation: int = 4
    packing: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2
    reasoning: bool = True  # documented no-op on data content today — see README
    gradient_checkpointing: bool = True


@dataclass
class EvaluationConfig:
    run_eval: bool = True
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    max_new_tokens_eval: int = 512


@dataclass
class WandbConfig:
    wandb_project: str = "telecom-t2c-trainer"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"
    wandb_log_gpu_stats: bool = True
    wandb_watch_model: bool = False


@dataclass
class DriveConfig:
    google_drive_directory: Optional[str] = "/content/drive/MyDrive/telecom_t2c"
    copy_to_drive: bool = True
    drive_mount_point: str = "/content/drive"


@dataclass
class HardwareConfig:
    training_gpu: int = 0
    gpu_profile_override: Optional[str] = None


@dataclass
class ReproducibilityConfig:
    output_directory: str = "outputs"
    resume_training: bool = True
    run_id: Optional[str] = None


@dataclass
class ExperimentConfig:
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigSection = field(default_factory=LoraConfigSection)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    drive: DriveConfig = field(default_factory=DriveConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        try:
            return cls(
                identity=IdentityConfig(**(data.get("identity") or {})),
                data=DataConfig(**(data.get("data") or {})),
                model=ModelConfig(**(data.get("model") or {})),
                lora=LoraConfigSection(**(data.get("lora") or {})),
                training=TrainingConfig(**(data.get("training") or {})),
                evaluation=EvaluationConfig(**(data.get("evaluation") or {})),
                wandb=WandbConfig(**(data.get("wandb") or {})),
                drive=DriveConfig(**(data.get("drive") or {})),
                hardware=HardwareConfig(**(data.get("hardware") or {})),
                reproducibility=ReproducibilityConfig(**(data.get("reproducibility") or {})),
            )
        except TypeError as exc:
            raise ConfigError(f"Unrecognized field in experiment.yaml: {exc}") from exc


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate configs/experiment.yaml into an ExperimentConfig.

    Raises ConfigError (not a bare exception) with an actionable message for:
    missing file, malformed YAML, unrecognized fields, or missing required
    data.train_path / data.val_path.
    """
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. Copy configs/experiment.yaml and point "
            "load_config at it, or check your working directory."
        )
    try:
        raw = utils.read_yaml(path)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at {path}: {exc}") from exc

    config = ExperimentConfig.from_dict(raw)

    missing = [name for name in _REQUIRED_DATA_FIELDS if not getattr(config.data, name)]
    if missing:
        raise ConfigError(
            f"Missing required data.* fields in {path}: {', '.join(missing)}. "
            "Set data.train_path and data.val_path to your dataset JSONL files."
        )
    return config


def validate_config(config: ExperimentConfig) -> list[str]:
    """Return a list of non-fatal warnings about the resolved config.

    Never raises — these are advisory, printed by the notebook's
    Configuration section so the user can catch mistakes before a multi-hour
    training run starts.
    """
    warnings: list[str] = []

    if config.data.eval_source == "golden" and not config.data.golden_path:
        warnings.append(
            "data.eval_source='golden' but data.golden_path is unset — "
            "falling back to 'val' for loss-based eval during training."
        )
    if config.data.golden_path and not Path(config.data.golden_path).is_file():
        warnings.append(
            f"data.golden_path is set but not found on disk: {config.data.golden_path} — "
            "golden evaluation will be skipped."
        )
    if config.model.continue_adapter and not Path(config.model.continue_adapter).is_dir():
        warnings.append(
            f"model.continue_adapter is set but not found on disk: {config.model.continue_adapter} — "
            "this will fail at model-load time unless the path exists at runtime (e.g. on Drive)."
        )
    if config.evaluation.early_stopping and not config.evaluation.run_eval:
        warnings.append(
            "evaluation.early_stopping=true requires evaluation.run_eval=true — "
            "early stopping will have no effect without eval runs to compare against."
        )
    if config.wandb.wandb_mode not in ("online", "offline", "disabled"):
        warnings.append(
            f"wandb.wandb_mode={config.wandb.wandb_mode!r} is not one of "
            "online/offline/disabled — wandb_logger will treat this as 'online'."
        )
    if config.drive.copy_to_drive and not config.drive.google_drive_directory:
        warnings.append(
            "drive.copy_to_drive=true but drive.google_drive_directory is unset — Drive sync will be skipped."
        )
    if config.data.eval_source not in ("val", "golden"):
        warnings.append(
            f"data.eval_source={config.data.eval_source!r} is not 'val' or 'golden' — treating as 'val'."
        )
    return warnings


def resolve_run_dir(config: ExperimentConfig) -> Path:
    """Resolve (and create) the run directory for this session.

    - If reproducibility.run_id is set, always use outputs/runs/<run_id>/
      (explicit override, created if it doesn't exist yet).
    - Else if resume_training is true, reuse the most recent existing run
      directory that already has a resumable checkpoint under adapter/ —
      without this, "resume automatically after interruption" has nowhere to
      resume *into*, since a fresh call would otherwise always mint a new
      timestamped directory.
    - Else mint a new outputs/runs/run_<timestamp>/ directory.
    """
    runs_root = utils.ensure_dir(Path(config.reproducibility.output_directory) / "runs")

    if config.reproducibility.run_id:
        return utils.ensure_dir(runs_root / config.reproducibility.run_id)

    if config.reproducibility.resume_training:
        resumable = _find_most_recent_resumable_run(runs_root)
        if resumable is not None:
            logger.info("Resuming existing run directory: %s", resumable)
            return resumable

    run_dir = runs_root / f"run_{utils.timestamp_run_id()}"
    return utils.ensure_dir(run_dir)


def _find_most_recent_resumable_run(runs_root: Path) -> Optional[Path]:
    if not runs_root.is_dir():
        return None
    candidates: list[tuple[str, Path]] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        match = _RUN_DIR_RE.match(child.name)
        if not match:
            continue
        adapter_dir = child / "adapter"
        if adapter_dir.is_dir() and any(adapter_dir.glob("checkpoint-*")):
            candidates.append((match.group(1), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def save_resolved_config(config: ExperimentConfig, path: Path) -> None:
    """Write the effective, fully-resolved config (post prior-manifest inheritance) to path."""
    utils.ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False, default_flow_style=False)
