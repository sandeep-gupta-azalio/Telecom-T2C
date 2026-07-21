"""Adapter/run provenance manifest.

Owns the manifest.json shape (a superset of the reference notebook's
telecom_finetune_manifest.json — see notebook section 11) plus the prior-
adapter inheritance rule (notebook section 2d), so both live in exactly one
place.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src import utils
from src.config import ExperimentConfig

logger = utils.get_logger("manifest")

# Legacy filename used by the reference notebook; supported as a read fallback
# for backward compatibility with existing adapter directories.
_LEGACY_MANIFEST_NAME = "telecom_finetune_manifest.json"
_MANIFEST_NAME = "manifest.json"

# Fields inherited from a prior adapter's manifest so a continued run stays
# architecturally compatible with the adapter it's continuing (notebook §2d).
_INHERITED_FIELDS = ("base_model", "lora_r", "lora_alpha", "max_seq_length")


@dataclass
class Manifest:
    base_model: str
    continued_from: Optional[str]
    prior_manifest: Optional[dict[str, Any]]
    reasoning: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_seq_length: int
    packing: bool
    learning_rate: float
    train_rows: int
    val_rows: int
    golden_rows: Optional[int]
    training_mode: str
    dataset_version: str
    lora_version: str
    generator_version: str
    validator_version: str
    git_hash: str
    run_id: str
    created_at: str
    experiment_name: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_prior_manifest(adapter_dir: Optional[str]) -> Optional[dict[str, Any]]:
    """Load a prior adapter's manifest.json (or legacy telecom_finetune_manifest.json).

    Returns None gracefully when adapter_dir is None, doesn't exist, or has
    no manifest file present — this is expected and normal for a fresh LoRA
    init run (the project's current default), not an error.
    """
    if not adapter_dir:
        return None
    adapter_path = Path(adapter_dir)
    if not adapter_path.is_dir():
        logger.info("continue_adapter path does not exist (yet): %s — no prior manifest loaded.", adapter_path)
        return None

    for name in (_MANIFEST_NAME, _LEGACY_MANIFEST_NAME):
        candidate = adapter_path / name
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                logger.info("Loaded prior manifest: %s", candidate)
                return data
            except json.JSONDecodeError as exc:
                logger.warning("Prior manifest at %s is corrupted JSON: %s — ignoring.", candidate, exc)
                return None

    logger.info("No manifest.json or %s found under %s — using config defaults.", _LEGACY_MANIFEST_NAME, adapter_path)
    return None


def apply_prior_manifest_inheritance(
    config: ExperimentConfig, prior: Optional[dict[str, Any]]
) -> ExperimentConfig:
    """Override base_model / lora_r / lora_alpha / max_seq_length from a prior manifest.

    A continued LoRA adapter must be attached to the same base model and rank
    it was trained with, so those fields take precedence over
    configs/experiment.yaml when continuing (matches notebook §2d). No-op
    (returns config unchanged) when prior is None, e.g. the default fresh-init
    run.
    """
    if not prior:
        return config

    resolved = copy.deepcopy(config)
    if "base_model" in prior:
        resolved.model.base_model = str(prior["base_model"])
    if "lora_r" in prior:
        resolved.lora.lora_r = int(prior["lora_r"])
    if "lora_alpha" in prior:
        resolved.lora.lora_alpha = int(prior["lora_alpha"])
    if "max_seq_length" in prior:
        resolved.data.max_seq_length = int(prior["max_seq_length"])

    changed = [f for f in _INHERITED_FIELDS if f in prior]
    if changed:
        logger.info("Inherited from prior manifest: %s", ", ".join(changed))
    return resolved


def build_manifest(
    config: ExperimentConfig,
    prior_manifest: Optional[dict[str, Any]],
    train_rows: int,
    val_rows: int,
    golden_rows: Optional[int],
    run_id: str,
) -> Manifest:
    """Assemble the manifest for a completed (or in-progress) run."""
    training_mode = "continue_lora" if config.model.continue_adapter else "fresh_lora"
    return Manifest(
        base_model=config.model.base_model,
        continued_from=config.model.continue_adapter,
        prior_manifest=prior_manifest,
        reasoning=config.training.reasoning,
        lora_r=config.lora.lora_r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        max_seq_length=config.data.max_seq_length,
        packing=config.training.packing,
        learning_rate=config.training.learning_rate,
        train_rows=train_rows,
        val_rows=val_rows,
        golden_rows=golden_rows,
        training_mode=training_mode,
        dataset_version=config.identity.dataset_version,
        lora_version=config.identity.lora_version,
        generator_version=config.identity.generator_version,
        validator_version=config.identity.validator_version,
        git_hash=utils.get_git_hash(),
        run_id=run_id,
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        experiment_name=config.identity.experiment_name,
    )


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Write a Manifest to path as pretty-printed JSON."""
    utils.ensure_dir(path.parent)
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    logger.info("Manifest written to %s", path)


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest.json file back into a plain dict. Raises FileNotFoundError as-is."""
    return json.loads(path.read_text(encoding="utf-8"))
