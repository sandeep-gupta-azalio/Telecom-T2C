"""Weights & Biases integration.

Every other module talks to wandb only through WandbLogger, never imports
`wandb` directly — this is the single no-op-safe gateway that guarantees
training never blocks or crashes because wandb is unavailable, uninstalled,
or the user isn't logged in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from src import utils
from src.config import WandbConfig

logger = utils.get_logger("wandb_logger")


class WandbLogger:
    """No-op-safe wrapper around the wandb SDK."""

    def __init__(self, wandb_config: WandbConfig, run_metadata: dict[str, Any]):
        self.wandb_config = wandb_config
        self.run_metadata = run_metadata
        self._enabled = False
        self._run: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def init(self) -> bool:
        """Attempt to start a wandb run. Never raises — sets enabled=False and warns on failure."""
        mode = self.wandb_config.wandb_mode
        if mode == "disabled":
            logger.info("wandb.wandb_mode=disabled — wandb logging is off.")
            self._enabled = False
            return False

        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed — continuing without experiment tracking.")
            self._enabled = False
            return False

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata  # type: ignore[import-not-found]

                api_key = userdata.get("WANDB_API_KEY")
                if api_key:
                    os.environ["WANDB_API_KEY"] = api_key
            except Exception:
                pass

        effective_mode = mode
        if not api_key and mode == "online":
            logger.warning(
                "No WANDB_API_KEY found in environment or Colab secrets — falling back to "
                "wandb_mode='offline' (metrics logged locally, not synced)."
            )
            effective_mode = "offline"

        try:
            self._run = wandb.init(
                project=self.wandb_config.wandb_project,
                entity=self.wandb_config.wandb_entity,
                name=self.run_metadata.get("experiment_name"),
                mode=effective_mode,
                config=self.run_metadata,
            )
            self._enabled = True
            logger.info("wandb run started: project=%s mode=%s", self.wandb_config.wandb_project, effective_mode)
        except Exception as exc:
            logger.warning("wandb.init() failed (%s) — continuing without experiment tracking.", exc)
            self._enabled = False
        return self._enabled

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if not self._enabled:
            return
        try:
            import wandb

            wandb.log(metrics, step=step)
        except Exception as exc:
            logger.warning("wandb.log() failed: %s", exc)

    def log_gpu_stats(self, stats: Optional[utils.GPUStats], step: Optional[int] = None) -> None:
        if not self._enabled or stats is None:
            return
        payload = {
            "gpu/memory_used_gb": stats.memory_used_gb,
            "gpu/memory_reserved_gb": stats.memory_reserved_gb,
            "gpu/memory_total_gb": stats.memory_total_gb,
        }
        if stats.utilization_pct is not None:
            payload["gpu/utilization_pct"] = stats.utilization_pct
        self.log_metrics(payload, step=step)

    def log_system(
        self,
        examples_per_sec: float,
        tokens_per_sec: float,
        eta_seconds: float,
        epoch: float,
        step: int,
    ) -> None:
        self.log_metrics(
            {
                "system/examples_per_sec": examples_per_sec,
                "system/tokens_per_sec": tokens_per_sec,
                "system/eta_seconds": eta_seconds,
                "system/epoch": epoch,
                "system/step": step,
            },
            step=step,
        )

    def upload_artifact(self, path: Path, artifact_type: str, name: str) -> None:
        """Upload a file or directory as a wandb Artifact.

        Each call is independently guarded so one failed upload (e.g. a
        missing predictions/ directory when golden eval was skipped) never
        aborts the others.
        """
        if not self._enabled:
            return
        if not path.exists():
            logger.info("Skipping artifact upload for %s (%s) — path does not exist.", name, path)
            return
        try:
            import wandb

            artifact = wandb.Artifact(name=name, type=artifact_type)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
            logger.info("Uploaded wandb artifact: %s (%s)", name, artifact_type)
        except Exception as exc:
            logger.warning("Failed to upload artifact %s: %s", name, exc)

    def watch_model(self, model: Any) -> None:
        if not self._enabled or not self.wandb_config.wandb_watch_model:
            return
        try:
            import wandb

            wandb.watch(model)
        except Exception as exc:
            logger.warning("wandb.watch() failed: %s", exc)

    def finish(self) -> None:
        if not self._enabled:
            return
        try:
            import wandb

            wandb.finish()
        except Exception as exc:
            logger.warning("wandb.finish() failed: %s", exc)
        finally:
            self._enabled = False
