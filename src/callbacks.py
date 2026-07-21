"""Custom transformers.TrainerCallback implementations.

PredictionCallback takes sample_fn/decode_fn as injected callables (wired by
trainer.py, which already depends on both evaluator.py and inference.py)
rather than importing those modules itself — this keeps the dependency graph
a strict DAG.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from transformers import TrainerCallback

from src import utils
from src.wandb_logger import WandbLogger

logger = utils.get_logger("callbacks")


class TrainingCallback(TrainerCallback):
    """Forwards train loss / learning rate / grad norm / epoch / step to wandb."""

    def __init__(self, wandb_logger: WandbLogger, run_metadata: dict[str, Any]):
        self.wandb_logger = wandb_logger
        self.run_metadata = run_metadata
        self._start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        scalar_metadata = {
            f"run/{key}": value
            for key, value in self.run_metadata.items()
            if isinstance(value, (str, int, float, bool))
        }
        if scalar_metadata:
            self.wandb_logger.log_metrics(scalar_metadata, step=0)

    def on_log(self, args, state, control, logs: Optional[dict] = None, **kwargs):
        if not logs:
            return
        metrics = {
            f"train/{key}": logs[key]
            for key in ("loss", "learning_rate", "grad_norm", "epoch")
            if key in logs
        }
        if metrics:
            self.wandb_logger.log_metrics(metrics, step=state.global_step)

        if self._start_time is not None and state.global_step > 0:
            elapsed = time.time() - self._start_time
            effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            examples_seen = state.global_step * effective_batch
            examples_per_sec = examples_seen / max(elapsed, 1e-6)
            max_len = getattr(args, "max_length", None) or 0
            tokens_per_sec = examples_per_sec * max_len
            total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else None
            eta_seconds = (
                ((total_steps - state.global_step) * (elapsed / state.global_step))
                if total_steps
                else 0.0
            )
            self.wandb_logger.log_system(
                examples_per_sec=examples_per_sec,
                tokens_per_sec=tokens_per_sec,
                eta_seconds=eta_seconds,
                epoch=logs.get("epoch", 0.0),
                step=state.global_step,
            )

    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Training loop finished at step %d.", state.global_step)


class EvaluationCallback(TrainerCallback):
    """Forwards validation loss/metrics to wandb after each eval pass."""

    def __init__(self, wandb_logger: WandbLogger):
        self.wandb_logger = wandb_logger

    def on_evaluate(self, args, state, control, metrics: Optional[dict] = None, **kwargs):
        if not metrics:
            return
        payload = {
            f"eval/{key.removeprefix('eval_')}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }
        if payload:
            self.wandb_logger.log_metrics(payload, step=state.global_step)


class GPUCallback(TrainerCallback):
    """Periodically samples and logs GPU memory/utilization stats to wandb."""

    def __init__(self, wandb_logger: WandbLogger, log_every_n_steps: int = 50):
        self.wandb_logger = wandb_logger
        self.log_every_n_steps = max(log_every_n_steps, 1)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        stats = utils.get_gpu_stats()
        if stats is not None:
            self.wandb_logger.log_gpu_stats(stats, step=state.global_step)


class PredictionCallback(TrainerCallback):
    """Periodically generates a small sample of predictions and logs them as a wandb Table."""

    def __init__(
        self,
        sample_fn: Callable[[], list[dict]],
        decode_fn: Callable[[Any, Any, list[dict], int], str],
        wandb_logger: WandbLogger,
        every_n_evals: int = 1,
        sample_size: int = 8,
        max_new_tokens: int = 256,
    ):
        self.sample_fn = sample_fn
        self.decode_fn = decode_fn
        self.wandb_logger = wandb_logger
        self.every_n_evals = max(every_n_evals, 1)
        self.sample_size = sample_size
        self.max_new_tokens = max_new_tokens
        self._eval_count = 0

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        self._eval_count += 1
        if self._eval_count % self.every_n_evals != 0 or not self.wandb_logger.enabled:
            return

        try:
            samples = self.sample_fn()
        except Exception as exc:
            logger.warning("PredictionCallback sample_fn failed: %s", exc)
            return

        rows: list[list[str]] = []
        for item in samples[: self.sample_size]:
            messages = item.get("messages", [])
            if not messages or messages[-1].get("role") != "assistant":
                continue
            prompt_messages, gold = messages[:-1], messages[-1].get("content", "")
            try:
                generated = self.decode_fn(model, tokenizer, prompt_messages, self.max_new_tokens)
            except Exception as exc:
                logger.warning("PredictionCallback decode_fn failed: %s", exc)
                continue
            rows.append([gold, generated])

        if not rows:
            return
        try:
            import wandb

            table = wandb.Table(columns=["gold", "generated"], data=rows)
            wandb.log({"predictions/sample": table}, step=state.global_step)
        except Exception as exc:
            logger.warning("Failed to log prediction sample table: %s", exc)
