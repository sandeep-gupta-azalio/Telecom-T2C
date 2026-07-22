"""TRL SFTTrainer orchestration.

Maps ExperimentConfig onto SFTConfig field-for-field as the reference
notebook uses them (section 10), substituting the notebook's manual
steps_per_epoch/warmup_steps precomputation with SFTConfig's native
warmup_ratio support — that estimate now lives in statistics.py instead, so
it isn't duplicated. Formats the dataset via tokenizer.apply_chat_template on
the raw "messages" column (no reformatting), matching the project's
resolved dataset-handling decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src import checkpoint, inference, utils
from src.callbacks import EvaluationCallback, GPUCallback, PredictionCallback, TrainingCallback
from src.config import ExperimentConfig
from src.model import GRADIENT_CHECKPOINTING_KWARGS
from src.wandb_logger import WandbLogger

logger = utils.get_logger("trainer")


def build_sft_config(config: ExperimentConfig, run_dir: Path, eval_available: bool) -> Any:
    """Build an SFTConfig from ExperimentConfig, matching notebook section 10's field usage."""
    import torch
    from trl import SFTConfig

    adapter_dir = run_dir / "adapter"

    # On the "unsloth" backend, attach_lora_unsloth() already configured
    # gradient checkpointing at the model level via
    # FastModel.get_peft_model(..., use_gradient_checkpointing="unsloth") —
    # Unsloth's own offloaded-checkpointing implementation. Also enabling
    # transformers' generic gradient_checkpointing here would make Trainer
    # try to re-configure checkpointing with different (HF-style, non-
    # reentrant) kwargs on top of Unsloth's own setup, which Unsloth's own
    # guidance says to avoid. On the "transformers" backend, this SFTConfig
    # setting is what actually enables/re-affirms the non-reentrant
    # checkpointing attach_lora() configured via prepare_model_for_kbit_training.
    use_hf_gradient_checkpointing = (
        config.training.gradient_checkpointing and config.model.backend != "unsloth"
    )

    kwargs: dict[str, Any] = dict(
        output_dir=str(adapter_dir),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim=config.training.optim,
        lr_scheduler_type=config.training.lr_scheduler_type,
        # wandb is driven through WandbLogger/callbacks, not Trainer's built-in
        # integration, so it stays no-op-safe even when wandb is unavailable.
        report_to="none",
        max_length=config.data.max_seq_length,
        dataset_text_field="text",
        packing=config.training.packing,
        gradient_checkpointing=use_hf_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            GRADIENT_CHECKPOINTING_KWARGS if use_hf_gradient_checkpointing else None
        ),
        seed=config.identity.seed,
    )

    if eval_available:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = config.training.eval_steps
    else:
        kwargs["eval_strategy"] = "no"

    if config.evaluation.early_stopping and eval_available:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = config.evaluation.metric_for_best_model
        kwargs["greater_is_better"] = config.evaluation.greater_is_better

    return SFTConfig(**kwargs)


def build_callbacks(
    config: ExperimentConfig,
    wandb_logger: WandbLogger,
    run_metadata: dict[str, Any],
    sample_fn: Any,
    decode_fn: Any,
    eval_available: bool,
) -> list[Any]:
    """Assemble the callback list: Training/GPU always, Evaluation/Prediction/EarlyStopping when eval is on."""
    callbacks: list[Any] = [
        TrainingCallback(wandb_logger, run_metadata),
        GPUCallback(wandb_logger, log_every_n_steps=max(config.training.logging_steps, 1)),
    ]

    if eval_available:
        callbacks.append(EvaluationCallback(wandb_logger))
        callbacks.append(
            PredictionCallback(
                sample_fn=sample_fn,
                decode_fn=decode_fn,
                wandb_logger=wandb_logger,
                max_new_tokens=min(config.evaluation.max_new_tokens_eval, 256),
            )
        )
        if config.evaluation.early_stopping:
            from transformers import EarlyStoppingCallback

            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.evaluation.early_stopping_patience,
                    early_stopping_threshold=config.evaluation.early_stopping_threshold,
                )
            )
    return callbacks


def _make_formatting_fn(tokenizer: Any):
    def _format(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    return _format


def train(
    config: ExperimentConfig,
    peft_model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Optional[Any],
    run_dir: Path,
    wandb_logger: WandbLogger,
    run_metadata: dict[str, Any],
) -> Any:
    """Run SFTTrainer training, with resume-from-checkpoint support.

    Returns the SFTTrainer instance (caller uses it for trainer.evaluate()
    and save_best_model()).
    """
    from trl import SFTTrainer

    adapter_dir = utils.ensure_dir(run_dir / "adapter")
    eval_available = eval_ds is not None and config.evaluation.run_eval

    sft_config = build_sft_config(config, run_dir, eval_available)

    format_fn = _make_formatting_fn(tokenizer)
    formatted_train = train_ds.map(format_fn, remove_columns=train_ds.column_names)
    formatted_eval = eval_ds.map(format_fn, remove_columns=eval_ds.column_names) if eval_available else None

    def _sample_fn() -> list[dict]:
        n = min(8, len(train_ds))
        return [train_ds[i] for i in range(n)]

    callbacks = build_callbacks(
        config, wandb_logger, run_metadata,
        sample_fn=_sample_fn, decode_fn=inference.generate, eval_available=eval_available,
    )

    sft_trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=formatted_train,
        eval_dataset=formatted_eval,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume_path = checkpoint.resolve_resume_path(adapter_dir, config.reproducibility.resume_training)
    logger.info("Starting training (resume_from_checkpoint=%s)...", resume_path)
    sft_trainer.train(resume_from_checkpoint=resume_path)
    logger.info("Training finished.")
    return sft_trainer


def save_best_model(trainer: Any, adapter_dir: Path, tokenizer: Any) -> None:
    """Save the final adapter + tokenizer, then prune old checkpoints (keep best + latest).

    HF Trainer's own save_total_limit already protects the checkpoint
    tracked as "best" (when load_best_model_at_end is set) from deletion;
    this is a defensive extra cleanup pass on top of that.
    """
    utils.ensure_dir(adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    keep: set[Path] = set()
    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_checkpoint:
        keep.add(Path(best_checkpoint))
    latest_checkpoint = checkpoint.find_latest_checkpoint(adapter_dir)
    if latest_checkpoint:
        keep.add(latest_checkpoint)
    checkpoint.prune_checkpoints(adapter_dir, keep)

    logger.info("Saved adapter + tokenizer to %s", adapter_dir)
