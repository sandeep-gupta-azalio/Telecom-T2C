"""TRL SFTTrainer orchestration.

Maps ExperimentConfig onto SFTConfig field-for-field as the reference
notebook uses them (section 10), substituting the notebook's manual
steps_per_epoch/warmup_steps precomputation with SFTConfig's native
warmup_ratio support — that estimate now lives in statistics.py instead, so
it isn't duplicated. train_ds/eval_ds keep their native "messages" column
(no reformatting) and are passed to SFTTrainer as-is — TRL auto-detects the
conversational format and applies the chat template itself, which is what
lets assistant_only_loss=True (masking the loss to just the assistant's own
response tokens) work at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src import checkpoint, inference, utils
from src.callbacks import EvaluationCallback, GPUCallback, PredictionCallback, TrainingCallback
from src.config import ExperimentConfig
from src.wandb_logger import WandbLogger

logger = utils.get_logger("trainer")


def build_sft_config(config: ExperimentConfig, run_dir: Path, eval_available: bool) -> Any:
    """Build an SFTConfig from ExperimentConfig, matching notebook section 10's field usage."""
    import torch
    from trl import SFTConfig

    adapter_dir = run_dir / "adapter"

    # model.attach_lora() already configures gradient checkpointing at the
    # model level, via FastModel.get_peft_model(use_gradient_checkpointing=
    # "unsloth" or False, driven by this same config.training.gradient_checkpointing
    # flag) — Unsloth's own offloaded-checkpointing implementation. Also
    # enabling transformers' generic gradient_checkpointing here would make
    # Trainer try to re-configure checkpointing on top of Unsloth's own
    # setup, which conflicts — so it's always off at the SFTConfig level.
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
        packing=config.training.packing,
        # "wrapped" instead of SFTConfig's default "bfd" packing strategy:
        # TRL unconditionally forces padding_free=True whenever
        # packing=True and packing_strategy=="bfd" (confirmed directly in
        # trl/trainer/sft_trainer.py: `self.padding_free = args.padding_free
        # or (args.packing and args.packing_strategy == "bfd")`), and
        # padding_free requires FlashAttention 2/3 to actually work — but
        # this project runs on Unsloth's own xformers-based attention
        # kernels (confirmed via the Section 7 load banner: "FA2 = False"),
        # not FA2. That mismatch surfaced as `ValueError: When
        # padding_free=True without packing, max_length is not enforced...`
        # once assistant_only_loss's conversational dataset path was wired
        # in. "wrapped" packing doesn't have this auto-enable behavior — the
        # tradeoff is that it can occasionally cut an example across a
        # pack boundary (vs. bfd's more careful bin-packing), a minor
        # quality cost given most conversations here are well under
        # max_seq_length, versus a hard crash.
        packing_strategy="wrapped",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        seed=config.identity.seed,
        # train_ds/eval_ds keep their native "messages" column (see train()
        # below — no more pre-flattening to a "text" field), so SFTTrainer
        # auto-detects the conversational format and applies the chat
        # template itself. That's required for assistant_only_loss=True:
        # it masks the loss to only the assistant's own response tokens,
        # instead of the entire conversation (which is mostly repeated
        # system-prompt/deployment-context boilerplate — see
        # tokenizer.patch_chat_template_for_assistant_masking, applied to
        # the tokenizer in model.load_base_model()).
        assistant_only_loss=True,
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

    # train_ds/eval_ds keep their native "messages" column — passed straight
    # to SFTTrainer rather than pre-flattened to a "text" field, so TRL's own
    # conversational-format auto-detection (is_conversational) applies the
    # chat template itself and can build the assistant_masks that
    # assistant_only_loss=True (set in build_sft_config above) needs.
    eval_dataset = eval_ds if eval_available else None

    def _sample_fn() -> list[dict]:
        n = min(8, len(train_ds))
        return [train_ds[i] for i in range(n)]

    callbacks = build_callbacks(
        config, wandb_logger, run_metadata,
        sample_fn=_sample_fn, decode_fn=inference.generate, eval_available=eval_available,
    )

    # `tokenizer` here is actually Unsloth's returned Gemma4UnifiedProcessor
    # (Gemma 4 is nominally multimodal — see inference.py's docstring for the
    # same fact biting a different bug). Passing a `ProcessorMixin` as
    # `processing_class` makes TRL's SFTTrainer set `self._is_vlm = True`
    # (isinstance-based, unconditional — see trl/trainer/sft_trainer.py),
    # which hard-blocks BOTH `packing` and `assistant_only_loss` with a
    # `ValueError` regardless of whether the dataset is actually
    # multimodal — confirmed by reading TRL's source directly. This project
    # never trains on images/audio/video, so passing the *inner* plain
    # tokenizer instead (TRL's own code does `processing_class.tokenizer`
    # internally for exactly this reason — every ProcessorMixin has one)
    # avoids VLM mode entirely, letting both packing and assistant_only_loss
    # work as intended. tokenizer.patch_chat_template_for_assistant_masking()
    # already patches this inner tokenizer's chat_template too (see its
    # docstring), so the chat template stays consistent either way.
    processing_class = getattr(tokenizer, "tokenizer", tokenizer)

    sft_trainer = SFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
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
