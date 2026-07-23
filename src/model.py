"""Base model + LoRA adapter loading via Unsloth.

This project is Unsloth-only: loads via Unsloth's FastModel, which provides
custom kernels/patches for a curated set of architectures — confirmed to
include Gemma 4 (unsloth/gemma-4-12b-it exists on the Hub) — typically
cutting VRAM usage substantially versus a plain transformers+bitsandbytes
path for QLoRA. An earlier version of this project also supported a plain
transformers+peft backend as a fallback; it was removed once Unsloth was
confirmed to be the working path, to keep the implementation to one code
path instead of two.

Neither the fresh-init nor continue-adapter path ever calls
merge_and_unload() — adapters are never merged into the base model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from src import tokenizer as tokenizer_mod
from src import utils
from src.config import ExperimentConfig, LoraConfigSection, ModelConfig

logger = utils.get_logger("model")


@dataclass
class GPUProfile:
    family: str
    recommended_batch_size: int
    recommended_gradient_accumulation: int
    recommended_max_train_samples: Optional[int]
    notes: str


_GPU_PROFILE_TABLE: dict[str, GPUProfile] = {
    "A100": GPUProfile(
        family="A100", recommended_batch_size=4, recommended_gradient_accumulation=4,
        recommended_max_train_samples=None,
        notes="A100 40GB — full-size defaults, no sample cap needed for a 12B model in 4-bit.",
    ),
    "L4": GPUProfile(
        family="L4", recommended_batch_size=2, recommended_gradient_accumulation=8,
        recommended_max_train_samples=30_000,
        notes="L4 24GB — verify VRAM headroom for a 12B model in 4-bit before a full run.",
    ),
    "T4": GPUProfile(
        family="T4", recommended_batch_size=1, recommended_gradient_accumulation=16,
        recommended_max_train_samples=5_000,
        notes="T4 16GB — a 12B model in 4-bit QLoRA may be marginal even at batch_size=1; "
              "reduce max_seq_length first if you hit OOM.",
    ),
    "OTHER": GPUProfile(
        family="OTHER", recommended_batch_size=1, recommended_gradient_accumulation=16,
        recommended_max_train_samples=5_000,
        notes="Unrecognized GPU — using conservative T4-like defaults.",
    ),
    "CPU": GPUProfile(
        family="CPU", recommended_batch_size=1, recommended_gradient_accumulation=1,
        recommended_max_train_samples=50,
        notes="No GPU detected — 4-bit QLoRA training is not practical on CPU; this profile "
              "exists only so code paths don't crash when probed.",
    ),
}


def detect_gpu_profile(override: Optional[str] = None) -> GPUProfile:
    """Return recommended defaults for the active (or overridden) GPU family."""
    if override:
        family = override.upper()
        if family not in _GPU_PROFILE_TABLE:
            raise ValueError(
                f"Unknown hardware.gpu_profile_override: {override!r}. "
                f"Expected one of {sorted(_GPU_PROFILE_TABLE)}."
            )
    else:
        family = utils.detect_gpu().family

    profile = _GPU_PROFILE_TABLE[family]
    logger.info("GPU profile: %s — %s", profile.family, profile.notes)
    return profile


def configure_cuda_visible_devices(gpu_index: int) -> None:
    """Set CUDA_VISIBLE_DEVICES and a fragmentation-reducing allocator config.

    Both must be set before any torch CUDA initialization. PYTORCH_CUDA_ALLOC_CONF
    defaults to expandable_segments:True — exactly what PyTorch's own OOM error
    message recommends when "reserved but unallocated" memory is large relative
    to what's actually needed; setdefault so an explicit user-set value (e.g. in
    the Colab environment already) is never overridden.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_base_model(
    model_config: ModelConfig, max_seq_length: int, hf_token: Optional[str] = None
) -> tuple[Any, Any]:
    """Load the base model + tokenizer together via Unsloth's FastModel.

    Returns (model, tokenizer) — FastModel.from_pretrained configures model
    and tokenizer together (chat template, padding, special tokens), so
    callers MUST use this returned tokenizer for training/inference from
    this point forward, not a separately-loaded one (see notebook Section 7,
    which reassigns its `tokenizer` variable to this return value).
    """
    # NOTE: an earlier version of this function unconditionally set
    # UNSLOTH_COMPILE_DISABLE=1 here to work around a confirmed CUDA
    # illegal-memory-access crash inside dynamo's tracing of Unsloth's
    # compiled Gemma4UnifiedTextAttention/RMSNorm forward. That crash only
    # ever reproduced during evaluate() (training itself completed
    # successfully with compilation enabled beforehand) — disabling
    # compilation globally here was fixing eval stability at the cost of
    # roughly half of Unsloth's advertised training speedup for the entire
    # multi-hour training run, not just the brief eval passes. The fix now
    # lives narrowly in evaluator.evaluate_validation() instead
    # (torch.compiler.set_stance("force_eager"), scoped to just that call),
    # so training here keeps Unsloth's full compiled-kernel speed. See
    # README Troubleshooting for the crash this avoids and why the fix
    # moved.

    # Must run before importing unsloth below: unsloth_zoo's own import chain
    # (via transformers.processing_utils.Unpack -> modeling_utils ->
    # quantizers.auto -> quantizer_torchao) is exactly what triggers
    # transformers' broken torchao availability check (see
    # utils.disable_unused_transformers_backends' docstring) — patching after
    # importing unsloth would be too late.
    utils.disable_unused_transformers_backends()

    try:
        from unsloth import FastModel
    except Exception as exc:
        # Broad except, not just ImportError: unsloth's own import-time code does
        # exec()-based monkeypatching of transformers internals (see
        # unsloth/models/_utils.py), which can raise arbitrary exception types
        # (a bare NameError has been observed in practice) when its patches don't
        # match the installed transformers version — not a clean ImportError.
        raise RuntimeError(
            f"`from unsloth import FastModel` failed: {exc}. This is commonly an "
            "unsloth/transformers version mismatch (unsloth's internal patches don't match "
            "the installed transformers release). Re-run the notebook's Install section — "
            "requirements.txt lists both unsloth and unsloth_zoo unpinned specifically so "
            "`pip install --upgrade` can pick up a compatibility fix — then Runtime -> "
            "Restart session and retry."
        ) from exc

    # FastModel constructs a tokenizer internally, bypassing tokenizer.py's
    # load_tokenizer() — apply the same defensive compat shim directly so
    # this path is equally robust to the transformers v4/v5
    # extra_special_tokens format mismatch (see tokenizer.py).
    tokenizer_mod.patch_extra_special_tokens_list_format()

    logger.info("Loading base model %s via Unsloth FastModel (4-bit QLoRA)...", model_config.base_model)
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_config.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            token=hf_token,
            dtype=None,  # let Unsloth pick the right compute dtype for the detected GPU
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unsloth FastModel.from_pretrained({model_config.base_model!r}) failed: {exc}."
        ) from exc

    # NOTE: tokenizer_mod.patch_chat_template_for_assistant_masking() —
    # which would enable SFTConfig(assistant_only_loss=True) so loss is
    # computed only on assistant response tokens, not the whole conversation
    # — is intentionally NOT called here. That combination was tried and
    # reverted; see trainer.py's module docstring for the confirmed crash
    # (packing=True + assistant_only_loss=True inside Unsloth's compiled
    # SFTTrainer cache) and why packing won the tradeoff. The patch function
    # itself is left in place, tested, and ready to re-enable.

    logger.info("Model + tokenizer loaded via Unsloth.")
    return model, tokenizer


def attach_lora(
    model: Any,
    lora_config: LoraConfigSection,
    continue_adapter: Optional[str],
    gradient_checkpointing: bool = True,
) -> Any:
    """Attach a LoRA adapter via Unsloth: continue an existing one, or initialize fresh.

    Fresh init uses FastModel.get_peft_model with use_gradient_checkpointing=
    "unsloth" when gradient_checkpointing is True — Unsloth's own offloaded-
    checkpointing implementation, their signature memory-saving feature —
    else False. trainer.py's build_sft_config always disables transformers'
    own gradient_checkpointing at the SFTConfig level, since this is the only
    place checkpointing gets configured; enabling both would conflict.
    Continuing a prior adapter reuses plain peft.PeftModel.from_pretrained,
    which Unsloth documents as compatible with its patched base models.
    Never calls merge_and_unload() on either path.

    target_modules: when lora.lora_target_modules is unset, Unsloth's own
    defaults for the detected architecture are used (target_modules=None).
    """
    from peft import PeftModel

    if continue_adapter:
        logger.info("Continuing training from prior adapter (Unsloth base model): %s", continue_adapter)
        peft_model = PeftModel.from_pretrained(model, continue_adapter, is_trainable=True)
    else:
        from unsloth import FastModel

        logger.info("No continue_adapter set — initializing a fresh LoRA adapter via Unsloth.")
        target_modules = list(lora_config.lora_target_modules) if lora_config.lora_target_modules else None
        logger.info(
            "LoRA target_modules: %s",
            target_modules if target_modules else "unsloth defaults for this architecture",
        )
        peft_model = FastModel.get_peft_model(
            model,
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=target_modules,
            bias=lora_config.lora_bias,
            use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
        )

    peft_model.print_trainable_parameters()
    return peft_model


def load_model_with_adapter(config: ExperimentConfig, hf_token: Optional[str]) -> tuple[Any, Any]:
    """Compose load_base_model + attach_lora for callers that want both steps at once.
    Returns (peft_model, tokenizer) — see load_base_model for the tokenizer contract.

    The notebook keeps "Load Model" and "Load Adapter" as separate cells, so
    both this composed function and the two individual steps remain available.
    """
    configure_cuda_visible_devices(config.hardware.training_gpu)
    base_model, tokenizer = load_base_model(config.model, config.data.max_seq_length, hf_token)
    peft_model = attach_lora(
        base_model, config.lora, config.model.continue_adapter, config.training.gradient_checkpointing
    )
    return peft_model, tokenizer
