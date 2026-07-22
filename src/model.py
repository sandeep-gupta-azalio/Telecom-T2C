"""Base model + LoRA adapter loading.

Two backends, selected via config.model.backend ("unsloth" | "transformers"):

- "unsloth" (default): loads via Unsloth's FastModel, which provides custom
  kernels/patches for a curated set of architectures — confirmed to include
  Gemma 4 (unsloth/gemma-4-12b-it exists on the Hub) as of this project's
  development — typically cutting VRAM usage substantially versus plain
  transformers+bitsandbytes for QLoRA. NOT validated on real hardware by
  this project (no GPU available during development); start with a small
  data.max_train_samples smoke test before a full run.
- "transformers": the original, proven-working path — preserves the
  reference notebook's 4-bit QLoRA loading order (NF4 double-quant
  BitsAndBytesConfig -> AutoModelForCausalLM -> use_cache off ->
  prepare_model_for_kbit_training) and its PEFT continue-training path
  (PeftModel.from_pretrained(..., is_trainable=True)). Kept as a documented
  fallback in case "unsloth" hits its own environment/compatibility issues.

Neither backend, on either the fresh-init or continue-adapter path, ever
calls merge_and_unload() — adapters are never merged into the base model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from src import utils
from src.config import ExperimentConfig, LoraConfigSection, ModelConfig

logger = utils.get_logger("model")

_DTYPE_NAMES = ("bfloat16", "float16", "float32")


@dataclass
class GPUProfile:
    family: str
    recommended_batch_size: int
    recommended_gradient_accumulation: int
    recommended_max_train_samples: Optional[int]
    attn_implementation: str
    notes: str


_GPU_PROFILE_TABLE: dict[str, GPUProfile] = {
    "A100": GPUProfile(
        family="A100", recommended_batch_size=4, recommended_gradient_accumulation=4,
        recommended_max_train_samples=None, attn_implementation="sdpa",
        notes="A100 40GB — full-size defaults, no sample cap needed for a 12B model in 4-bit.",
    ),
    "L4": GPUProfile(
        family="L4", recommended_batch_size=2, recommended_gradient_accumulation=8,
        recommended_max_train_samples=30_000, attn_implementation="sdpa",
        notes="L4 24GB — verify VRAM headroom for a 12B model in 4-bit before a full run.",
    ),
    "T4": GPUProfile(
        family="T4", recommended_batch_size=1, recommended_gradient_accumulation=16,
        recommended_max_train_samples=5_000, attn_implementation="sdpa",
        notes="T4 16GB — a 12B model in 4-bit QLoRA may be marginal even at batch_size=1; "
              "reduce max_seq_length first if you hit OOM.",
    ),
    "OTHER": GPUProfile(
        family="OTHER", recommended_batch_size=1, recommended_gradient_accumulation=16,
        recommended_max_train_samples=5_000, attn_implementation="sdpa",
        notes="Unrecognized GPU — using conservative T4-like defaults.",
    ),
    "CPU": GPUProfile(
        family="CPU", recommended_batch_size=1, recommended_gradient_accumulation=1,
        recommended_max_train_samples=50, attn_implementation="sdpa",
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


def resolve_torch_dtype(name: str) -> Any:
    """Map a config string ("bfloat16"/"float16"/"float32") to a torch.dtype.

    Public (not a leading-underscore helper) because inference.py's reload
    path needs the exact same mapping.
    """
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        logger.warning("Unknown model.torch_dtype=%r — defaulting to bfloat16.", name)
    return mapping.get(name, torch.bfloat16)


def build_bnb_config(compute_dtype: Any) -> Any:
    """NF4 double-quant 4-bit config — verbatim from the reference notebook (section 7)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def resolve_attn_implementation(preference: str) -> str:
    """Resolve "auto" to flash_attention_2 if available, else the notebook's proven "sdpa"."""
    if preference in ("flash_attention_2", "sdpa"):
        return preference
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


def load_base_model(model_config: ModelConfig, hf_token: Optional[str], device_index: int = 0) -> Any:
    """Load the quantized base model.

    Tries AutoModelForCausalLM first (the reference notebook's proven path
    for Gemma 3, also nominally multimodal). On an unrecognized-architecture
    failure, attempts AutoModelForImageTextToText as a guarded fallback
    (Gemma 4 is a genuinely new, multimodal-native architecture whose exact
    text-only loading class is unverified — see README "Troubleshooting").
    Raises an actionable RuntimeError, not a bare exception, if both fail.
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU available. In Colab: Runtime -> Change runtime type -> select a GPU "
            "(A100 recommended for this 12B model)."
        )

    compute_dtype = resolve_torch_dtype(model_config.torch_dtype)
    bnb_config = build_bnb_config(compute_dtype)
    attn_impl = resolve_attn_implementation(model_config.attn_implementation)
    device_map = {"": device_index}

    logger.info("Loading base model %s (4-bit QLoRA, attn=%s)...", model_config.base_model, attn_impl)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model,
            token=hf_token,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=compute_dtype,
            attn_implementation=attn_impl,
        )
    except (ValueError, KeyError) as exc:
        logger.warning(
            "AutoModelForCausalLM.from_pretrained(%s) failed (%s) — retrying via "
            "AutoModelForImageTextToText.", model_config.base_model, exc,
        )
        import transformers

        fallback_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if fallback_cls is None:
            raise RuntimeError(
                f"Could not load {model_config.base_model}: unrecognized architecture with the "
                "installed transformers version, and no AutoModelForImageTextToText fallback class "
                "is available. This model may require a newer transformers release — try: "
                "pip install -U transformers"
            ) from exc
        try:
            model = fallback_cls.from_pretrained(
                model_config.base_model,
                token=hf_token,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=compute_dtype,
                attn_implementation=attn_impl,
            )
        except Exception as exc2:
            raise RuntimeError(
                f"Could not load {model_config.base_model} via AutoModelForCausalLM or "
                f"AutoModelForImageTextToText: {exc2}. This model may require a newer transformers "
                "release — try: pip install -U transformers"
            ) from exc2

    model.config.use_cache = False
    logger.info("Model loaded.")
    return model


def resolve_target_modules(model: Any, override: Optional[list[str]]) -> list[str]:
    """Resolve LoRA target_modules: explicit override, else auto-detect nn.Linear leaf names.

    The reference notebook never exercised this path (it only ever continued
    existing adapters, whose target_modules already live in adapter_config.json).
    Auto-detection excludes lm_head/embedding layers, matching the common
    PEFT "all-linear" convention.
    """
    if override:
        return list(override)

    import torch.nn as nn

    excluded_substrings = ("lm_head", "embed_tokens", "embed_positions", "embed_out")
    names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.rsplit(".", 1)[-1]
            if any(sub in leaf for sub in excluded_substrings):
                continue
            names.add(leaf)

    if not names:
        raise RuntimeError(
            "resolve_target_modules found no nn.Linear layers to target — check the model "
            "architecture, or set lora.lora_target_modules explicitly in configs/experiment.yaml."
        )
    return sorted(names)


# Non-reentrant checkpointing recomputes with a different autograd graph
# structure than the (older) reentrant default, generally holding fewer
# saved tensors at once — HF's own recommended setting for QLoRA, and one
# real lever against CUDA OOM during backward(). trainer.py's SFTConfig
# uses this exact same dict so the Trainer-level setting agrees with what
# prepare_model_for_kbit_training already configured below.
GRADIENT_CHECKPOINTING_KWARGS: dict[str, bool] = {"use_reentrant": False}


def attach_lora(model: Any, lora_config: LoraConfigSection, continue_adapter: Optional[str]) -> Any:
    """Attach a LoRA adapter: continue an existing one, or initialize fresh.

    Always: disable KV cache and run prepare_model_for_kbit_training with
    non-reentrant gradient checkpointing (matches reference notebook section
    8's intent, but lets peft's own kbit-training prep own the checkpointing
    setup — including the enable_input_require_grads() dance LoRA needs on a
    quantized model — rather than a separate, redundant
    model.gradient_checkpointing_enable() call beforehand). Never calls
    merge_and_unload() on either path.
    """
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs=GRADIENT_CHECKPOINTING_KWARGS,
    )

    if continue_adapter:
        logger.info("Continuing training from prior adapter: %s", continue_adapter)
        peft_model = PeftModel.from_pretrained(model, continue_adapter, is_trainable=True)
    else:
        logger.info("No continue_adapter set — initializing a fresh LoRA adapter.")
        target_modules = resolve_target_modules(model, lora_config.lora_target_modules)
        logger.info("LoRA target_modules: %s", target_modules)
        peft_config = LoraConfig(
            r=lora_config.lora_r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=target_modules,
            bias=lora_config.lora_bias,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, peft_config)

    peft_model.print_trainable_parameters()
    return peft_model


def load_base_model_unsloth(
    model_config: ModelConfig, max_seq_length: int, hf_token: Optional[str] = None
) -> tuple[Any, Any]:
    """Load the base model + tokenizer together via Unsloth's FastModel.

    Returns (model, tokenizer) — unlike load_base_model(), the tokenizer
    comes back from this call because FastModel.from_pretrained configures
    model and tokenizer together (chat template, padding, special tokens).
    Callers MUST use this returned tokenizer for training/inference from
    this point forward, not a separately-loaded one (see notebook Section 7,
    which reassigns its `tokenizer` variable to this return value).
    """
    try:
        from unsloth import FastModel
    except Exception as exc:
        # Broad except, not just ImportError: unsloth's own import-time code does
        # exec()-based monkeypatching of transformers internals (see
        # unsloth/models/_utils.py), which can raise arbitrary exception types
        # (a bare NameError has been observed in practice) when its patches don't
        # match the installed transformers version — not a clean ImportError.
        raise RuntimeError(
            f"model.backend='unsloth' but `from unsloth import FastModel` failed: {exc}. This is "
            "commonly an unsloth/transformers version mismatch (unsloth's internal patches don't "
            "match the installed transformers release), not a missing-package issue. Set "
            "model.backend='transformers' in configs/experiment.yaml to fall back to the "
            "proven-working plain transformers+peft path — no other code changes needed."
        ) from exc

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
            f"Unsloth FastModel.from_pretrained({model_config.base_model!r}) failed: {exc}. "
            "If unsloth doesn't (yet) support this exact model/architecture, set "
            "model.backend='transformers' in configs/experiment.yaml to fall back to the "
            "plain transformers+peft path."
        ) from exc

    logger.info("Model + tokenizer loaded via Unsloth.")
    return model, tokenizer


def attach_lora_unsloth(model: Any, lora_config: LoraConfigSection, continue_adapter: Optional[str]) -> Any:
    """Attach a LoRA adapter via Unsloth: continue an existing one, or initialize fresh.

    Fresh init uses FastModel.get_peft_model with use_gradient_checkpointing=
    "unsloth" — Unsloth's own offloaded-checkpointing implementation, their
    signature memory-saving feature. trainer.py's build_sft_config knows not
    to also enable transformers' own gradient_checkpointing on the SFTConfig
    side when this backend is active, to avoid double-configuring it.
    Continuing a prior adapter reuses plain peft.PeftModel.from_pretrained,
    which Unsloth documents as compatible with its patched base models.
    Never calls merge_and_unload() on either path.

    target_modules: when lora.lora_target_modules is unset, this intentionally
    does NOT call resolve_target_modules() (that function's generic nn.Linear
    detection targets plain transformers models) — Unsloth's own defaults for
    its patched architecture are used instead by passing target_modules=None.
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
            use_gradient_checkpointing="unsloth",
        )

    peft_model.print_trainable_parameters()
    return peft_model


def load_base_model_for_backend(
    config: ExperimentConfig, hf_token: Optional[str]
) -> tuple[Any, Optional[Any]]:
    """Load the base model using config.model.backend. Returns (model, tokenizer_or_None).

    tokenizer is only non-None for the "unsloth" backend — callers on the
    "transformers" backend keep using the tokenizer they already loaded via
    tokenizer.load_tokenizer().
    """
    configure_cuda_visible_devices(config.hardware.training_gpu)
    backend = config.model.backend
    if backend == "unsloth":
        return load_base_model_unsloth(config.model, config.data.max_seq_length, hf_token)
    if backend == "transformers":
        model = load_base_model(config.model, hf_token, device_index=config.hardware.training_gpu)
        return model, None
    raise ValueError(f"Unknown model.backend: {backend!r}. Expected 'unsloth' or 'transformers'.")


def attach_lora_for_backend(config: ExperimentConfig, model: Any) -> Any:
    """Attach a LoRA adapter using config.model.backend."""
    backend = config.model.backend
    if backend == "unsloth":
        return attach_lora_unsloth(model, config.lora, config.model.continue_adapter)
    if backend == "transformers":
        return attach_lora(model, config.lora, config.model.continue_adapter)
    raise ValueError(f"Unknown model.backend: {backend!r}. Expected 'unsloth' or 'transformers'.")


def load_model_with_adapter(config: ExperimentConfig, hf_token: Optional[str]) -> tuple[Any, Optional[Any]]:
    """Compose load_base_model_for_backend + attach_lora_for_backend for callers that want
    both steps at once. Returns (peft_model, tokenizer_or_None) — see
    load_base_model_for_backend for the tokenizer contract.

    The notebook keeps "Load Model" and "Load Adapter" as separate cells, so
    both this composed function and the two individual dispatchers remain
    available.
    """
    base_model, tokenizer = load_base_model_for_backend(config, hf_token)
    peft_model = attach_lora_for_backend(config, base_model)
    return peft_model, tokenizer
