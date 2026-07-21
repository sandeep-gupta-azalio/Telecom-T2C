"""Base model + LoRA adapter loading.

Preserves the reference notebook's proven-working 4-bit QLoRA loading order
(NF4 double-quant BitsAndBytesConfig -> AutoModelForCausalLM -> use_cache off
-> gradient_checkpointing_enable -> prepare_model_for_kbit_training) and its
PEFT continue-training path (PeftModel.from_pretrained(..., is_trainable=True)).

The reference notebook never actually builds a fresh peft.LoraConfig (it only
ever continues existing adapters) — the fresh-init path here (attach_lora
when continue_adapter is unset) is new, since this project's default run has
no prior adapter to continue from. Neither path ever calls
merge_and_unload() — adapters are never merged into the base model.
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
    """Set CUDA_VISIBLE_DEVICES. Call this before any torch CUDA initialization."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)


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


def attach_lora(model: Any, lora_config: LoraConfigSection, continue_adapter: Optional[str]) -> Any:
    """Attach a LoRA adapter: continue an existing one, or initialize fresh.

    Always: disable KV cache, enable gradient checkpointing, and run
    prepare_model_for_kbit_training (matches reference notebook section 8's
    ordering). Never calls merge_and_unload() on either path.
    """
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

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


def load_model_with_adapter(config: ExperimentConfig, hf_token: Optional[str]) -> Any:
    """Compose load_base_model + attach_lora for callers that want both steps at once.

    The notebook keeps "Load Model" and "Load Adapter" as separate cells, so
    both the composed function and the two individual steps remain available.
    """
    configure_cuda_visible_devices(config.hardware.training_gpu)
    base_model = load_base_model(config.model, hf_token, device_index=config.hardware.training_gpu)
    return attach_lora(base_model, config.lora, config.model.continue_adapter)
