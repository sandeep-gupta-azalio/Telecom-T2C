"""Tokenizer loading and HF token resolution.

Mirrors the reference notebook's working tokenizer setup (section 7) with one
addition: a fallback to AutoProcessor when AutoTokenizer fails to load,
since google/gemma-4-12B-it is a nominally multimodal checkpoint and some
multimodal HF repos only ship a processor rather than a standalone tokenizer.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src import utils

logger = utils.get_logger("tokenizer")

_PATCH_MARKER = "_t2c_extra_special_tokens_patch"


def patch_extra_special_tokens_list_format() -> None:
    """Defensive compat shim for a confirmed transformers v4-vs-v5 incompatibility.

    google/gemma-4-12B-it's tokenizer_config.json defines "extra_special_tokens"
    as a list (transformers v5's format). transformers v4.x's
    PreTrainedTokenizerBase._set_model_specific_special_tokens unconditionally
    calls `.keys()` on that field (a v4-only, dict-shaped assumption), raising
    `AttributeError: 'list' object has no attribute 'keys'` — confirmed via
    https://github.com/huggingface/transformers/issues/45376 and
    https://huggingface.co/google/gemma-4-E4B-it/discussions/17. Since
    requirements.txt intentionally leaves transformers unpinned-above (Gemma 4
    itself requires v5 for this reason), this should be moot in practice — but
    patches defensively anyway so tokenizer loading stays robust even if a
    future resolve somehow lands on an older transformers again.

    Safe by construction: only intervenes when calling the *original* method
    raises exactly this AttributeError (i.e. only on the buggy v4.x
    implementation). Transformers v5's own (correct) list handling is never
    touched — the original method is always tried first and used as-is
    whenever it doesn't raise. Idempotent — safe to call multiple times.
    """
    utils.disable_unused_transformers_backends()

    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        return

    current = PreTrainedTokenizerBase._set_model_specific_special_tokens
    if getattr(current, _PATCH_MARKER, False):
        return  # already patched

    original = current

    def _patched(self, special_tokens):  # type: ignore[no-untyped-def]
        try:
            return original(self, special_tokens)
        except AttributeError:
            if isinstance(special_tokens, list):
                logger.info(
                    "Patched around a transformers v4/v5 extra_special_tokens format "
                    "mismatch (list -> dict) — see tokenizer.py's "
                    "patch_extra_special_tokens_list_format docstring."
                )
                return original(self, {str(token): token for token in special_tokens})
            raise

    setattr(_patched, _PATCH_MARKER, True)
    PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched


def resolve_hf_token(env_var_name: str = "HF_TOKEN") -> Optional[str]:
    """Resolve a Hugging Face token from the environment, then Colab secrets.

    Mirrors notebook section 4. Returns None (not an error) if no token is
    found — public models download anonymously, and a private/gated model
    will simply fail later with a clear HF-side error if a token was needed.
    """
    token = os.environ.get(env_var_name)
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        token = userdata.get(env_var_name)
        if token:
            return token
    except Exception:
        pass
    logger.info("No %s found in environment or Colab secrets — downloading anonymously.", env_var_name)
    return None


def load_tokenizer(model_id: str, hf_token: Optional[str] = None) -> Any:
    """Load a tokenizer for model_id, with a processor-based fallback.

    Tries AutoTokenizer.from_pretrained first (the reference notebook's
    proven-working path for Gemma 3). On failure — some multimodal repos only
    ship an AutoProcessor — falls back to AutoProcessor.from_pretrained(...).tokenizer.
    Sets pad_token=eos_token if unset and padding_side="right", matching the
    notebook exactly.
    """
    # Must run before the transformers import below: AutoProcessor's lazy
    # submodule load is exactly what triggers transformers.quantizers.auto's
    # broken torchao import chain (see disable_unused_transformers_backends'
    # docstring) — patching after importing AutoProcessor would be too late.
    utils.disable_unused_transformers_backends()

    from transformers import AutoProcessor, AutoTokenizer

    patch_extra_special_tokens_list_format()

    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    except (ValueError, OSError, KeyError) as exc:
        logger.warning(
            "AutoTokenizer.from_pretrained(%s) failed (%s) — retrying via AutoProcessor.",
            model_id, exc,
        )
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        tok = processor.tokenizer

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def get_special_tokens_info(tokenizer: Any) -> dict[str, Any]:
    """Summarize special-token configuration for logging/manifest purposes."""
    return {
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
