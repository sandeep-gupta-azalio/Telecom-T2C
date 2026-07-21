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
    from transformers import AutoProcessor, AutoTokenizer

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
