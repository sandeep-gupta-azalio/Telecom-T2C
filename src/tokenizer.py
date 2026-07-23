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


_GENERATION_MARKER_ANCHOR = "{{- captured_content -}}"
_GENERATION_MARKER_REPLACEMENT = (
    "{%- if role == 'model' -%}"
    "{% generation %}{{- captured_content -}}{% endgeneration %}"
    "{%- else -%}"
    "{{- captured_content -}}"
    "{%- endif -%}"
)


def patch_chat_template_for_assistant_masking(tokenizer: Any) -> None:
    """Insert `{% generation %}`/`{% endgeneration %}` markers around the
    assistant-content span in google/gemma-4-12B-it's chat template.

    Why this is needed: TRL's `SFTConfig.assistant_only_loss=True` (masking
    the training loss to only the assistant's own response tokens, instead
    of the entire conversation including the repeated system-prompt/
    deployment-context boilerplate that precedes every turn) depends on
    `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)`,
    which itself depends on the chat template containing a `{% generation %}`
    block around assistant content. Confirmed directly (downloaded the real
    tokenizer/template and tested locally): google/gemma-4-12B-it's own
    `chat_template.jinja` does not have this marker at all — transformers
    logs `return_assistant_tokens_mask==True but chat template does not
    contain '{% generation %}' keyword` and silently returns an all-zero
    mask, which TRL then turns into a hard `RuntimeError` the moment
    `assistant_only_loss=True` is set ("at least one example has no
    assistant tokens... missing the `{% generation %}` keyword").

    This patches the template's `{{- captured_content -}}` line (the exact
    span that renders an assistant/model turn's actual text, excluding the
    surrounding `<|turn>model` header and `<turn|>` closing token) to wrap
    it in the marker only when `role == 'model'`, leaving every other role
    (`system`/`user`) and the overall rendered *text* output completely
    unchanged — verified locally (byte-identical rendering, correct
    per-turn assistant token spans decoding back to exactly the PASS_0-4
    content) before this was wired into training.

    Raises RuntimeError if the exact anchor text isn't found — e.g. if
    Google revises the template's structure upstream — rather than silently
    no-op'ing and leaving assistant_only_loss broken without any signal.
    Idempotent: no-ops if the template already has `{% generation %}`
    (covers a future template that ships this natively).
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return
    if "{% generation %}" in template:
        return

    if _GENERATION_MARKER_ANCHOR not in template:
        raise RuntimeError(
            "Could not find the expected assistant-content anchor "
            f"({_GENERATION_MARKER_ANCHOR!r}) in the tokenizer's chat_template — "
            "google/gemma-4-12B-it's chat_template.jinja structure may have "
            "changed upstream. patch_chat_template_for_assistant_masking() needs "
            "updating to match the new template before assistant_only_loss=True "
            "can work; until then, either fix this patch or set "
            "training.assistant_only_loss: false (if exposed) / remove "
            "assistant_only_loss=True from trainer.build_sft_config()."
        )

    tokenizer.chat_template = template.replace(_GENERATION_MARKER_ANCHOR, _GENERATION_MARKER_REPLACEMENT, 1)
    logger.info("Patched chat template with generation markers for assistant-only-loss masking.")


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
