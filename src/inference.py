"""Inference: post-training reload, prompt building, and generation.

greedy_decode is a verbatim port of the reference notebook's manual per-token
decode loop (section 13) — a documented workaround for a known
Gemma+PEFT+transformers bug where `model.generate()` misbehaves. generate()
tries that path first and falls back to `model.generate()` only if the
manual loop itself raises, logging a warning either way so a decode failure
is never silent.
"""

from __future__ import annotations

from typing import Any, Optional

from src import utils
from src.config import ModelConfig

logger = utils.get_logger("inference")


def load_model_for_inference(
    model_config: ModelConfig, max_seq_length: int, adapter_dir: str, hf_token: Optional[str] = None
) -> tuple[Any, Any]:
    """Reload a saved Unsloth-trained adapter for standalone inference via FastModel.

    Loads the adapter directory directly as model_name — Unsloth auto-detects
    a saved PEFT/LoRA checkpoint and reconstructs base+adapter — then calls
    FastModel.for_inference(), Unsloth's documented pre-generation step that
    enables its fast-inference kernels. Never merges the adapter.
    """
    try:
        from unsloth import FastModel
    except Exception as exc:
        # Broad except, not just ImportError — see model.py's load_base_model
        # for why (unsloth's exec()-based transformers monkeypatching can
        # raise arbitrary exception types).
        raise RuntimeError(
            f"`from unsloth import FastModel` failed: {exc}. This is commonly an "
            "unsloth/transformers version mismatch. Re-run the notebook's Install section, "
            "then Runtime -> Restart session and retry."
        ) from exc

    from src.tokenizer import patch_extra_special_tokens_list_format

    # See model.py's load_base_model for why: FastModel constructs a
    # tokenizer internally, bypassing tokenizer.py's own load_tokenizer().
    patch_extra_special_tokens_list_format()

    logger.info("Reloading Unsloth adapter %s for inference...", adapter_dir)
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=adapter_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            token=hf_token,
            dtype=None,
        )
    except Exception as exc:
        raise RuntimeError(f"Unsloth FastModel.from_pretrained({adapter_dir!r}) failed: {exc}.") from exc

    FastModel.for_inference(model)
    logger.info("Reloaded model + adapter %s for inference via Unsloth.", adapter_dir)
    return model, tokenizer


_PROMPT_SENTINEL = "\x00__T2C_BUILD_PROMPT_SENTINEL__\x00"


def build_prompt(tokenizer: Any, messages: list[dict]) -> str:
    """Format a prompt-only message list ending right where the assistant should continue.

    Deliberately does NOT use apply_chat_template(..., add_generation_prompt=True)
    — confirmed via direct Colab diagnostic (see README Troubleshooting,
    "Fine-tuned model generates garbled/prose text instead of PASS_0-4") that
    for google/gemma-4-12B-it specifically, add_generation_prompt=True appends
    "<|channel>thought\\n<channel|>", opening Gemma 4's native thinking-mode
    channel. Training never demonstrates continuing from that: trainer.train()
    renders full conversations with add_generation_prompt=False, and every
    assistant turn in this dataset is bare PASS_0-4 text immediately following
    the turn header, with no channel wrapper. Feeding that unfamiliar suffix
    at inference produced garbled, prose-like generations mixed with
    fragments of the trained PASS_0-4 structure — confirmed empirically
    (exact_match_rate 1.95%, PASS_3 accuracy 0.4%, the literal word "system"
    spliced into generated text ~5x per example on average).

    Instead, renders the full conversation with add_generation_prompt=False
    (the exact call trainer.train() uses) plus a placeholder final assistant
    turn holding a unique sentinel, then truncates the rendered text right
    before that sentinel — giving a prompt that ends exactly where a real
    training example's assistant turn begins, byte-for-byte, without
    hardcoding this template's special-token spelling (robust to the
    template changing upstream, unlike splicing in a literal "<|turn>model\\n").
    """
    rendered = tokenizer.apply_chat_template(
        list(messages) + [{"role": "assistant", "content": _PROMPT_SENTINEL}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prefix, separator, _ = rendered.partition(_PROMPT_SENTINEL)
    if not separator:
        raise RuntimeError(
            "build_prompt(): sentinel not found in the rendered chat template — this "
            "tokenizer's apply_chat_template may not render assistant content verbatim "
            "(e.g. it escapes/transforms it). Inspect the rendered template directly."
        )
    return prefix


def _infer_device(model: Any) -> Any:
    return next(model.parameters()).device


def greedy_decode(
    model: Any,
    input_ids: Any,
    attention_mask: Optional[Any],
    tokenizer: Any,
    max_new_tokens: int = 512,
    fast: bool = False,
) -> Any:
    """Manual greedy per-token decode loop — verbatim port of notebook section 13.

    Workaround for a known bug where `model.generate()` misbehaves with
    Gemma + PEFT + certain transformers versions. Disables gradient
    checkpointing for the duration of the loop.

    With ``fast=False`` (default) the whole sequence is re-forwarded every step
    with ``use_cache=False``, matching the notebook exactly. That is O(n^2):
    512 new tokens over a ~500-token prompt costs ~380x the compute of cached
    decoding, which is what makes generation-based eval crawl.

    With ``fast=True`` the KV cache is kept and only the newest token is fed
    each step. The emitted tokens are identical — it is the same argmax over
    the same logits — but it re-enables the cache the original workaround
    deliberately avoided, so it stays opt-in via evaluation.fast_decode.
    """
    import torch

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    generated = input_ids
    attn = attention_mask
    eos_id = tokenizer.eos_token_id

    with torch.inference_mode():
        if not fast:
            for _ in range(max_new_tokens):
                outputs = model(input_ids=generated, attention_mask=attn, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if attn is not None:
                    attn = torch.cat([attn, torch.ones_like(next_token, dtype=attn.dtype)], dim=-1)
                if eos_id is not None and int(next_token[0, 0]) == int(eos_id):
                    break
            return generated

        past = None
        step_ids = generated
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=step_ids, attention_mask=attn, use_cache=True, past_key_values=past
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if attn is not None:
                attn = torch.cat([attn, torch.ones_like(next_token, dtype=attn.dtype)], dim=-1)
            if eos_id is not None and int(next_token[0, 0]) == int(eos_id):
                break
            # Only the new token goes in next round; the cache holds the rest.
            step_ids = next_token

    return generated


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    max_new_tokens: int = 512,
    fast: bool = False,
) -> str:
    """Generate a completion for `messages`.

    Tries the manual greedy_decode workaround first; on any exception, logs
    a warning and falls back to model.generate(do_sample=False). `fast` selects
    cached decoding inside greedy_decode (see evaluation.fast_decode).
    """
    prompt = build_prompt(tokenizer, messages)
    device = _infer_device(model)
    # text= must be an explicit keyword, not positional: Gemma 4 is nominally
    # multimodal, so Unsloth/transformers loads `tokenizer` as a
    # Gemma4UnifiedProcessor whose __call__ signature is
    # (self, images=None, text=None, videos=None, audio=None, **kwargs) — a
    # positional tokenizer(prompt, ...) call binds prompt to `images` instead
    # of `text`, and the processor then tries to interpret the entire prompt
    # string as an image URL/path/base64 (confirmed, reproduced: `ValueError:
    # Incorrect image source ... Failed with Incorrect padding`). A plain
    # AutoTokenizer's __call__ also names its first parameter `text`, so this
    # keyword form is correct for both.
    inputs = tokenizer(text=prompt, return_tensors="pt").to(device)

    try:
        out = greedy_decode(
            model,
            inputs["input_ids"],
            inputs.get("attention_mask"),
            tokenizer,
            max_new_tokens=max_new_tokens,
            fast=fast,
        )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    except Exception as exc:
        logger.warning(
            "greedy_decode failed (%s) — falling back to model.generate(do_sample=False).", exc
        )
        import torch

        model.eval()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def run_smoke_test(model: Any, tokenizer: Any, sample_messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate from sample_messages and print the result (notebook section 13 tail)."""
    generated = generate(model, tokenizer, sample_messages, max_new_tokens=max_new_tokens)
    print("=== GENERATED ===")
    print(generated[:1500])
    return generated
