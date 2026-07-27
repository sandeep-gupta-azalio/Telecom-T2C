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


_STOP_TOKEN_PROBE_SENTINEL = "\x00__T2C_STOP_TOKEN_PROBE__\x00"


def _resolve_stop_token_ids(tokenizer: Any) -> set[int]:
    """Discover this tokenizer's real "turn complete" stop token(s), not just eos_token_id.

    Confirmed directly against google/gemma-4-12B-it's tokenizer_config.json:
    this template's actual turn-closing marker is a SEPARATE token from
    eos_token ("eot_token": "<turn|>" vs "eos_token": "<eos>"). The model
    correctly learns to predict the turn-closer (that's what training
    targets — see build_prompt()'s docstring), but a decode loop that only
    checks eos_token_id never recognizes it as a stop signal, so generation
    ran to max_new_tokens on every single example even after producing a
    fully correct PASS_0-4 answer — confirmed empirically (generated text
    ~2x gold's length on median, 12% of a real 256-example eval run
    degenerating into a repeated garbage tail after the correct answer).

    Rather than hardcode the literal "<turn|>" (fragile if the template
    changes upstream, and not necessarily correct for any other model this
    project might target later), this discovers it the same way
    build_prompt() discovers the prompt prefix: render a short conversation
    ending in a real assistant turn with add_generation_prompt=False (the
    exact call trainer.train() uses) via a unique sentinel, then look at
    what immediately follows that sentinel in the rendered text — that
    suffix is the template's own turn-closing sequence, tokenized to get
    its real id directly from this tokenizer's own vocabulary. Only added
    as a stop id when it resolves to exactly one token; a multi-token
    closing sequence isn't safe to stop on after a single argmax step, so
    that case logs a warning and falls back to eos_token_id alone rather
    than silently doing something wrong.

    `tokenizer` here is often actually Unsloth's Gemma4UnifiedProcessor
    (Gemma 4 is nominally multimodal — see load_model_for_inference's
    docstring for the same fact biting other bugs). apply_chat_template is
    exposed on the processor directly, but `.encode()` is not — only its
    *inner* `.tokenizer` has it (confirmed: a real run raised
    `'Gemma4UnifiedProcessor' object has no attribute 'encode'`, silently
    degrading discovery to eos_token_id-only every time) — so encode() is
    called on getattr(tokenizer, "tokenizer", tokenizer), matching the same
    processor/tokenizer duality trainer.py already handles.
    """
    stop_ids: set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    try:
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "probe"},
                {"role": "assistant", "content": _STOP_TOKEN_PROBE_SENTINEL},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        after_sentinel = rendered.split(_STOP_TOKEN_PROBE_SENTINEL, 1)[1].strip()
        if after_sentinel:
            encoder = getattr(tokenizer, "tokenizer", tokenizer)
            closing_ids = encoder.encode(after_sentinel, add_special_tokens=False)
            if len(closing_ids) == 1:
                stop_ids.add(int(closing_ids[0]))
            else:
                logger.warning(
                    "Turn-closing sequence %r tokenizes to %d tokens %r, not 1 — not adding "
                    "it as a stop token (falling back to eos_token_id only, which may "
                    "cause over-generation).",
                    after_sentinel, len(closing_ids), closing_ids,
                )
    except Exception as exc:
        logger.warning(
            "Could not discover an additional turn-closing stop token (%s) — using "
            "eos_token_id only, which may cause over-generation.", exc,
        )

    return stop_ids


def greedy_decode(
    model: Any,
    input_ids: Any,
    attention_mask: Optional[Any],
    tokenizer: Any,
    max_new_tokens: int = 512,
    fast: bool = False,
    stop_token_ids: Optional[set] = None,
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

    ``stop_token_ids``, if given, is unioned with tokenizer.eos_token_id —
    see _resolve_stop_token_ids' docstring for why eos_token_id alone isn't
    enough for this project's chat template. Callers that don't pass it get
    eos_token_id-only behavior (unchanged from before this was added).
    """
    import torch

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    generated = input_ids
    attn = attention_mask
    stop_ids = set(stop_token_ids or ())
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    def _is_stop(token_id: int) -> bool:
        return token_id in stop_ids

    with torch.inference_mode():
        if not fast:
            for _ in range(max_new_tokens):
                outputs = model(input_ids=generated, attention_mask=attn, use_cache=False)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)
                if attn is not None:
                    attn = torch.cat([attn, torch.ones_like(next_token, dtype=attn.dtype)], dim=-1)
                if _is_stop(int(next_token[0, 0])):
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
            if _is_stop(int(next_token[0, 0])):
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
    stop_token_ids = _resolve_stop_token_ids(tokenizer)
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
            stop_token_ids=stop_token_ids,
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
                eos_token_id=list(stop_token_ids) if stop_token_ids else tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _left_pad_batch(tokenizer: Any, prompts: list[str], device: Any) -> tuple[Any, Any]:
    """Tokenize prompts and left-pad them to a common length.

    Left-padding (not this project's training-time padding_side="right") is
    required for batched generation: every row's "next token to generate"
    must sit at the same trailing column so a single argmax over the whole
    batch's last-position logits is valid. Padding is done manually here
    rather than by flipping tokenizer.padding_side, to avoid mutating
    shared tokenizer state other callers (training) depend on being "right".
    """
    import torch

    pad_id = tokenizer.pad_token_id
    encoded = [tokenizer(text=p, return_tensors="pt")["input_ids"][0] for p in prompts]
    max_len = max(e.shape[0] for e in encoded)

    input_ids = torch.full((len(encoded), max_len), pad_id, dtype=encoded[0].dtype)
    attention_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        n = ids.shape[0]
        input_ids[i, max_len - n :] = ids
        attention_mask[i, max_len - n :] = 1
    return input_ids.to(device), attention_mask.to(device)


def _position_ids_from_mask(attention_mask: Any) -> Any:
    """Derive position_ids from a (possibly left-padded) attention mask.

    Matches transformers' own GenerationMixin.prepare_inputs_for_generation
    logic. This project's manual decode loop bypasses that method entirely
    (the documented Gemma+PEFT model.generate() workaround), so nothing
    else computes this — left without it, a model may default to
    torch.arange(seq_len) internally, which is wrong for left-padded rows
    (their real content doesn't start at position 0) and produces
    plausible-looking but incorrect output, not an obvious crash.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def greedy_decode_batch(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    max_new_tokens: int = 512,
    stop_token_ids: Optional[set] = None,
) -> Any:
    """Batched greedy decode with a KV cache — the batched counterpart to
    greedy_decode(fast=True) (always cached; batching an uncached O(n^2)
    loop isn't worth the added code path).

    stop_token_ids, if given, is unioned with tokenizer.eos_token_id — see
    _resolve_stop_token_ids' docstring for why eos_token_id alone isn't
    enough for this project's chat template.

    A sequence that reaches a stop token is "frozen": its next token is
    forced to pad_token_id for every remaining step rather than removing it
    from the batch (removing a row mid-batch would mean re-slicing every
    tensor, including past_key_values, which is more failure-prone than
    just padding it out and truncating the result afterward in
    generate_batch()). Its continued presence doesn't affect other rows —
    batched attention is per-row/independent as long as
    attention_mask/position_ids are correct, which they are.
    """
    import torch

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    pad_id = tokenizer.pad_token_id
    stop_ids = set(stop_token_ids or ())
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    stop_ids_tensor = (
        torch.tensor(sorted(stop_ids), dtype=torch.long, device=input_ids.device) if stop_ids else None
    )

    batch_size = input_ids.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    generated = input_ids
    attn = attention_mask
    past = None
    step_ids = generated
    step_position_ids = _position_ids_from_mask(attn)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=step_ids,
                attention_mask=attn,
                position_ids=step_position_ids,
                use_cache=True,
                past_key_values=past,
            )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if pad_id is not None:
                next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_id), next_token)
            generated = torch.cat([generated, next_token], dim=-1)
            attn = torch.cat([attn, torch.ones_like(next_token, dtype=attn.dtype)], dim=-1)
            if stop_ids_tensor is not None:
                finished = finished | torch.isin(next_token.squeeze(1), stop_ids_tensor)
            if bool(finished.all()):
                break
            step_ids = next_token
            step_position_ids = step_position_ids[:, -1:] + 1

    return generated


def generate_batch(
    model: Any,
    tokenizer: Any,
    messages_batch: list[list[dict]],
    max_new_tokens: int = 512,
) -> list[str]:
    """Generate completions for a batch of prompt-turn lists in one forward-pass batch.

    The batched counterpart to generate() — same build_prompt() +
    greedy-decode-then-fallback structure, but decodes every example in
    messages_batch together instead of one sequential call per example.
    On any failure in the batched path, falls back to generate() one
    example at a time (matches generate()'s own model.generate() fallback
    philosophy: a decode-path failure should degrade the eval loop's
    speed, not crash it outright).
    """
    if not messages_batch:
        return []

    device = _infer_device(model)
    prompts = [build_prompt(tokenizer, m) for m in messages_batch]
    stop_token_ids = _resolve_stop_token_ids(tokenizer)

    try:
        input_ids, attention_mask = _left_pad_batch(tokenizer, prompts, device)
        out = greedy_decode_batch(
            model, input_ids, attention_mask, tokenizer, max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        prompt_len = input_ids.shape[1]
        results = []
        for row in out[:, prompt_len:]:
            ids = row.tolist()
            stop_positions = [ids.index(sid) for sid in stop_token_ids if sid in ids]
            if stop_positions:
                ids = ids[: min(stop_positions) + 1]
            results.append(tokenizer.decode(ids, skip_special_tokens=True))
        return results
    except Exception as exc:
        logger.warning(
            "greedy_decode_batch failed (%s) — falling back to one-at-a-time generate().", exc
        )
        return [generate(model, tokenizer, m, max_new_tokens=max_new_tokens, fast=True) for m in messages_batch]


def run_smoke_test(model: Any, tokenizer: Any, sample_messages: list[dict], max_new_tokens: int = 512) -> str:
    """Generate from sample_messages and print the result (notebook section 13 tail)."""
    generated = generate(model, tokenizer, sample_messages, max_new_tokens=max_new_tokens)
    print("=== GENERATED ===")
    print(generated[:1500])
    return generated
