"""Smoke tests for src/inference.py using fake model/tokenizer stand-ins.

Exercises build_prompt's chat-template call and generate()'s
greedy-decode-first-then-model.generate()-fallback logic without needing a
real (multi-GB) model.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this environment")
nn = pytest.importorskip("torch.nn", reason="torch not installed in this environment")

from src.inference import build_prompt, generate

_EOS_ID = 2
_VOCAB_SIZE = 5


class _FakeBatchEncoding(dict):
    def to(self, device):  # noqa: ARG002 - mirrors HF BatchEncoding.to(device) signature
        return self


class FakeTokenizer:
    eos_token_id = _EOS_ID
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            return [1, 2, 3]
        prompt = " | ".join(m["content"] for m in messages)
        return f"PROMPT:{prompt}"

    def __call__(self, images=None, *, text=None, return_tensors="pt"):
        # Mirrors the REAL bug this stand-in guards against: Gemma 4 is
        # nominally multimodal, so Unsloth/transformers loads the real
        # tokenizer as a Gemma4UnifiedProcessor whose __call__ signature is
        # (self, images=None, text=None, videos=None, audio=None, **kwargs).
        # `images` deliberately comes first and `text` is keyword-only here
        # so that a regression back to a positional `tokenizer(prompt, ...)`
        # call in inference.generate() fails loudly (prompt would bind to
        # `images`, then this assert catches it) instead of silently passing.
        assert images is None, "text must be passed as a keyword, not positionally (see inference.generate())"
        assert text is not None
        return _FakeBatchEncoding(
            {
                "input_ids": torch.tensor([[1, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    def decode(self, ids, skip_special_tokens=True):
        return "decoded-output"


class FakeCausalLM(nn.Module):
    """A minimal stand-in that either succeeds at manual greedy decode or fails it."""

    def __init__(self, fail_forward: bool = False):
        super().__init__()
        self._linear = nn.Linear(1, 1)  # gives .parameters() something with a .device
        self.fail_forward = fail_forward
        self.generate_called = False

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, input_ids=None, attention_mask=None, use_cache=None):
        if self.fail_forward:
            raise RuntimeError("simulated forward failure")

        class _Output:
            pass

        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, _VOCAB_SIZE)
        logits[:, -1, _EOS_ID] = 10.0  # force argmax -> eos, so the decode loop ends in one step
        out = _Output()
        out.logits = logits
        return out

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, do_sample=None, pad_token_id=None):
        self.generate_called = True
        extra = torch.full((input_ids.shape[0], 2), 4, dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=-1)


class TestBuildPrompt:
    def test_uses_chat_template_with_generation_prompt(self):
        tokenizer = FakeTokenizer()
        result = build_prompt(tokenizer, [{"role": "user", "content": "hi"}])
        assert result == "PROMPT:hi"


class TestGenerate:
    def test_greedy_decode_succeeds_without_fallback(self):
        model = FakeCausalLM(fail_forward=False)
        tokenizer = FakeTokenizer()
        result = generate(model, tokenizer, [{"role": "user", "content": "hi"}], max_new_tokens=5)
        assert result == "decoded-output"
        assert model.generate_called is False

    def test_falls_back_to_model_generate_on_greedy_decode_failure(self):
        model = FakeCausalLM(fail_forward=True)
        tokenizer = FakeTokenizer()
        result = generate(model, tokenizer, [{"role": "user", "content": "hi"}], max_new_tokens=5)
        assert result == "decoded-output"
        assert model.generate_called is True
