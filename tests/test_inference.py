"""Smoke tests for src/inference.py using fake model/tokenizer stand-ins.

Exercises build_prompt's chat-template call and generate()'s
greedy-decode-first-then-model.generate()-fallback logic without needing a
real (multi-GB) model.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed in this environment")
nn = pytest.importorskip("torch.nn", reason="torch not installed in this environment")

from src.inference import (
    _left_pad_batch,
    _position_ids_from_mask,
    _resolve_stop_token_ids,
    build_prompt,
    generate,
    generate_batch,
    greedy_decode,
    greedy_decode_batch,
)

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
        rendered = "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages)
        if add_generation_prompt:
            # Mirrors the REAL quirk this stand-in guards against: for
            # google/gemma-4-12B-it specifically, add_generation_prompt=True
            # opens a "thinking" channel that training data never
            # demonstrates continuing from (confirmed via direct Colab
            # diagnostic — see README Troubleshooting and build_prompt's
            # docstring). build_prompt() must never produce this suffix.
            rendered += "<|turn>model\n<|channel>thought\n<channel|>"
        return rendered

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

    def encode(self, text, add_special_tokens=True):
        # Single-token stand-in for a turn-closing marker, so
        # _resolve_stop_token_ids' discovery succeeds realistically instead
        # of degrading (this fake's chat template doesn't distinguish a
        # real closing sequence from anything else, so any fixed id works).
        return [9]


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

    def generate(
        self, input_ids=None, attention_mask=None, max_new_tokens=None, do_sample=None,
        pad_token_id=None, eos_token_id=None,
    ):
        self.generate_called = True
        extra = torch.full((input_ids.shape[0], 2), 4, dtype=input_ids.dtype)
        return torch.cat([input_ids, extra], dim=-1)


class TestResolveStopTokenIds:
    def test_always_includes_eos_token_id(self):
        assert _EOS_ID in _resolve_stop_token_ids(FakeTokenizer())

    def test_discovers_the_turn_closing_token_when_it_is_a_single_token(self):
        # FakeTokenizer's chat template renders "...<turn|>\n" after the
        # assistant's content, and its encode() resolves any text to a
        # single token id (9) — mirrors the real confirmed shape
        # (eot_token "<turn|>" distinct from eos_token "<eos>").
        result = _resolve_stop_token_ids(FakeTokenizer())
        assert result == {_EOS_ID, 9}

    def test_falls_back_to_eos_only_when_closing_sequence_is_multi_token(self):
        class MultiTokenTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens=True):
                return [9, 10]  # not a single token -> not safe to use as a stop id

        assert _resolve_stop_token_ids(MultiTokenTokenizer()) == {_EOS_ID}

    def test_falls_back_to_eos_only_when_template_rendering_raises(self):
        class BrokenTemplateTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                raise RuntimeError("template broken")

        assert _resolve_stop_token_ids(BrokenTemplateTokenizer()) == {_EOS_ID}

    def test_falls_back_to_eos_only_when_encode_is_missing(self):
        class NoEncodeTokenizer:
            eos_token_id = _EOS_ID

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages)

        assert _resolve_stop_token_ids(NoEncodeTokenizer()) == {_EOS_ID}

    def test_no_eos_token_id_and_no_discoverable_closer_returns_empty_set(self):
        class NoStopTokenizer:
            eos_token_id = None

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                raise RuntimeError("no template")

        assert _resolve_stop_token_ids(NoStopTokenizer()) == set()


class TestBuildPrompt:
    def test_ends_right_after_assistant_turn_header(self):
        tokenizer = FakeTokenizer()
        result = build_prompt(tokenizer, [{"role": "user", "content": "hi"}])
        assert result == "<|turn>user\nhi<turn|>\n<|turn>assistant\n"

    def test_preserves_multiple_prior_turns(self):
        tokenizer = FakeTokenizer()
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        result = build_prompt(tokenizer, messages)
        assert result == "<|turn>system\ns<turn|>\n<|turn>user\nq<turn|>\n<|turn>assistant\n"

    def test_never_opens_the_thinking_channel(self):
        # The regression this function exists to prevent: add_generation_prompt=True
        # would append "<|channel>thought\n<channel|>" for this template, a
        # suffix training data never demonstrates continuing from.
        tokenizer = FakeTokenizer()
        result = build_prompt(tokenizer, [{"role": "user", "content": "hi"}])
        assert "<|channel>thought" not in result
        assert "<|channel>" not in result

    def test_calls_apply_chat_template_with_add_generation_prompt_false(self):
        calls = []

        class RecordingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                calls.append(add_generation_prompt)
                return super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)

        build_prompt(RecordingTokenizer(), [{"role": "user", "content": "hi"}])
        assert calls == [False]

    def test_raises_if_sentinel_not_found_in_rendered_output(self):
        class BrokenTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "no sentinel here"

        with pytest.raises(RuntimeError):
            build_prompt(BrokenTokenizer(), [{"role": "user", "content": "hi"}])


class _FakeCausalLMAlwaysPredicts(nn.Module):
    """Predicts a fixed token id every step, regardless of position — used to
    prove greedy_decode's stop check works for a stop id other than
    eos_token_id (the exact bug: the real model reliably predicts the
    template's actual turn-closer, a token distinct from eos_token_id, and
    a decode loop that only recognizes eos_token_id never stops)."""

    def __init__(self, predicted_token_id: int):
        super().__init__()
        self._linear = nn.Linear(1, 1)
        self.predicted_token_id = predicted_token_id

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, input_ids=None, attention_mask=None, use_cache=None):
        class _Output:
            pass

        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, _VOCAB_SIZE), -10.0)
        logits[:, -1, self.predicted_token_id] = 10.0
        out = _Output()
        out.logits = logits
        return out


class TestGreedyDecodeStopTokens:
    _NON_EOS_STOP_ID = 3

    def test_stops_on_a_discovered_non_eos_stop_token(self):
        model = _FakeCausalLMAlwaysPredicts(self._NON_EOS_STOP_ID)
        tokenizer = FakeTokenizer()
        input_ids = torch.tensor([[1, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1]])

        out = greedy_decode(
            model, input_ids, attention_mask, tokenizer, max_new_tokens=20,
            stop_token_ids={self._NON_EOS_STOP_ID},
        )
        assert out.shape[1] == input_ids.shape[1] + 1

    def test_without_the_extra_stop_token_runs_to_max_new_tokens(self):
        # Same model, but no stop_token_ids given — reproduces the actual
        # bug being fixed: a model that reliably closes its turn with a
        # token other than eos_token_id, decoded by a loop that only knows
        # eos_token_id, never stops and burns the full token budget.
        model = _FakeCausalLMAlwaysPredicts(self._NON_EOS_STOP_ID)
        tokenizer = FakeTokenizer()
        input_ids = torch.tensor([[1, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1]])

        out = greedy_decode(model, input_ids, attention_mask, tokenizer, max_new_tokens=20)
        assert out.shape[1] == input_ids.shape[1] + 20


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


class TestPositionIdsFromMask:
    def test_no_padding_is_a_plain_arange(self):
        mask = torch.tensor([[1, 1, 1]])
        assert _position_ids_from_mask(mask).tolist() == [[0, 1, 2]]

    def test_left_padding_starts_real_tokens_at_zero(self):
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        # Padded slots get masked_fill'd to 1 (transformers' own convention);
        # real content still starts counting at 0 from its own first token.
        assert _position_ids_from_mask(mask).tolist() == [[1, 1, 0, 1, 2]]

    def test_batch_rows_with_different_padding_amounts(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        assert _position_ids_from_mask(mask).tolist() == [[1, 0, 1], [0, 1, 2]]


class _VarLenTokenizer:
    """Minimal tokenizer stand-in whose __call__ length varies with the prompt,
    so _left_pad_batch has genuinely different lengths to pad."""

    pad_token_id = 0

    def __call__(self, text=None, return_tensors="pt"):
        n = len(text.split())
        return {"input_ids": torch.tensor([[7] * n])}


class TestLeftPadBatch:
    def test_pads_shorter_prompts_on_the_left(self):
        tokenizer = _VarLenTokenizer()
        input_ids, attention_mask = _left_pad_batch(tokenizer, ["a b c", "a"], device="cpu")
        assert input_ids.shape == (2, 3)
        assert input_ids[0].tolist() == [7, 7, 7]
        assert input_ids[1].tolist() == [0, 0, 7]
        assert attention_mask[0].tolist() == [1, 1, 1]
        assert attention_mask[1].tolist() == [0, 0, 1]


class FakeBatchCausalLM(nn.Module):
    """Row i emits EOS after exactly targets[i] new tokens (else a filler
    token), letting tests verify a finished row freezes at pad_token_id
    while other rows keep decoding — the interaction greedy_decode_batch's
    `finished` bookkeeping exists for.
    """

    def __init__(self, targets: list[int]):
        super().__init__()
        self._linear = nn.Linear(1, 1)
        self.targets = targets
        self.calls = 0

    def gradient_checkpointing_disable(self):
        pass

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, use_cache=None, past_key_values=None):
        self.calls += 1
        step = self.calls - 1  # call #1 (the prompt call) produces the 1st new token -> step 0

        class _Output:
            pass

        batch = input_ids.shape[0]
        logits = torch.full((batch, input_ids.shape[1], _VOCAB_SIZE), -10.0)
        for row in range(batch):
            token = _EOS_ID if step >= self.targets[row] - 1 else 3
            logits[row, -1, token] = 10.0
        out = _Output()
        out.logits = logits
        out.past_key_values = "fake-cache"
        return out


class FakeBatchTokenizer(FakeTokenizer):
    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(i) for i in ids)


class TestGreedyDecodeBatch:
    def test_finished_row_freezes_at_pad_while_other_row_keeps_going(self):
        model = FakeBatchCausalLM(targets=[1, 3])
        tokenizer = FakeBatchTokenizer()
        input_ids = torch.tensor([[1, 5, 6], [1, 5, 6]])
        attention_mask = torch.ones_like(input_ids)

        out = greedy_decode_batch(model, input_ids, attention_mask, tokenizer, max_new_tokens=10)

        prompt_len = input_ids.shape[1]
        row0_new = out[0, prompt_len:].tolist()
        row1_new = out[1, prompt_len:].tolist()

        assert row0_new[0] == _EOS_ID
        assert all(t == tokenizer.pad_token_id for t in row0_new[1:])
        assert row1_new[:3] == [3, 3, _EOS_ID]

    def test_stops_once_every_row_has_reached_eos(self):
        model = FakeBatchCausalLM(targets=[1, 1])
        tokenizer = FakeBatchTokenizer()
        input_ids = torch.tensor([[1, 5], [1, 5]])
        attention_mask = torch.ones_like(input_ids)

        out = greedy_decode_batch(model, input_ids, attention_mask, tokenizer, max_new_tokens=50)

        # Both rows finish after 1 new token — loop must not run all 50 steps.
        assert out.shape[1] == input_ids.shape[1] + 1


class TestGenerateBatch:
    def test_empty_batch_returns_empty_list(self):
        model = FakeCausalLM()
        tokenizer = FakeTokenizer()
        assert generate_batch(model, tokenizer, [], max_new_tokens=5) == []

    def test_falls_back_to_generate_one_at_a_time_on_batched_failure(self):
        model = FakeCausalLM(fail_forward=False)
        tokenizer = FakeTokenizer()
        messages_batch = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "there"}],
        ]

        # FakeCausalLM.forward() doesn't accept position_ids/past_key_values
        # (only input_ids/attention_mask/use_cache) — greedy_decode_batch
        # always passes them, so this raises a TypeError inside the batched
        # path, exactly the "batched path failed" case generate_batch must
        # degrade from (fall back to generate() per example) rather than
        # propagate.
        result = generate_batch(model, tokenizer, messages_batch, max_new_tokens=5)
        assert result == ["decoded-output", "decoded-output"]
