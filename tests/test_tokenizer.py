"""Smoke tests for src/tokenizer.py's transformers v4/v5 compat shim.

Covers patch_extra_special_tokens_list_format() in isolation, using fake
stand-ins for PreTrainedTokenizerBase._set_model_specific_special_tokens
rather than real model weights — no GPU/network needed. Each test explicitly
saves/restores the patched class attribute (not via pytest's monkeypatch
fixture) since patch_extra_special_tokens_list_format() intentionally
mutates the real, shared transformers class in place, the same way it would
in production.
"""

import pytest

transformers = pytest.importorskip("transformers", reason="transformers not installed in this environment")

from src.tokenizer import patch_chat_template_for_assistant_masking, patch_extra_special_tokens_list_format


def _get_base_class():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    return PreTrainedTokenizerBase


class _DummySelf:
    SPECIAL_TOKENS_ATTRIBUTES: list = []


def _buggy_v4_style(self, special_tokens):
    """Mimics the real, confirmed-buggy transformers v4.x implementation."""
    self.SPECIAL_TOKENS_ATTRIBUTES = self.SPECIAL_TOKENS_ATTRIBUTES + list(special_tokens.keys())
    for key, value in special_tokens.items():
        setattr(self, key, value)


class TestPatchExtraSpecialTokensListFormat:
    def test_converts_list_to_dict_when_original_raises_attribute_error(self):
        base_cls = _get_base_class()
        original = base_cls._set_model_specific_special_tokens
        base_cls._set_model_specific_special_tokens = _buggy_v4_style
        try:
            patch_extra_special_tokens_list_format()
            instance = _DummySelf()
            # A list input crashes the unpatched (buggy v4-style) method —
            # the patch should transparently convert it to a dict and retry.
            base_cls._set_model_specific_special_tokens(instance, ["<image>", "<audio>"])
            assert "<image>" in instance.SPECIAL_TOKENS_ATTRIBUTES
            assert "<audio>" in instance.SPECIAL_TOKENS_ATTRIBUTES
            assert getattr(instance, "<image>") == "<image>"
        finally:
            base_cls._set_model_specific_special_tokens = original

    def test_noop_when_original_already_handles_input(self):
        base_cls = _get_base_class()
        calls = []

        def _v5_style(self, special_tokens):
            calls.append(special_tokens)

        original = base_cls._set_model_specific_special_tokens
        base_cls._set_model_specific_special_tokens = _v5_style
        try:
            patch_extra_special_tokens_list_format()
            instance = _DummySelf()
            base_cls._set_model_specific_special_tokens(instance, ["<image>"])
            # Original (v5-style) implementation handled the list itself —
            # the patch's fallback conversion must never have kicked in.
            assert calls == [["<image>"]]
        finally:
            base_cls._set_model_specific_special_tokens = original

    def test_dict_input_that_still_raises_attribute_error_propagates(self):
        base_cls = _get_base_class()

        def _always_broken(self, special_tokens):
            raise AttributeError("some unrelated attribute error")

        original = base_cls._set_model_specific_special_tokens
        base_cls._set_model_specific_special_tokens = _always_broken
        try:
            patch_extra_special_tokens_list_format()
            instance = _DummySelf()
            # Not a list -> the patch must not swallow the error.
            with pytest.raises(AttributeError):
                base_cls._set_model_specific_special_tokens(instance, {"image_token": "<image>"})
        finally:
            base_cls._set_model_specific_special_tokens = original

    def test_idempotent_when_called_multiple_times(self):
        base_cls = _get_base_class()
        original = base_cls._set_model_specific_special_tokens
        base_cls._set_model_specific_special_tokens = _buggy_v4_style
        try:
            patch_extra_special_tokens_list_format()
            once_patched = base_cls._set_model_specific_special_tokens
            patch_extra_special_tokens_list_format()
            # Second call must not wrap an already-wrapped method again.
            assert base_cls._set_model_specific_special_tokens is once_patched
        finally:
            base_cls._set_model_specific_special_tokens = original


class _FakeTokenizerWithTemplate:
    def __init__(self, chat_template):
        self.chat_template = chat_template


class TestPatchChatTemplateForAssistantMasking:
    def test_wraps_captured_content_for_model_role_only(self):
        # Minimal stand-in for the real anchor line's surrounding structure —
        # exercises only the string-replacement logic, not real Jinja
        # rendering (that was verified separately against the real
        # downloaded google/gemma-4-12B-it template before this was wired
        # into training: byte-identical rendered text, correct per-turn
        # assistant token spans).
        original = "before {{- captured_content -}} after"
        tok = _FakeTokenizerWithTemplate(original)
        patch_chat_template_for_assistant_masking(tok)
        assert "{% generation %}" in tok.chat_template
        assert "{% endgeneration %}" in tok.chat_template
        assert "role == 'model'" in tok.chat_template
        # The anchor's original (unwrapped) form must still appear once, in
        # the else-branch for non-model roles.
        assert tok.chat_template.count("{{- captured_content -}}") == 2

    def test_noop_when_template_already_has_generation_marker(self):
        already_patched = "{% generation %}{{- captured_content -}}{% endgeneration %}"
        tok = _FakeTokenizerWithTemplate(already_patched)
        patch_chat_template_for_assistant_masking(tok)
        assert tok.chat_template == already_patched

    def test_noop_when_no_chat_template(self):
        tok = _FakeTokenizerWithTemplate(None)
        patch_chat_template_for_assistant_masking(tok)  # must not raise
        assert tok.chat_template is None

    def test_raises_when_anchor_not_found(self):
        tok = _FakeTokenizerWithTemplate("some unrelated template with no matching anchor")
        with pytest.raises(RuntimeError, match="chat_template.jinja structure may have changed"):
            patch_chat_template_for_assistant_masking(tok)

    def test_patches_both_outer_processor_and_inner_tokenizer_independently(self):
        # Mirrors the REAL google/gemma-4-12B-it structure, confirmed by
        # loading the actual AutoProcessor locally: the outer processor and
        # its .tokenizer are separate objects with separate (but
        # content-equal) chat_template strings — patching only the outer
        # object would leave the inner one broken, which matters because
        # trainer.train() passes the INNER tokenizer to SFTTrainer (see its
        # docstring) to avoid TRL's VLM detection.
        template = "before {{- captured_content -}} after"
        inner = _FakeTokenizerWithTemplate(template)
        outer = _FakeTokenizerWithTemplate(template)
        outer.tokenizer = inner

        patch_chat_template_for_assistant_masking(outer)

        assert "{% generation %}" in outer.chat_template
        assert "{% generation %}" in inner.chat_template
        # Independent copies — confirm neither patch call accidentally
        # aliased the other's string.
        assert outer.chat_template == inner.chat_template

    def test_noop_when_inner_tokenizer_is_outer_itself(self):
        # Some tokenizer-like objects have a `.tokenizer` attribute that
        # points back at themselves (or don't have one at all) — must not
        # double-patch or infinite-loop.
        template = "before {{- captured_content -}} after"
        tok = _FakeTokenizerWithTemplate(template)
        tok.tokenizer = tok
        patch_chat_template_for_assistant_masking(tok)
        assert tok.chat_template.count("{% generation %}") == 1
