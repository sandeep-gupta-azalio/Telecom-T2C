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

from src.tokenizer import patch_extra_special_tokens_list_format


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
