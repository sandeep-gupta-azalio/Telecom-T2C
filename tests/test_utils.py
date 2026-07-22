"""Smoke tests for src/utils.py's disable_unused_transformers_backends().

Uses the real, installed transformers package (no fakes) since the function
patches real module attributes directly — a fake stand-in wouldn't exercise
the actual attribute names being patched. Each test explicitly saves/restores
the patched attributes rather than relying on pytest's monkeypatch fixture,
matching the pattern in test_tokenizer.py for the same reason: the function
intentionally mutates shared, real transformers module state in place.
"""

import pytest

transformers_utils = pytest.importorskip(
    "transformers.utils", reason="transformers not installed in this environment"
)

from src.utils import disable_unused_transformers_backends

_PATCHED_NAMES = ("is_torchaudio_available", "is_torchao_available")


class TestDisableUnusedTransformersBackends:
    def test_forces_both_checks_to_return_false(self):
        import transformers.utils.import_utils as import_utils

        originals = {name: getattr(transformers_utils, name) for name in _PATCHED_NAMES}
        try:
            disable_unused_transformers_backends()
            for name in _PATCHED_NAMES:
                assert getattr(transformers_utils, name)() is False
                assert getattr(import_utils, name)() is False
        finally:
            for name, original in originals.items():
                setattr(transformers_utils, name, original)
                setattr(import_utils, name, original)

    def test_idempotent_when_called_multiple_times(self):
        originals = {name: getattr(transformers_utils, name) for name in _PATCHED_NAMES}
        try:
            disable_unused_transformers_backends()
            disable_unused_transformers_backends()
            for name in _PATCHED_NAMES:
                assert getattr(transformers_utils, name)() is False
        finally:
            for name, original in originals.items():
                setattr(transformers_utils, name, original)

    def test_ignores_extra_call_arguments(self):
        """quantizer_torchao.py calls is_torchao_available(min_version=...) —
        the stub must accept and ignore arbitrary args/kwargs."""
        originals = {name: getattr(transformers_utils, name) for name in _PATCHED_NAMES}
        try:
            disable_unused_transformers_backends()
            assert transformers_utils.is_torchao_available("0.15.0") is False
            assert transformers_utils.is_torchao_available(min_version="0.15.0") is False
        finally:
            for name, original in originals.items():
                setattr(transformers_utils, name, original)
