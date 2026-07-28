"""Smoke tests for src/utils.py's disable_unused_transformers_backends().

Uses the real, installed transformers package (no fakes) since the function
patches real module attributes directly — a fake stand-in wouldn't exercise
the actual attribute names being patched. Each test explicitly saves/restores
the patched attributes rather than relying on pytest's monkeypatch fixture,
matching the pattern in test_tokenizer.py for the same reason: the function
intentionally mutates shared, real transformers module state in place.
"""

import sys
import types

import pytest

transformers_utils = pytest.importorskip(
    "transformers.utils", reason="transformers not installed in this environment"
)

from src.utils import disable_unused_transformers_backends, resolve_secret


class TestResolveSecret:
    def test_env_var_takes_priority(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "from-env")
        assert resolve_secret("MY_SECRET") == "from-env"

    def test_falls_back_to_colab_secret(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        fake_colab = types.ModuleType("google.colab")
        fake_userdata = types.ModuleType("google.colab.userdata")
        fake_userdata.get = lambda name: "from-colab" if name == "MY_SECRET" else None
        fake_colab.userdata = fake_userdata
        monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
        monkeypatch.setitem(sys.modules, "google.colab.userdata", fake_userdata)

        assert resolve_secret("MY_SECRET") == "from-colab"

    def test_falls_back_to_kaggle_secret_when_colab_unavailable(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab.userdata", raising=False)

        class FakeUserSecretsClient:
            def get_secret(self, name):
                return "from-kaggle" if name == "MY_SECRET" else None

        fake_kaggle_secrets = types.ModuleType("kaggle_secrets")
        fake_kaggle_secrets.UserSecretsClient = FakeUserSecretsClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle_secrets)

        assert resolve_secret("MY_SECRET") == "from-kaggle"

    def test_returns_none_when_not_found_anywhere(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        monkeypatch.delitem(sys.modules, "kaggle_secrets", raising=False)

        assert resolve_secret("MY_SECRET") is None

    def test_kaggle_secret_not_found_falls_through_to_none(self, monkeypatch):
        # A missing named secret (not a missing platform) must not raise.
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)

        class FakeUserSecretsClient:
            def get_secret(self, name):
                raise Exception("secret not found")

        fake_kaggle_secrets = types.ModuleType("kaggle_secrets")
        fake_kaggle_secrets.UserSecretsClient = FakeUserSecretsClient
        monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_kaggle_secrets)

        assert resolve_secret("MY_SECRET") is None

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
