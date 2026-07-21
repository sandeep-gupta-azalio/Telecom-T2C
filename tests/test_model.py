"""Smoke tests for src/model.py.

Covers the CPU-safe logic (target-module resolution, GPU profile table,
attention-implementation resolution, dtype mapping). Actual 4-bit model
loading requires a GPU + network access and is out of scope for these tests
— see README "Testing" for the recommended low-max_train_samples first run
on Colab as the real validation of that path.
"""

import pytest

nn = pytest.importorskip("torch.nn", reason="torch not installed in this environment")

from src.model import (
    GPUProfile,
    detect_gpu_profile,
    resolve_attn_implementation,
    resolve_target_modules,
    resolve_torch_dtype,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 100)
        self.embed_tokens = nn.Embedding(100, 8)


class _EmptyModel(nn.Module):
    pass


class TestResolveTargetModules:
    def test_explicit_override_returned_as_is(self):
        assert resolve_target_modules(_TinyModel(), ["q_proj"]) == ["q_proj"]

    def test_auto_detect_excludes_lm_head_and_embeddings(self):
        result = resolve_target_modules(_TinyModel(), None)
        assert "q_proj" in result
        assert "k_proj" in result
        assert "lm_head" not in result

    def test_no_linear_layers_raises(self):
        with pytest.raises(RuntimeError):
            resolve_target_modules(_EmptyModel(), None)


class TestGpuProfile:
    def test_known_family_override(self):
        profile = detect_gpu_profile(override="A100")
        assert profile.family == "A100"
        assert profile.recommended_batch_size == 4

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError):
            detect_gpu_profile(override="H200")

    def test_auto_detect_returns_a_valid_profile(self):
        profile = detect_gpu_profile(override=None)
        assert isinstance(profile, GPUProfile)
        assert profile.family in ("A100", "L4", "T4", "OTHER", "CPU")

    def test_t4_profile_warns_about_12b_model(self):
        profile = detect_gpu_profile(override="T4")
        assert "marginal" in profile.notes.lower()


class TestAttnImplementation:
    def test_explicit_choices_pass_through(self):
        assert resolve_attn_implementation("sdpa") == "sdpa"
        assert resolve_attn_implementation("flash_attention_2") == "flash_attention_2"

    def test_auto_resolves_to_a_supported_backend(self):
        result = resolve_attn_implementation("auto")
        assert result in ("sdpa", "flash_attention_2")


class TestResolveTorchDtype:
    def test_known_names(self):
        import torch

        assert resolve_torch_dtype("bfloat16") == torch.bfloat16
        assert resolve_torch_dtype("float16") == torch.float16
        assert resolve_torch_dtype("float32") == torch.float32

    def test_unknown_name_defaults_to_bfloat16(self):
        import torch

        assert resolve_torch_dtype("not-a-real-dtype") == torch.bfloat16
