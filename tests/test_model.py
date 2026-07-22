"""Smoke tests for src/model.py.

Covers the CPU-safe logic (GPU profile table). Actual 4-bit model loading
via Unsloth requires a GPU + network access and is out of scope for these
tests — see README "Testing" for the recommended low-max_train_samples first
run on Colab as the real validation of that path.
"""

import pytest

pytest.importorskip("torch", reason="torch not installed in this environment")

from src.model import GPUProfile, detect_gpu_profile


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
