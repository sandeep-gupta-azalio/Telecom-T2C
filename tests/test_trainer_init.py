"""Smoke tests for src/trainer.py's build_sft_config field mapping.

No actual .train() call — trainer initialization only. Skips cleanly if trl
isn't installed in this environment (it's a required dependency per
requirements.txt, but this dev machine may not have it installed).
"""

import pytest

pytest.importorskip("trl", reason="trl package not installed in this environment")

from src.config import ExperimentConfig
from src.trainer import build_sft_config


def _config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.training.epochs = 2.0
    config.training.batch_size = 4
    config.training.eval_batch_size = 4
    config.training.gradient_accumulation = 4
    config.training.learning_rate = 1e-4
    config.training.packing = True
    config.training.eval_steps = 500
    config.data.max_seq_length = 1536
    return config


class TestBuildSftConfig:
    def test_maps_core_training_fields(self, tmp_path):
        config = _config()
        run_dir = tmp_path / "run_x"
        sft_config = build_sft_config(config, run_dir, eval_available=True)

        assert sft_config.num_train_epochs == 2.0
        assert sft_config.per_device_train_batch_size == 4
        assert sft_config.gradient_accumulation_steps == 4
        assert sft_config.learning_rate == 1e-4
        assert sft_config.packing is True
        assert sft_config.max_length == 1536
        assert sft_config.output_dir == str(run_dir / "adapter")

    def test_eval_strategy_no_when_unavailable(self, tmp_path):
        config = _config()
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=False)
        assert sft_config.eval_strategy == "no"

    def test_eval_strategy_steps_when_available(self, tmp_path):
        config = _config()
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=True)
        assert sft_config.eval_strategy == "steps"
        assert sft_config.eval_steps == 500

    def test_early_stopping_enables_load_best_model(self, tmp_path):
        config = _config()
        config.evaluation.early_stopping = True
        config.evaluation.metric_for_best_model = "eval_loss"
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=True)
        assert sft_config.load_best_model_at_end is True
        assert sft_config.metric_for_best_model == "eval_loss"

    def test_early_stopping_ignored_when_eval_unavailable(self, tmp_path):
        config = _config()
        config.evaluation.early_stopping = True
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=False)
        assert getattr(sft_config, "load_best_model_at_end", False) is False

    def test_hf_gradient_checkpointing_always_disabled(self, tmp_path):
        # model.attach_lora() configures checkpointing at the model level via
        # FastModel.get_peft_model(use_gradient_checkpointing="unsloth" or False,
        # driven by this same config.training.gradient_checkpointing flag).
        # SFTConfig must never also enable transformers' own
        # gradient_checkpointing, or Trainer would try to re-configure it on
        # top of Unsloth's own setup — regardless of the flag's value.
        config = _config()
        config.training.gradient_checkpointing = True
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=True)
        assert sft_config.gradient_checkpointing is False
        assert sft_config.gradient_checkpointing_kwargs is None

    def test_packing_strategy_is_wrapped_not_bfd(self, tmp_path):
        # "bfd"/"bfd_split" packing without a supported FlashAttention
        # variant risks real cross-contamination between packed examples —
        # confirmed directly in the actual generated
        # unsloth_compiled_cache/UnslothSFTTrainer.py (pasted back by the
        # user, not the plain pip trl package), which warns about exactly
        # this. This project runs on Unsloth's xformers-based attention
        # kernels, not FlashAttention 2/3 — "wrapped" avoids the risk.
        config = _config()
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=True)
        assert sft_config.packing_strategy == "wrapped"

    def test_padding_free_explicitly_false(self, tmp_path):
        # Confirmed directly in the actual generated
        # unsloth_compiled_cache/UnslothSFTTrainer.py (pasted back by the
        # user): `self.padding_free = args.padding_free or (args.packing and
        # args.packing_strategy in {"bfd", "bfd_split"})`. Even with
        # packing_strategy="wrapped" (making the OR's second term False),
        # `ValueError: When padding_free=True without packing, max_length is
        # not enforced...` still reproduced — meaning args.padding_free
        # itself was already truthy before this ran, evidently injected by
        # Unsloth's own new_init wrapper (unsloth/trainer.py) regardless of
        # packing_strategy. Explicitly setting padding_free=False overrides
        # whatever default gets injected.
        config = _config()
        sft_config = build_sft_config(config, tmp_path / "run_x", eval_available=True)
        assert sft_config.padding_free is False
