"""Smoke tests for src/config.py — pure Python, no heavy dependencies."""

from pathlib import Path

import pytest
import yaml

from src.config import (
    ConfigError,
    ExperimentConfig,
    load_config,
    resolve_run_dir,
    save_resolved_config,
    validate_config,
)


def _minimal_yaml(tmp_path: Path, train_path: str = "train.jsonl", val_path: str = "val.jsonl") -> Path:
    data = {
        "data": {"train_path": train_path, "val_path": val_path},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class TestLoadConfig:
    def test_missing_file_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_missing_required_fields_raises_config_error(self, tmp_path):
        path = tmp_path / "experiment.yaml"
        path.write_text(yaml.safe_dump({"data": {}}), encoding="utf-8")
        with pytest.raises(ConfigError, match="train_path"):
            load_config(path)

    def test_valid_minimal_config_loads_with_defaults(self, tmp_path):
        path = _minimal_yaml(tmp_path)
        config = load_config(path)
        assert config.data.train_path == "train.jsonl"
        assert config.data.val_path == "val.jsonl"
        assert config.model.base_model == "google/gemma-4-12B-it"
        assert config.model.continue_adapter is None
        assert config.training.reasoning is True

    def test_unrecognized_field_raises_config_error(self, tmp_path):
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.safe_dump({"data": {"train_path": "t", "val_path": "v", "not_a_real_field": 1}}),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError):
            load_config(path)

    def test_real_shipped_config_loads(self):
        shipped = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
        config = load_config(shipped)
        assert config.model.base_model == "google/gemma-4-12B-it"

    def test_bare_scientific_notation_learning_rate_coerced_to_float(self, tmp_path):
        # PyYAML's SafeLoader does not recognize bare scientific notation
        # (no decimal point) as a float — `learning_rate: 1e-4` parses as the
        # string "1e-4", which used to flow silently into TrainingConfig and
        # crash deep inside Unsloth's compiled SFTConfig instead of here.
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "data": {"train_path": "t", "val_path": "v"},
                    "training": {"learning_rate": "1e-4"},
                }
            ),
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.training.learning_rate == 1e-4
        assert isinstance(config.training.learning_rate, float)

    def test_non_numeric_string_for_float_field_raises_config_error(self, tmp_path):
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "data": {"train_path": "t", "val_path": "v"},
                    "training": {"learning_rate": "not_a_number"},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="learning_rate"):
            load_config(path)


class TestValidateConfig:
    def test_golden_eval_source_without_golden_path_warns(self, tmp_path):
        config = load_config(_minimal_yaml(tmp_path))
        config.data.eval_source = "golden"
        warnings = validate_config(config)
        assert any("golden_path is unset" in w for w in warnings)

    def test_clean_config_has_no_warnings(self, tmp_path):
        config = load_config(_minimal_yaml(tmp_path))
        config.drive.copy_to_drive = False
        warnings = validate_config(config)
        assert warnings == []

    def test_continue_adapter_missing_dir_warns(self, tmp_path):
        config = load_config(_minimal_yaml(tmp_path))
        config.model.continue_adapter = str(tmp_path / "no_such_adapter")
        warnings = validate_config(config)
        assert any("continue_adapter" in w for w in warnings)


class TestResolveRunDir:
    def test_creates_new_timestamped_dir_when_nothing_to_resume(self, tmp_path):
        config = ExperimentConfig()
        config.reproducibility.output_directory = str(tmp_path)
        config.reproducibility.resume_training = True
        run_dir = resolve_run_dir(config)
        assert run_dir.is_dir()
        assert run_dir.parent.name == "runs"
        assert run_dir.name.startswith("run_")

    def test_explicit_run_id_overrides_auto_naming(self, tmp_path):
        config = ExperimentConfig()
        config.reproducibility.output_directory = str(tmp_path)
        config.reproducibility.run_id = "my_custom_run"
        run_dir = resolve_run_dir(config)
        assert run_dir.name == "my_custom_run"

    def test_resumes_most_recent_run_with_checkpoint(self, tmp_path):
        runs_root = tmp_path / "runs"
        old_run = runs_root / "run_20260101_000000"
        new_run = runs_root / "run_20260601_000000"
        (old_run / "adapter" / "checkpoint-100").mkdir(parents=True)
        (new_run / "adapter" / "checkpoint-200").mkdir(parents=True)

        config = ExperimentConfig()
        config.reproducibility.output_directory = str(tmp_path)
        config.reproducibility.resume_training = True
        resolved = resolve_run_dir(config)
        assert resolved == new_run

    def test_does_not_resume_when_resume_training_false(self, tmp_path):
        runs_root = tmp_path / "runs"
        old_run = runs_root / "run_20260101_000000"
        (old_run / "adapter" / "checkpoint-100").mkdir(parents=True)

        config = ExperimentConfig()
        config.reproducibility.output_directory = str(tmp_path)
        config.reproducibility.resume_training = False
        resolved = resolve_run_dir(config)
        assert resolved != old_run
        assert resolved.name.startswith("run_")


class TestSaveResolvedConfig:
    def test_round_trips_via_yaml(self, tmp_path):
        config = load_config(_minimal_yaml(tmp_path))
        out_path = tmp_path / "resolved" / "config.yaml"
        save_resolved_config(config, out_path)
        assert out_path.is_file()
        reloaded = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert reloaded["data"]["train_path"] == "train.jsonl"
        assert reloaded["model"]["base_model"] == "google/gemma-4-12B-it"
