"""Smoke tests for src/dataset.py.

Validation-logic tests (validate_json / validate_messages / validate_roles)
are pure Python and always run. Split-loading tests need the `datasets`
package and are skipped cleanly if it isn't installed in this environment.
"""

import json
from pathlib import Path

import pytest

from src.config import DataConfig
from src.dataset import DatasetLoader

FIXTURES = Path(__file__).resolve().parent / "fixtures"

datasets = pytest.importorskip("datasets", reason="datasets package not installed in this environment")


class TestValidateJson:
    def test_valid(self):
        ok, record, reason = DatasetLoader.validate_json('{"messages": []}')
        assert ok
        assert record == {"messages": []}
        assert reason is None

    def test_invalid_json(self):
        ok, record, reason = DatasetLoader.validate_json("{not valid json")
        assert not ok
        assert record is None
        assert reason.startswith("invalid_json")

    def test_not_an_object(self):
        ok, record, reason = DatasetLoader.validate_json("[1, 2, 3]")
        assert not ok
        assert reason == "not_a_json_object"


class TestValidateMessages:
    def test_missing_key(self):
        ok, reason = DatasetLoader.validate_messages({})
        assert not ok
        assert reason == "missing_messages_key"

    def test_empty_list(self):
        ok, reason = DatasetLoader.validate_messages({"messages": []})
        assert not ok
        assert reason == "messages_not_a_nonempty_list"

    def test_valid(self):
        ok, reason = DatasetLoader.validate_messages({"messages": [{"role": "user", "content": "hi"}]})
        assert ok
        assert reason is None


class TestValidateRoles:
    def test_valid_real_shape(self):
        # Matches the real dataset: system, deployment-context user, then
        # repeated (user query, assistant PASS_0-4) pairs.
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "deployment context"},
            {"role": "user", "content": "query 1"},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "query 2"},
            {"role": "assistant", "content": "response 2"},
        ]
        ok, reason = DatasetLoader.validate_roles(messages)
        assert ok, reason

    def test_invalid_role(self):
        ok, reason = DatasetLoader.validate_roles([{"role": "bot", "content": "x"}])
        assert not ok
        assert "invalid_role" in reason

    def test_empty_content(self):
        ok, reason = DatasetLoader.validate_roles([{"role": "user", "content": "   "}])
        assert not ok
        assert "empty_content" in reason

    def test_system_not_first(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "late system"},
        ]
        ok, reason = DatasetLoader.validate_roles(messages)
        assert not ok
        assert reason == "system_role_not_first"

    def test_does_not_start_with_user(self):
        ok, reason = DatasetLoader.validate_roles([{"role": "assistant", "content": "hi"}])
        assert not ok
        assert reason == "does_not_start_with_user_turn"

    def test_consecutive_assistant_turns(self):
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
        ]
        ok, reason = DatasetLoader.validate_roles(messages)
        assert not ok
        assert reason == "consecutive_assistant_turns"


class TestLoadSplits:
    def _config(self, max_train=None):
        return DataConfig(
            train_path=str(FIXTURES / "sample_train.jsonl"),
            val_path=str(FIXTURES / "sample_val.jsonl"),
            golden_path=None,
            max_train_samples=max_train,
            max_eval_samples=None,
            max_seq_length=1536,
        )

    def test_load_train_valid_fixture(self):
        loader = DatasetLoader(self._config(), tokenizer=None, seed=42)
        ds = loader.load_train()
        assert len(ds) == 5
        assert "messages" in ds.column_names

    def test_load_validation_valid_fixture(self):
        loader = DatasetLoader(self._config(), tokenizer=None, seed=42)
        ds = loader.load_validation()
        assert ds is not None
        assert len(ds) == 2

    def test_load_golden_missing_returns_none(self):
        loader = DatasetLoader(self._config(), tokenizer=None, seed=42)
        assert loader.load_golden() is None

    def test_load_train_missing_file_raises(self, tmp_path):
        config = DataConfig(train_path=str(tmp_path / "does_not_exist.jsonl"), val_path="")
        loader = DatasetLoader(config, tokenizer=None, seed=42)
        with pytest.raises(FileNotFoundError):
            loader.load_train()

    def test_corrupted_json_line_is_dropped_not_fatal(self, tmp_path):
        bad_path = tmp_path / "train_with_bad_line.jsonl"
        good_line = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "there"}]}
        )
        bad_path.write_text(good_line + "\n{not valid json\n" + good_line + "\n", encoding="utf-8")
        config = DataConfig(train_path=str(bad_path), val_path="")
        loader = DatasetLoader(config, tokenizer=None, seed=42)
        ds = loader.load_train()
        assert len(ds) == 2  # both good lines kept, the corrupted line dropped

    def test_max_train_samples_caps_dataset(self):
        loader = DatasetLoader(self._config(max_train=2), tokenizer=None, seed=42)
        ds = loader.load_train()
        assert len(ds) == 2
