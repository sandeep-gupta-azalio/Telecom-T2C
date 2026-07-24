"""Smoke tests for src/checkpoint.py's pure filesystem logic (no GPU/Drive needed)."""

from src.checkpoint import find_latest_synced_run


class TestFindLatestSyncedRun:
    def test_returns_none_when_base_dir_missing(self, tmp_path):
        assert find_latest_synced_run(tmp_path / "does_not_exist") is None

    def test_returns_none_when_no_run_dirs(self, tmp_path):
        (tmp_path / "not_a_run").mkdir()
        assert find_latest_synced_run(tmp_path) is None

    def test_ignores_run_dirs_without_adapter_subdir(self, tmp_path):
        (tmp_path / "run_20260101_000000").mkdir()
        assert find_latest_synced_run(tmp_path) is None

    def test_returns_lexicographically_latest_run(self, tmp_path):
        for name in ["run_20260101_000000", "run_20260722_101500", "run_20260305_120000"]:
            (tmp_path / name / "adapter").mkdir(parents=True)
        latest = find_latest_synced_run(tmp_path)
        assert latest.name == "run_20260722_101500"
