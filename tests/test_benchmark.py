"""Smoke tests for src/benchmark.py's dataset-selection and report-assembly logic.

Monkeypatches inference.load_model_for_inference/generate so no real
model/GPU is needed — run_benchmark's own logic (which dataset gets used,
how golden_metrics/pass_metrics get assembled) is pure Python otherwise.
"""

from src import benchmark, inference
from src.config import ExperimentConfig

_GOLD_TEXT = (
    'PASS_0\nNormalization\n(none)\n\n'
    'PASS_1\nLexical Detection\n- "query"\n\n'
    'PASS_2\nIntent\nLOOKUP\n\n'
    'PASS_3\nsemantic:\n  operation: LOOKUP\n\n'
    'PASS_4\n{"status": "SUCCESS"}'
)


def _fake_dataset(n: int) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"query {i}"},
                {"role": "assistant", "content": _GOLD_TEXT},
            ]
        }
        for i in range(n)
    ]


def _patch_model_loading(monkeypatch):
    monkeypatch.setattr(
        inference, "load_model_for_inference", lambda model_config, max_seq_length, adapter_dir, hf_token: (object(), object())
    )
    monkeypatch.setattr(inference, "generate", lambda model, tokenizer, messages, max_new_tokens=512: _GOLD_TEXT)


class TestRunBenchmarkDatasetSelection:
    def test_uses_golden_when_available(self, monkeypatch, tmp_path):
        _patch_model_loading(monkeypatch)
        config = ExperimentConfig()
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=_fake_dataset(2),
            fallback_dataset=_fake_dataset(5),
        )
        assert report.eval_dataset_source == "golden"
        assert report.num_golden_examples == 2

    def test_falls_back_to_val_when_golden_missing(self, monkeypatch, tmp_path):
        _patch_model_loading(monkeypatch)
        config = ExperimentConfig()
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=None,
            fallback_dataset=_fake_dataset(3), fallback_dataset_name="val",
        )
        assert report.eval_dataset_source == "val"
        assert report.num_golden_examples == 3

    def test_no_dataset_available_leaves_metrics_empty(self, monkeypatch, tmp_path):
        _patch_model_loading(monkeypatch)
        config = ExperimentConfig()
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=None, fallback_dataset=None,
        )
        assert report.eval_dataset_source is None
        assert report.golden_metrics is None
        assert report.pass_metrics == {}
        assert report.num_golden_examples == 0


class TestRunBenchmarkPassMetrics:
    def test_perfect_predictions_score_100_percent_every_pass(self, monkeypatch, tmp_path):
        _patch_model_loading(monkeypatch)
        config = ExperimentConfig()
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=_fake_dataset(4),
        )
        assert set(report.pass_metrics) == {"PASS_0", "PASS_1", "PASS_2", "PASS_3", "PASS_4"}
        for pass_name, stats in report.pass_metrics.items():
            assert stats["accuracy"] == 1.0, pass_name
            assert stats["num_scored"] == 4
        assert report.golden_metrics["exact_match_rate"] == 1.0

    def test_predictions_exported_under_dataset_source_name(self, monkeypatch, tmp_path):
        _patch_model_loading(monkeypatch)
        config = ExperimentConfig()
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=None,
            fallback_dataset=_fake_dataset(1), fallback_dataset_name="val",
        )
        assert report.predictions_path.endswith("val_predictions.jsonl")
