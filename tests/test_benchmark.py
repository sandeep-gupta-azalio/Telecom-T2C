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
    monkeypatch.setattr(
        inference,
        "generate",
        lambda model, tokenizer, messages, max_new_tokens=512, fast=False: _GOLD_TEXT,
    )


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

    def test_uses_batched_decode_when_generation_batch_size_configured(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            inference, "load_model_for_inference", lambda model_config, max_seq_length, adapter_dir, hf_token: (object(), object())
        )
        batch_calls = []

        def fake_generate_batch(model, tokenizer, messages_batch, max_new_tokens=512):
            batch_calls.append(len(messages_batch))
            return [_GOLD_TEXT] * len(messages_batch)

        monkeypatch.setattr(inference, "generate_batch", fake_generate_batch)
        # decode_fn (single-example) must NOT be used at all once batching kicks in.
        monkeypatch.setattr(
            inference, "generate", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called"))
        )

        config = ExperimentConfig()
        config.evaluation.generation_batch_size = 3
        report = benchmark.run_benchmark(
            config, tmp_path / "adapter", tmp_path, golden_dataset=_fake_dataset(5),
        )

        assert batch_calls == [3, 2]
        assert report.golden_metrics["exact_match_rate"] == 1.0
        assert report.golden_metrics["num_examples"] == 5
