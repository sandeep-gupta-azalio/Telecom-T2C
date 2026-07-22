"""Post-training benchmark report.

Thin orchestration layer (mirrors the sibling t2c project's
benchmark/runner.py pattern): composes evaluator.py (metrics) + inference.py
(generation) into one reproducible BenchmarkReport artifact. Receives an
already-loaded golden Dataset from the caller rather than importing
dataset.py itself, keeping this module's dependency footprint limited to
config/utils/evaluator/inference/manifest/wandb_logger.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src import evaluator, inference, utils
from src.config import ExperimentConfig

logger = utils.get_logger("benchmark")


@dataclass
class BenchmarkReport:
    run_id: str
    timestamp: str
    base_model: str
    adapter_dir: str
    val_metrics: Optional[dict[str, float]]
    golden_metrics: Optional[dict[str, Any]]
    num_golden_examples: int
    predictions_path: Optional[str]
    pass_metric_status: dict[str, str]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_benchmark(
    config: ExperimentConfig,
    adapter_dir: Path,
    run_dir: Path,
    golden_dataset: Optional[Any],
    hf_token: Optional[str] = None,
    val_metrics: Optional[dict[str, float]] = None,
) -> BenchmarkReport:
    """Reload the trained adapter, run golden generation-eval if available, and write a report.

    val_metrics, if provided, is the loss-based metrics dict already computed
    by trainer.evaluate() during training — passed through here rather than
    recomputed, since recomputing it would require the live SFTTrainer
    instance, which a standalone benchmark run doesn't have.
    """
    inference_model, tokenizer = inference.load_model_for_inference(
        config.model, config.data.max_seq_length, str(adapter_dir), hf_token
    )

    golden_metrics: Optional[dict[str, Any]] = None
    num_golden = 0
    predictions_path: Optional[Path] = None

    if golden_dataset is not None:
        predictions = evaluator.generate_predictions(
            inference_model,
            tokenizer,
            golden_dataset,
            max_new_tokens=config.evaluation.max_new_tokens_eval,
            decode_fn=inference.generate,
        )
        scores = [p.exact_match for p in predictions if p.exact_match is not None]
        golden_metrics = {
            "exact_match_rate": (sum(scores) / len(scores)) if scores else 0.0,
            "num_examples": len(predictions),
        }
        num_golden = len(predictions)
        predictions_dir = utils.ensure_dir(run_dir / "predictions")
        predictions_path = evaluator.export_predictions(predictions, predictions_dir / "golden_predictions.jsonl")
    else:
        logger.info("No golden dataset provided — golden benchmark metrics will be omitted.")

    pass_metric_status = {name: "not_implemented" for name in evaluator.PASS_METRIC_STUBS}

    return BenchmarkReport(
        run_id=run_dir.name,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        base_model=config.model.base_model,
        adapter_dir=str(adapter_dir),
        val_metrics=val_metrics,
        golden_metrics=golden_metrics,
        num_golden_examples=num_golden,
        predictions_path=str(predictions_path) if predictions_path else None,
        pass_metric_status=pass_metric_status,
        notes=(
            "cypher_exact_match compares parsed PASS_4 TIR envelopes (or raw text as a fallback) — "
            "there is no literal Cypher in this dataset. PASS_0-PASS_3 metrics are interfaces only "
            "(pass_metric_status), not yet implemented."
        ),
    )


def write_benchmark_report(report: BenchmarkReport, path: Path) -> Path:
    """Write a BenchmarkReport to path as pretty-printed JSON."""
    utils.ensure_dir(path.parent)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Benchmark report written to %s", path)
    return path
