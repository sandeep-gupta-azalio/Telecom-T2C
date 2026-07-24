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
    pass_metrics: dict[str, dict[str, Any]]
    eval_dataset_source: Optional[str]
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
    fallback_dataset: Optional[Any] = None,
    fallback_dataset_name: str = "val",
) -> BenchmarkReport:
    """Reload the trained adapter, run generation-based eval, and write a report.

    val_metrics, if provided, is the loss-based metrics dict already computed
    by trainer.evaluate() during training — passed through here rather than
    recomputed, since recomputing it would require the live SFTTrainer
    instance, which a standalone benchmark run doesn't have.

    fallback_dataset (typically val_ds) is used for the generation-based
    metrics (exact-match + per-PASS accuracy) ONLY when golden_dataset is
    None — this project's default config has data.golden_path unset, so
    without this fallback the benchmark would never actually produce
    generation metrics for a typical run. eval_dataset_source in the
    returned report records which one was actually used ("golden",
    fallback_dataset_name, or None if neither was available).
    """
    inference_model, tokenizer = inference.load_model_for_inference(
        config.model, config.data.max_seq_length, str(adapter_dir), hf_token
    )

    eval_dataset = golden_dataset if golden_dataset is not None else fallback_dataset
    eval_dataset_source = "golden" if golden_dataset is not None else (fallback_dataset_name if fallback_dataset is not None else None)

    golden_metrics: Optional[dict[str, Any]] = None
    pass_metrics: dict[str, dict[str, Any]] = {}
    num_examples = 0
    predictions_path: Optional[Path] = None

    if eval_dataset is not None:
        predictions = evaluator.generate_predictions(
            inference_model,
            tokenizer,
            eval_dataset,
            max_new_tokens=config.evaluation.max_new_tokens_eval,
            decode_fn=inference.generate,
        )
        scores = [p.exact_match for p in predictions if p.exact_match is not None]
        golden_metrics = {
            "exact_match_rate": (sum(scores) / len(scores)) if scores else 0.0,
            "num_examples": len(predictions),
        }
        pass_metrics = {name: acc.to_dict() for name, acc in evaluator.evaluate_passes(predictions).items()}
        num_examples = len(predictions)
        predictions_dir = utils.ensure_dir(run_dir / "predictions")
        predictions_path = evaluator.export_predictions(
            predictions, predictions_dir / f"{eval_dataset_source}_predictions.jsonl"
        )
    else:
        logger.info("No golden or fallback dataset provided — generation-based benchmark metrics will be omitted.")

    return BenchmarkReport(
        run_id=run_dir.name,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        base_model=config.model.base_model,
        adapter_dir=str(adapter_dir),
        val_metrics=val_metrics,
        golden_metrics=golden_metrics,
        num_golden_examples=num_examples,
        predictions_path=str(predictions_path) if predictions_path else None,
        pass_metrics=pass_metrics,
        eval_dataset_source=eval_dataset_source,
        notes=(
            "golden_metrics.exact_match_rate compares parsed PASS_4 TIR envelopes (or raw text as a "
            "fallback) — there is no literal Cypher in this dataset. pass_metrics gives per-PASS_0-4 "
            "accuracy plus parse-failure counts (see evaluator.evaluate_passes); eval_dataset_source "
            "records whether golden or the val-set fallback was used."
        ),
    )


def write_benchmark_report(report: BenchmarkReport, path: Path) -> Path:
    """Write a BenchmarkReport to path as pretty-printed JSON."""
    utils.ensure_dir(path.parent)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Benchmark report written to %s", path)
    return path
