"""Evaluation: validation loss, golden generation-eval, prediction export.

Decode logic (inference.generate) is injected as a callable rather than
imported directly, so this module never depends on inference.py — that keeps
the dependency graph a strict DAG (evaluator sits below callbacks/inference,
not above them) even though the concepts are related.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src import utils

logger = utils.get_logger("evaluator")

PassMetric = Callable[[str, str], Optional[float]]


@dataclass
class PredictionRecord:
    prompt: str
    generated: str
    gold: str
    exact_match: Optional[float]


@dataclass
class EvalResult:
    val_loss: Optional[float]
    val_metrics: dict[str, float] = field(default_factory=dict)
    golden_metrics: Optional[dict[str, Any]] = None
    predictions_path: Optional[Path] = None
    num_golden_examples: int = 0


def evaluate_validation(trainer: Any) -> dict[str, float]:
    """Thin wrapper around SFTTrainer.evaluate().

    Forces eager (non-compiled) execution for the duration of this call via
    torch.compiler.set_stance("force_eager") — training itself keeps
    Unsloth's full torch.compile-based speedup (a large fraction of its
    advertised "2x faster" claim), since this context manager only affects
    code within its scope. This works around a confirmed, reproduced crash
    (`InternalTorchDynamoError: AcceleratorError: CUDA error: an illegal
    memory access was encountered`) inside dynamo's tracing of Unsloth's
    compiled Gemma4UnifiedTextAttention/RMSNorm forward specifically during
    evaluate() — training completed successfully with compilation enabled
    before this crash was ever hit, so the instability appears scoped to
    eval mode, not compilation in general (see README Troubleshooting).
    Deliberately NOT using torch.compiler.disable() here: it's documented as
    unreliable as a context manager (pytorch/pytorch#123771); set_stance is
    the stable, confirmed-working API for this.
    """
    import torch

    with torch.compiler.set_stance("force_eager"):
        return trainer.evaluate()


def parse_pass4_envelope(text: str) -> Optional[dict[str, Any]]:
    """Locate and parse the PASS_4 TIR JSON envelope within an assistant turn.

    Pure text parsing — no dependency on the sibling t2c package. The
    extracted shape (status/operation/subject/qualifiers/...) matches that
    project's TirL1.to_dict() conceptually only; this function does not
    import or validate against it.

    Uses brace-depth matching (not just json.loads on a fixed slice) so
    nested objects inside the envelope parse correctly. Returns None if no
    "PASS_4" marker is found, or the located braces don't parse as JSON.
    """
    marker_idx = text.find("PASS_4")
    if marker_idx == -1:
        return None
    start = text.find("{", marker_idx)
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def cypher_exact_match(prediction: str, gold: str) -> float:
    """Exact-match comparator between a generated and gold assistant turn.

    NOTE on naming: despite the name (kept to match the project spec
    literally), there is no literal Cypher text anywhere in this dataset —
    each assistant turn is a PASS_0-4 structured blob ending in a TIR JSON
    envelope, not a Cypher query. This function prefers a structural
    comparison of the parsed PASS_4 envelope (order-independent — dict
    equality) when both sides parse; it falls back to raw stripped-text
    equality otherwise.
    """
    pred_envelope = parse_pass4_envelope(prediction)
    gold_envelope = parse_pass4_envelope(gold)
    if pred_envelope is not None and gold_envelope is not None:
        return 1.0 if pred_envelope == gold_envelope else 0.0
    return 1.0 if prediction.strip() == gold.strip() else 0.0


def _pass_stub(pass_name: str, description: str) -> PassMetric:
    def _metric(prediction: str, gold: str) -> Optional[float]:
        raise NotImplementedError(
            f"{pass_name} metric ({description}) is not implemented yet — interface only, "
            "per project spec. Wire in a real comparator here when ready."
        )

    _metric.__name__ = f"{pass_name.lower()}_metric"
    _metric.__doc__ = f"{pass_name}: {description}. Not yet implemented — placeholder interface only."
    return _metric


# Placeholder interfaces only (explicitly requested — do not implement yet).
# Each corresponds 1:1 to a block in the dataset's assistant-turn structure.
PASS_METRIC_STUBS: dict[str, PassMetric] = {
    "PASS_0": _pass_stub("PASS_0", "Normalization — spelling/token fixes only"),
    "PASS_1": _pass_stub("PASS_1", "Lexical Detection — quoted verbatim phrases from normalized text"),
    "PASS_2": _pass_stub("PASS_2", "Intent — exactly one canonical operation"),
    "PASS_3": _pass_stub("PASS_3", "Semantic Resolution — YAML semantic record"),
    "PASS_4": _pass_stub("PASS_4", "TIR envelope JSON — status and diagnostics"),
}


def _prepare_prompt_and_gold(messages: list[dict]) -> Optional[tuple[list[dict], str]]:
    """Split a conversation into (prompt turns, gold reply) using its final assistant turn.

    Returns None if the conversation doesn't end on an assistant turn (not
    evaluable this way).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    return messages[:-1], messages[-1].get("content", "")


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_new_tokens: int,
    decode_fn: Callable[[Any, Any, list[dict], int], str],
) -> list[PredictionRecord]:
    """Generate a prediction for each example's final assistant turn and score it.

    decode_fn(model, tokenizer, prompt_messages, max_new_tokens) -> str is
    injected by the caller (trainer.py / benchmark.py wire in
    inference.generate) so this module never imports inference.py directly.
    """
    records: list[PredictionRecord] = []
    for example in dataset:
        messages = example["messages"] if isinstance(example, dict) else example["messages"]
        split = _prepare_prompt_and_gold(list(messages))
        if split is None:
            continue
        prompt_messages, gold = split
        generated = decode_fn(model, tokenizer, prompt_messages, max_new_tokens)
        score = cypher_exact_match(generated, gold)
        records.append(
            PredictionRecord(
                prompt=json.dumps(prompt_messages, ensure_ascii=False),
                generated=generated,
                gold=gold,
                exact_match=score,
            )
        )
    return records


def run_golden_evaluation(
    model: Any,
    tokenizer: Any,
    golden_dataset: Optional[Any],
    decode_fn: Callable[[Any, Any, list[dict], int], str],
    max_new_tokens: int = 512,
    max_samples: Optional[int] = None,
) -> Optional[EvalResult]:
    """Run generation-based evaluation against the golden set.

    Returns None gracefully if golden_dataset is None (no golden file
    configured/found today — the normal case).
    """
    if golden_dataset is None:
        logger.info("No golden dataset available — skipping golden evaluation.")
        return None

    dataset = golden_dataset
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    predictions = generate_predictions(model, tokenizer, dataset, max_new_tokens, decode_fn)
    scores = [p.exact_match for p in predictions if p.exact_match is not None]
    exact_match_rate = (sum(scores) / len(scores)) if scores else 0.0

    golden_metrics = {
        "exact_match_rate": exact_match_rate,
        "num_examples": len(predictions),
    }
    return EvalResult(
        val_loss=None,
        val_metrics={},
        golden_metrics=golden_metrics,
        predictions_path=None,
        num_golden_examples=len(predictions),
    )


def export_predictions(predictions: list[PredictionRecord], out_path: Path) -> Path:
    """Write predictions to JSONL at out_path."""
    rows = (dataclasses.asdict(p) for p in predictions)
    utils.write_jsonl(out_path, rows)
    logger.info("Exported %d predictions to %s", len(predictions), out_path)
    return out_path
