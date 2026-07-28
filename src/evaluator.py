"""Evaluation: validation loss, golden generation-eval, prediction export.

Decode logic (inference.generate) is injected as a callable rather than
imported directly, so this module never depends on inference.py — that keeps
the dependency graph a strict DAG (evaluator sits below callbacks/inference,
not above them) even though the concepts are related.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

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


_PASS_NAMES: tuple[str, ...] = ("PASS_0", "PASS_1", "PASS_2", "PASS_3", "PASS_4")


def _extract_pass_section(text: str, pass_name: str) -> Optional[str]:
    """Return the raw text between `pass_name`'s marker and the next PASS_N marker (or end of text).

    Locates the EARLIEST occurrence of any other PASS_N marker after
    pass_name's position, rather than assuming PASS_0..PASS_4 always appear
    in order immediately after one another — robust to a malformed
    generation that garbles ordering or omits a section. Returns None if
    pass_name itself isn't found at all.
    """
    start = text.find(pass_name)
    if start == -1:
        return None
    start += len(pass_name)
    end = len(text)
    for other in _PASS_NAMES:
        if other == pass_name:
            continue
        idx = text.find(other, start)
        if idx != -1 and idx < end:
            end = idx
    return text[start:end].strip("\n")


def _section_lines(section: str, header: str) -> list[str]:
    """Split a pass section into non-blank lines, dropping its literal header line if present."""
    lines = [ln for ln in section.splitlines() if ln.strip()]
    if lines and lines[0].strip() == header:
        lines = lines[1:]
    return lines


def parse_pass0_normalizations(text: str) -> Optional[list[tuple[str, str]]]:
    """Parse PASS_0's Normalization block into (surface, normalized) pairs.

    Matches pass_builder.PassLabels.render_assistant's rendering: "(none)"
    on its own for zero normalizations, else src/"↓"/dst line triples.
    Returns None if the PASS_0 marker is missing or the section doesn't fit
    either shape (malformed generation).
    """
    section = _extract_pass_section(text, "PASS_0")
    if section is None:
        return None
    lines = _section_lines(section, "Normalization")
    if not lines or lines[0].strip() == "(none)":
        return []
    if len(lines) % 3 != 0:
        return None
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(lines), 3):
        src, arrow, dst = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
        if arrow != "↓":
            return None
        pairs.append((src, dst))
    return pairs


def parse_pass1_lexemes(text: str) -> Optional[list[str]]:
    """Parse PASS_1's Lexical Detection block into an ordered list of quoted lexemes.

    Each real line is `- "lexeme"`; returns None if the PASS_1 marker is
    missing or any non-blank line doesn't match that shape.
    """
    section = _extract_pass_section(text, "PASS_1")
    if section is None:
        return None
    lexemes: list[str] = []
    for line in _section_lines(section, "Lexical Detection"):
        match = re.match(r'^-\s*"(.*)"\s*$', line.strip())
        if match is None:
            return None
        lexemes.append(match.group(1))
    return lexemes


def parse_pass2_intent(text: str) -> Optional[str]:
    """Parse PASS_2's Intent block into its single canonical operation string."""
    section = _extract_pass_section(text, "PASS_2")
    if section is None:
        return None
    lines = _section_lines(section, "Intent")
    if len(lines) != 1:
        return None
    return lines[0].strip()


def parse_pass3_semantic(text: str) -> Optional[dict[str, Any]]:
    """Parse PASS_3's Semantic Resolution block (plain YAML, no literal header line) into a dict."""
    section = _extract_pass_section(text, "PASS_3")
    if not section:
        return None
    try:
        data = yaml.safe_load(section)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


# Parser registry: one entry per block in the dataset's assistant-turn
# structure (PassLabels.render_assistant in the sibling t2c project). Each
# parser takes a full assistant-turn string and returns a structured,
# equality-comparable value, or None if that pass's marker is missing or its
# content doesn't fit the expected shape — used by evaluate_passes() below
# to score prediction vs. gold per pass, and to distinguish "wrong value"
# from "failed to produce a parseable section at all".
PASS_PARSERS: dict[str, Callable[[str], Any]] = {
    "PASS_0": parse_pass0_normalizations,
    "PASS_1": parse_pass1_lexemes,
    "PASS_2": parse_pass2_intent,
    "PASS_3": parse_pass3_semantic,
    "PASS_4": parse_pass4_envelope,
}


def _make_pass_metric(parser: Callable[[str], Any]) -> PassMetric:
    def _metric(prediction: str, gold: str) -> Optional[float]:
        gold_value = parser(gold)
        if gold_value is None:
            return None
        return 1.0 if parser(prediction) == gold_value else 0.0

    return _metric


# One (prediction, gold) -> Optional[float] comparator per pass, for callers
# that just want a single score rather than evaluate_passes' full breakdown.
PASS_METRICS: dict[str, PassMetric] = {
    name: _make_pass_metric(parser) for name, parser in PASS_PARSERS.items()
}


@dataclass
class PassAccuracy:
    accuracy: float
    num_scored: int
    num_gold_unparseable: int
    num_prediction_unparseable: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_passes(predictions: list["PredictionRecord"]) -> dict[str, PassAccuracy]:
    """Score every prediction against its gold reply, per PASS_0..PASS_4.

    For each pass: examples whose GOLD reply doesn't parse for that pass are
    excluded from the denominator entirely (num_gold_unparseable tracks how
    many — this should be ~0 on real data; a nonzero count usually means a
    parser bug here, not a bad dataset row). Among the rest, a prediction
    that doesn't parse counts as wrong (num_prediction_unparseable tracks
    how many of those wrong answers were specifically "produced no
    parseable section" rather than "parsed but had the wrong value") —
    distinguishing malformed-output failures from wrong-value failures is
    the main diagnostic value of this function over a single blended score.
    """
    report: dict[str, PassAccuracy] = {}
    for pass_name, parser in PASS_PARSERS.items():
        scored = 0
        correct = 0
        gold_unparseable = 0
        prediction_unparseable = 0
        for record in predictions:
            gold_value = parser(record.gold)
            if gold_value is None:
                gold_unparseable += 1
                continue
            scored += 1
            prediction_value = parser(record.generated)
            if prediction_value is None:
                prediction_unparseable += 1
            elif prediction_value == gold_value:
                correct += 1
        report[pass_name] = PassAccuracy(
            accuracy=(correct / scored) if scored else 0.0,
            num_scored=scored,
            num_gold_unparseable=gold_unparseable,
            num_prediction_unparseable=prediction_unparseable,
        )
    return report


def _prepare_prompt_and_gold(messages: list[dict]) -> Optional[tuple[list[dict], str]]:
    """Split a conversation into (prompt turns, gold reply) using its final assistant turn.

    Returns None if the conversation doesn't end on an assistant turn (not
    evaluable this way).
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    return messages[:-1], messages[-1].get("content", "")


def _build_prediction_record(prompt_messages: list[dict], generated: str, gold: str) -> PredictionRecord:
    return PredictionRecord(
        prompt=json.dumps(prompt_messages, ensure_ascii=False),
        generated=generated,
        gold=gold,
        exact_match=cypher_exact_match(generated, gold),
    )


def generate_predictions(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_new_tokens: int,
    decode_fn: Callable[[Any, Any, list[dict], int], str],
    batch_size: int = 1,
    batch_decode_fn: Optional[Callable[[Any, Any, list[list[dict]], int], list[str]]] = None,
) -> list[PredictionRecord]:
    """Generate a prediction for each example's final assistant turn and score it.

    decode_fn(model, tokenizer, prompt_messages, max_new_tokens) -> str is
    injected by the caller (trainer.py / benchmark.py wire in
    inference.generate) so this module never imports inference.py directly.

    batch_size > 1 groups examples into batches and calls batch_decode_fn
    (required in that case — inference.generate_batch is the canonical
    implementation) instead of decode_fn once per example, which is what
    makes a 256-example generation-eval run take minutes instead of tens of
    minutes on a single GPU. batch_size == 1 (the default) is unchanged
    behavior — existing callers passing only decode_fn are unaffected.
    Scoring/record-building is identical either way.
    """
    if batch_size > 1 and batch_decode_fn is None:
        raise ValueError("batch_size > 1 requires batch_decode_fn (e.g. inference.generate_batch).")

    prepared: list[tuple[list[dict], str]] = []
    for example in dataset:
        messages = example["messages"] if isinstance(example, dict) else example["messages"]
        split = _prepare_prompt_and_gold(list(messages))
        if split is not None:
            prepared.append(split)

    records: list[PredictionRecord] = []
    if batch_size <= 1:
        for prompt_messages, gold in prepared:
            generated = decode_fn(model, tokenizer, prompt_messages, max_new_tokens)
            records.append(_build_prediction_record(prompt_messages, generated, gold))
        return records

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        generated_list = batch_decode_fn(model, tokenizer, [p for p, _ in chunk], max_new_tokens)
        for (prompt_messages, gold), generated in zip(chunk, generated_list):
            records.append(_build_prediction_record(prompt_messages, generated, gold))
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


@dataclass
class LookupLevelQuery:
    """One phrasing of a lookup, at one of 4 increasing-implicitness levels.

    group_id ties together every phrasing of the SAME underlying
    entity+identifier lookup (e.g. "Show ONU <sn>" / "Find subscriber <sn>"
    / "Show <sn>" / "Need details for subscriber <sn>" all share a
    group_id) — see run_lookup_level_benchmark's docstring for why the
    group is the unit of comparison, not an externally-supplied gold answer.
    """

    group_id: str
    level: int
    level_name: str
    query: str


@dataclass
class LookupLevelResult:
    group_id: str
    level: int
    level_name: str
    query: str
    generated: str
    pass4_envelope: Optional[dict[str, Any]]
    matches_level1: Optional[bool]


def run_lookup_level_benchmark(
    model: Any,
    tokenizer: Any,
    queries: list[LookupLevelQuery],
    system_prompt: str,
    deployment_context: str,
    decode_fn: Callable[[Any, Any, list[dict], int], str],
    max_new_tokens: int = 400,
    on_result: Optional[Callable[[LookupLevelResult], None]] = None,
) -> list[LookupLevelResult]:
    """Measure whether the model's PASS_4 answer holds up as the SAME lookup
    is phrased with progressively less explicit, more natural language
    (Level 1: explicit entity + explicit identifier, ... Level 4: natural
    operator language, entity and identifier both implicit).

    Deliberately does NOT compare against a hand-authored "gold" PASS_4 for
    every query: this project's fine-tuning only covers phase-1 (simple
    lookups), and hand-guessing the correct qualifier-attribute name/entity
    resolution for phrasing this project's own dataset pipeline didn't
    generate risks silently encoding a WRONG answer as ground truth —
    exactly what this project's evidence-first approach exists to avoid
    (see README). Instead, within each group_id, Level 1's answer — the
    easiest, most template-like phrasing, closest to the training
    distribution — becomes that group's OWN reference; every other level
    in the group is compared against it, not against an external gold.
    This measures the thing that actually matters for a phase-1 model:
    does correctness degrade as phrasing gets less explicit, without
    requiring an independently-verified ground truth this project doesn't
    have for arbitrary hand-written queries.

    A group missing a valid Level 1 (absent, or its PASS_4 didn't parse)
    leaves every other level in that group with matches_level1=None —
    "not scored," never silently counted as a match or a miss.

    on_result, if given, is called with each LookupLevelResult immediately
    after it's produced (not batched at the end) — for a long-running
    remote benchmark (scripts/run_remote_lookup_benchmark.py), this lets a
    caller persist results to a log file as they arrive rather than losing
    everything generated so far if a later query in the run fails/hangs.
    """
    groups: dict[str, list[LookupLevelQuery]] = {}
    for q in queries:
        groups.setdefault(q.group_id, []).append(q)

    results: list[LookupLevelResult] = []
    for group_id, group_queries in groups.items():
        ordered = sorted(group_queries, key=lambda q: q.level)
        level1_envelope: Optional[dict[str, Any]] = None
        seen_level1 = False
        for q in ordered:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": deployment_context},
                {"role": "user", "content": f"## Query\n{q.query}"},
            ]
            generated = decode_fn(model, tokenizer, messages, max_new_tokens)
            envelope = parse_pass4_envelope(generated)

            if q.level == 1:
                level1_envelope = envelope
                seen_level1 = True
                matches_level1 = None
            elif not seen_level1 or level1_envelope is None:
                matches_level1 = None
            else:
                matches_level1 = envelope == level1_envelope

            result = LookupLevelResult(
                group_id=group_id,
                level=q.level,
                level_name=q.level_name,
                query=q.query,
                generated=generated,
                pass4_envelope=envelope,
                matches_level1=matches_level1,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


def summarize_lookup_level_results(results: list[LookupLevelResult]) -> dict[int, dict[str, Any]]:
    """Per-level consistency-with-Level-1 rate, aggregated across all groups."""
    by_level: dict[int, list[LookupLevelResult]] = {}
    for r in results:
        by_level.setdefault(r.level, []).append(r)

    summary: dict[int, dict[str, Any]] = {}
    for level, level_results in sorted(by_level.items()):
        if level == 1:
            summary[level] = {
                "level_name": level_results[0].level_name,
                "num_queries": len(level_results),
                "note": "reference level — each group's own baseline, not scored against itself",
            }
            continue
        scoreable = [r for r in level_results if r.matches_level1 is not None]
        matches = sum(1 for r in scoreable if r.matches_level1)
        summary[level] = {
            "level_name": level_results[0].level_name,
            "num_queries": len(level_results),
            "num_scoreable": len(scoreable),
            "consistency_with_level1": (matches / len(scoreable)) if scoreable else 0.0,
        }
    return summary
