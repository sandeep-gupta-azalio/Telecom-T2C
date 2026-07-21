"""Dataset statistics and token analysis.

Kept separate from dataset.py so dataset.py can call into this module without
a circular import (dataset -> statistics, never the reverse). All percentile
math is hand-rolled (not the stdlib `statistics` module) to avoid any
ambiguity with this module's own name once imported as `src.statistics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src import utils

logger = utils.get_logger("statistics")

# Rough, undocumented-by-benchmark tokens/sec estimates per GPU family for a
# ~12B-parameter model under 4-bit QLoRA. These are heuristic placeholders —
# see README "Troubleshooting" / plan Risk #10 — not measured throughput.
_TOKENS_PER_SEC_ESTIMATE: dict[str, float] = {
    "A100": 3000.0,
    "L4": 900.0,
    "T4": 250.0,
    "OTHER": 500.0,
    "CPU": 5.0,
}


@dataclass
class DatasetStatistics:
    split_name: str
    num_examples: int
    avg_turns: float
    avg_prompt_tokens: float
    avg_response_tokens: float
    avg_total_tokens: float
    p95_total_tokens: float
    max_total_tokens: int
    num_exceeding_max_seq_length: int
    packing_efficiency: Optional[float]
    estimated_optimizer_steps: Optional[int]
    estimated_training_time_hours: Optional[float]


@dataclass
class TokenAnalysis:
    mean: float
    median: float
    p95: float
    max: int
    num_exceeding_max_seq_length: int
    per_example_lengths: list[int]


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (pct / 100.0) * (len(sorted_values) - 1)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _get_messages(example: Any) -> list[dict]:
    if isinstance(example, dict):
        return example.get("messages", [])
    return list(example["messages"])


def _tokenize_example(tokenizer: Any, messages: list[dict]) -> tuple[int, int, int, int]:
    """Return (turns, prompt_tokens, response_tokens, total_tokens) for one conversation.

    prompt/response split is a per-message sum (system+user vs. assistant);
    total_tokens uses the tokenizer's chat template (what's actually fed to
    the model, including template markup overhead), falling back to
    prompt+response if the template application fails for any reason.
    """
    turns = len(messages)
    prompt_tokens = 0
    response_tokens = 0
    for message in messages:
        content = message.get("content", "") or ""
        n = len(tokenizer.encode(content, add_special_tokens=False))
        if message.get("role") == "assistant":
            response_tokens += n
        else:
            prompt_tokens += n
    try:
        total_tokens = len(
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        )
    except Exception:
        total_tokens = prompt_tokens + response_tokens
    return turns, prompt_tokens, response_tokens, total_tokens


def estimate_training_time(
    *,
    num_examples: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    packing: bool,
    avg_tokens: float,
    max_seq_length: int,
    gpu_family: str,
) -> float:
    """Rough heuristic estimate of total training wall-clock time, in hours.

    NOT based on a real Gemma-4 throughput benchmark (none exists yet) —
    uses a small per-GPU-family tokens/sec lookup table. Treat this as a
    ballpark, not a guarantee.
    """
    tokens_per_sec = _TOKENS_PER_SEC_ESTIMATE.get(gpu_family, _TOKENS_PER_SEC_ESTIMATE["OTHER"])
    if packing:
        total_tokens = num_examples * avg_tokens
    else:
        # Without packing, sequences are padded to max_seq_length in the worst case.
        total_tokens = num_examples * max_seq_length
    total_tokens_all_epochs = total_tokens * epochs
    seconds = total_tokens_all_epochs / max(tokens_per_sec, 1e-6)
    return seconds / 3600.0


def compute_dataset_statistics(
    dataset: Any,
    tokenizer: Any,
    max_seq_length: int,
    split_name: str,
    batch_size: int,
    gradient_accumulation: int,
    epochs: float,
    packing: bool,
    gpu_family: str,
) -> DatasetStatistics:
    """Tokenize an entire split and compute the statistics the spec requires.

    dataset is any iterable of {"messages": [...]} examples (a HF Dataset or
    a plain list both work).
    """
    num_examples = len(dataset)
    if num_examples == 0:
        return DatasetStatistics(
            split_name=split_name, num_examples=0, avg_turns=0.0, avg_prompt_tokens=0.0,
            avg_response_tokens=0.0, avg_total_tokens=0.0, p95_total_tokens=0.0,
            max_total_tokens=0, num_exceeding_max_seq_length=0, packing_efficiency=None,
            estimated_optimizer_steps=None, estimated_training_time_hours=None,
        )

    logger.info("Computing dataset statistics for '%s' (%d examples)...", split_name, num_examples)
    turns_list: list[int] = []
    prompt_list: list[int] = []
    response_list: list[int] = []
    total_list: list[int] = []
    for i, example in enumerate(dataset):
        messages = _get_messages(example)
        turns, prompt_tokens, response_tokens, total_tokens = _tokenize_example(tokenizer, messages)
        turns_list.append(turns)
        prompt_list.append(prompt_tokens)
        response_list.append(response_tokens)
        total_list.append(total_tokens)
        if (i + 1) % 5000 == 0:
            logger.info("  tokenized %d/%d (%s)", i + 1, num_examples, split_name)

    avg_turns = sum(turns_list) / num_examples
    avg_prompt = sum(prompt_list) / num_examples
    avg_response = sum(response_list) / num_examples
    avg_total = sum(total_list) / num_examples
    sorted_total = sorted(total_list)
    p95_total = _percentile(sorted_total, 95)
    max_total = max(total_list)
    num_exceeding = sum(1 for t in total_list if t > max_seq_length)

    packing_efficiency = (avg_total / max_seq_length) if (packing and max_seq_length) else None

    effective_batch = max(batch_size * gradient_accumulation, 1)
    if packing:
        approx_packed_sequences = max(int(sum(total_list) / max(max_seq_length, 1)), 1)
        steps_per_epoch = max(approx_packed_sequences // effective_batch, 1)
    else:
        steps_per_epoch = max(num_examples // effective_batch, 1)
    estimated_optimizer_steps = int(steps_per_epoch * epochs)

    estimated_hours = estimate_training_time(
        num_examples=num_examples,
        batch_size=batch_size,
        grad_accum=gradient_accumulation,
        epochs=epochs,
        packing=packing,
        avg_tokens=avg_total,
        max_seq_length=max_seq_length,
        gpu_family=gpu_family,
    )

    return DatasetStatistics(
        split_name=split_name,
        num_examples=num_examples,
        avg_turns=avg_turns,
        avg_prompt_tokens=avg_prompt,
        avg_response_tokens=avg_response,
        avg_total_tokens=avg_total,
        p95_total_tokens=p95_total,
        max_total_tokens=max_total,
        num_exceeding_max_seq_length=num_exceeding,
        packing_efficiency=packing_efficiency,
        estimated_optimizer_steps=estimated_optimizer_steps,
        estimated_training_time_hours=estimated_hours,
    )


def analyze_tokens(dataset: Any, tokenizer: Any, max_seq_length: int) -> TokenAnalysis:
    """Tokenize the full dataset (chat-template length) and report distribution stats.

    Logs a warning naming how many conversations exceed max_seq_length, per
    the spec's "warn if conversation exceeds max sequence length" requirement.
    """
    lengths: list[int] = []
    for example in dataset:
        messages = _get_messages(example)
        try:
            n = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        except Exception:
            n = sum(len(tokenizer.encode(m.get("content", "") or "", add_special_tokens=False)) for m in messages)
        lengths.append(n)

    if not lengths:
        return TokenAnalysis(mean=0.0, median=0.0, p95=0.0, max=0, num_exceeding_max_seq_length=0, per_example_lengths=[])

    sorted_lengths = sorted(lengths)
    mean = sum(lengths) / len(lengths)
    median = _percentile(sorted_lengths, 50)
    p95 = _percentile(sorted_lengths, 95)
    mx = max(lengths)
    exceeding = sum(1 for n in lengths if n > max_seq_length)

    if exceeding:
        logger.warning(
            "%d/%d conversations exceed max_seq_length=%d — they will be truncated by "
            "the trainer (or, if packing is enabled, may split awkwardly across packed blocks).",
            exceeding, len(lengths), max_seq_length,
        )

    return TokenAnalysis(
        mean=mean, median=median, p95=p95, max=mx,
        num_exceeding_max_seq_length=exceeding, per_example_lengths=lengths,
    )


def display_histogram(values: Sequence[float], bins: int = 20, width: int = 50, title: str = "") -> None:
    """Print a plain ASCII bucket histogram to stdout (no matplotlib dependency)."""
    if not values:
        print("(no data to histogram)")
        return
    lo, hi = min(values), max(values)
    if title:
        print(title)
    if lo == hi:
        print(f"  all {len(values)} values equal {lo}")
        return
    bucket_width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / bucket_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts) or 1
    for i, count in enumerate(counts):
        bucket_lo = lo + i * bucket_width
        bucket_hi = bucket_lo + bucket_width
        bar_len = int((count / max_count) * width)
        bar = "#" * bar_len
        print(f"  [{bucket_lo:8.0f}, {bucket_hi:8.0f}) {bar} {count}")


def print_statistics_report(stats: DatasetStatistics) -> None:
    """Pretty-print a DatasetStatistics to stdout."""
    print(f"=== Dataset statistics: {stats.split_name} ===")
    print(f"  examples:                 {stats.num_examples:,}")
    print(f"  avg turns/conversation:   {stats.avg_turns:.2f}")
    print(f"  avg prompt tokens:        {stats.avg_prompt_tokens:.1f}")
    print(f"  avg response tokens:      {stats.avg_response_tokens:.1f}")
    print(f"  avg total tokens:         {stats.avg_total_tokens:.1f}")
    print(f"  p95 total tokens:         {stats.p95_total_tokens:.1f}")
    print(f"  max total tokens:         {stats.max_total_tokens:,}")
    print(f"  exceeding max_seq_length: {stats.num_exceeding_max_seq_length:,}")
    if stats.packing_efficiency is not None:
        print(f"  packing efficiency:      {stats.packing_efficiency:.2%}")
    if stats.estimated_optimizer_steps is not None:
        print(f"  est. optimizer steps:     {stats.estimated_optimizer_steps:,}")
    if stats.estimated_training_time_hours is not None:
        print(
            f"  est. training time:      {stats.estimated_training_time_hours:.1f} h "
            "(rough heuristic, not a benchmark)"
        )
