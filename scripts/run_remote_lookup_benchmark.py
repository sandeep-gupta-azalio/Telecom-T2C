#!/usr/bin/env python
"""Run the lookup-level robustness benchmark from your local PC against a
hosted inference server — no GPU/model needed locally, since generation
happens on whichever server this script talks to over HTTP. Two providers:

  --provider openai (default): a Colab/Kaggle-hosted, ngrok-tunneled
      src/server.py, via its streaming /chat/completions endpoint.
  --provider ollama: a model already `ollama create`'d on this same
      machine (or another one on your network), via Ollama's own
      streaming /api/chat endpoint — no bearer token needed.

See README "Running the lookup-level benchmark locally".

Usage (openai/ngrok):
    python scripts/run_remote_lookup_benchmark.py \\
        --base-url https://<your-subdomain>.ngrok-free.dev \\
        --api-token "$OPENAI_API_KEY"

Usage (ollama):
    python scripts/run_remote_lookup_benchmark.py --provider ollama

Every per-query result (generated text, PASS_4 parse, consistency-with-
Level-1, and performance: TTFT, total latency, tokens/sec) is appended to a
JSONL log file AS IT ARRIVES — a long run interrupted partway through still
leaves a usable partial record on disk, not just an in-memory list lost on
crash/Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluator  # noqa: E402
from src.lookup_level_queries import (  # noqa: E402
    DEFAULT_LOOKUP_LEVEL_QUERIES,
    LOOKUP_LEVEL_DEPLOYMENT_CONTEXT,
    LOOKUP_LEVEL_SYSTEM_PROMPT,
)
from src.remote_client import (  # noqa: E402
    RemoteGenerationMetrics,
    generate_ollama,
    generate_remote,
    summarize_remote_metrics,
)

_DEFAULT_BASE_URL_BY_PROVIDER = {"openai": None, "ollama": "http://localhost:11434"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="'openai': the ngrok-tunneled src/server.py. 'ollama': a locally ollama-created model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="e.g. https://<your-subdomain>.ngrok-free.dev for --provider openai. "
        "Defaults to http://localhost:11434 for --provider ollama.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("OPENAI_API_KEY") or os.environ.get("T2C_API_TOKEN"),
        help="Bearer token printed by the notebook's Start Server cell. Only used for "
        "--provider openai (Ollama has no built-in auth). Defaults to $OPENAI_API_KEY, "
        "then $T2C_API_TOKEN.",
    )
    parser.add_argument("--model", default="t2c-gemma4")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument(
        "--think",
        action="store_true",
        help="--provider ollama only: let Ollama's 'thinking' mode run instead of disabling it. "
        "Off by default because training never demonstrates continuing from that channel (see "
        "generate_ollama's docstring) — the whole max-tokens budget can otherwise be spent on "
        "invisible reasoning tokens instead of the trained PASS_0-4 answer.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-query HTTP timeout, in seconds.")
    parser.add_argument(
        "--log-file",
        default=None,
        help="JSONL path to append results to as they arrive. "
        "Defaults to logs/lookup_level_benchmark_<timestamp>.jsonl",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_url = args.base_url or _DEFAULT_BASE_URL_BY_PROVIDER[args.provider]
    if not base_url:
        print(f"error: --base-url is required for --provider {args.provider}", file=sys.stderr)
        return 2
    if args.provider == "openai" and not args.api_token:
        print(
            "error: no API token given (--api-token, or set OPENAI_API_KEY / T2C_API_TOKEN)",
            file=sys.stderr,
        )
        return 2

    log_path = (
        Path(args.log_file)
        if args.log_file
        else Path("logs") / f"lookup_level_benchmark_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging per-query results to {log_path}")

    # Populated by decode_fn, one entry per query, in the same order
    # run_lookup_level_benchmark calls decode_fn — on_result (fired
    # immediately after each call) always reads metrics_by_query[-1] for the
    # result it was just given.
    metrics_by_query: list[RemoteGenerationMetrics] = []

    def decode_fn(_model: object, _tokenizer: object, messages: list[dict], max_new_tokens: int) -> str:
        if args.provider == "ollama":
            text, metrics = generate_ollama(
                base_url,
                messages,
                model=args.model,
                max_tokens=max_new_tokens,
                think=args.think,
                timeout=args.timeout,
            )
        else:
            text, metrics = generate_remote(
                base_url,
                args.api_token,
                messages,
                max_tokens=max_new_tokens,
                model=args.model,
                timeout=args.timeout,
            )
        metrics_by_query.append(metrics)
        return text

    with log_path.open("a", encoding="utf-8") as log_file:

        def on_result(result: evaluator.LookupLevelResult) -> None:
            metrics = metrics_by_query[-1]
            record = {
                "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "group_id": result.group_id,
                "level": result.level,
                "level_name": result.level_name,
                "query": result.query,
                "generated": result.generated,
                "pass4_envelope": result.pass4_envelope,
                "matches_level1": result.matches_level1,
                "metrics": asdict(metrics),
            }
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()

            match_str = "(reference)" if result.level == 1 else str(result.matches_level1)
            print(
                f"  [{result.group_id}] L{result.level} {result.level_name!r}: {result.query!r} "
                f"-> matches L1: {match_str} | ttft={metrics.ttft_seconds:.2f}s "
                f"total={metrics.total_seconds:.2f}s tok/s={metrics.tokens_per_second:.1f}"
            )

        results = evaluator.run_lookup_level_benchmark(
            model=None,
            tokenizer=None,
            queries=DEFAULT_LOOKUP_LEVEL_QUERIES,
            system_prompt=LOOKUP_LEVEL_SYSTEM_PROMPT,
            deployment_context=LOOKUP_LEVEL_DEPLOYMENT_CONTEXT,
            decode_fn=decode_fn,
            max_new_tokens=args.max_tokens,
            on_result=on_result,
        )

    print()
    print("Checks covered (consistency with each group's own Level 1):")
    for level, stats in evaluator.summarize_lookup_level_results(results).items():
        print(f"  L{level} ({stats['level_name']}): {stats}")

    print()
    print("Performance:")
    for key, value in summarize_remote_metrics(metrics_by_query).items():
        print(f"  {key}: {value}")

    print()
    print(f"Full per-query log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
