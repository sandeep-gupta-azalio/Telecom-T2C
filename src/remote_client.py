"""HTTP client for talking to src/server.py's hosted /chat/completions endpoint
from a machine with no GPU/model of its own (see README "Testing the adapter
locally") — used by scripts/run_remote_lookup_benchmark.py to run the
lookup-level robustness benchmark from the developer's own PC against a
Colab/Kaggle-hosted, ngrok-tunneled server.

Uses only the standard library (urllib), not `requests` — this module is
meant to run on a bare local Python install with none of this project's
GPU/ML dependencies, so it deliberately adds none of its own.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from src import utils

logger = utils.get_logger("remote_client")


@dataclass
class RemoteGenerationMetrics:
    ttft_seconds: float
    total_seconds: float
    tokens_generated: int
    tokens_per_second: float


def generate_remote(
    base_url: str,
    api_token: str,
    messages: list[dict],
    max_tokens: int = 400,
    model: str = "t2c-gemma4",
    timeout: float = 120.0,
) -> tuple[str, RemoteGenerationMetrics]:
    """POST messages to {base_url}/chat/completions with stream=True and consume
    the SSE response, measuring time-to-first-token and steady-state
    tokens/sec directly from real wall-clock chunk arrivals (not estimated
    from total latency alone).

    tokens_per_second is computed over the decode phase only (excludes TTFT,
    which is dominated by prompt processing) — (tokens_generated - 1) /
    (total_seconds - ttft_seconds); it is 0.0 when only a single token was
    generated (no decode-phase interval to measure).

    Raises urllib.error.HTTPError (e.g. 401 on a wrong bearer token, 400 on
    empty messages) or urllib.error.URLError (e.g. connection refused/timed
    out) on failure — callers should catch these and report a clear message
    rather than letting a raw traceback surface.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
    )

    start = time.monotonic()
    ttft: Optional[float] = None
    chunks: list[str] = []
    tokens_generated = 0

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                break
            event: dict[str, Any] = json.loads(payload)
            choice = event["choices"][0]
            delta = choice.get("delta", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.monotonic() - start
                chunks.append(delta)
            usage = event.get("usage")
            if usage is not None:
                tokens_generated = usage.get("completion_tokens", tokens_generated)

    total = time.monotonic() - start
    if ttft is None:
        ttft = total

    tokens_per_second = 0.0
    if tokens_generated > 1 and total > ttft:
        tokens_per_second = (tokens_generated - 1) / (total - ttft)

    metrics = RemoteGenerationMetrics(
        ttft_seconds=ttft,
        total_seconds=total,
        tokens_generated=tokens_generated,
        tokens_per_second=tokens_per_second,
    )
    return "".join(chunks), metrics


def summarize_remote_metrics(metrics: list[RemoteGenerationMetrics]) -> dict[str, float]:
    """Aggregate per-query RemoteGenerationMetrics into averages for a run summary."""
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "num_queries": n,
        "avg_ttft_seconds": sum(m.ttft_seconds for m in metrics) / n,
        "avg_total_seconds": sum(m.total_seconds for m in metrics) / n,
        "avg_tokens_per_second": sum(m.tokens_per_second for m in metrics) / n,
        "total_tokens_generated": sum(m.tokens_generated for m in metrics),
    }
