"""HTTP clients for scripts/run_remote_lookup_benchmark.py: one for
src/server.py's hosted /chat/completions endpoint (see README "Testing the
adapter locally") — a Colab/Kaggle-hosted, ngrok-tunneled server — and one
for a model already `ollama create`'d locally (see README "Running the
fine-tuned model in Ollama"). Either way, generation happens on a server
this machine talks to over HTTP, not locally — no GPU/model needed here.

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


def generate_ollama(
    base_url: str,
    messages: list[dict],
    model: str = "t2c-gemma4",
    max_tokens: Optional[int] = None,
    think: bool = False,
    timeout: float = 120.0,
) -> tuple[str, RemoteGenerationMetrics]:
    """Like generate_remote, but speaks Ollama's own streaming POST /api/chat
    shape instead of the OpenAI-compatible SSE shape src/server.py exposes —
    for benchmarking a model already `ollama create`'d locally (see README
    "Running the fine-tuned model in Ollama"). No bearer token: Ollama has
    no built-in auth, unlike the ngrok server generate_remote() targets.

    think=False (the default) explicitly disables Ollama's "thinking" mode
    via the request's top-level "think" field. Added after a real run
    against an Ollama-served t2c-gemma4 with think left unset: every query
    hit exactly max_tokens per eval_count, yet message.content was empty
    or cut off mid-PASS_1/PASS_2 — i.e. the reported token budget was
    fully spent on something never reaching this function's returned text
    at all. The leading suspect is Ollama auto-detecting this as a
    thinking-capable model and routing reasoning tokens to a separate
    message.thinking field instead of message.content (not independently
    confirmed by inspecting that field here, but consistent with the
    symptom and with Ollama's documented behavior for thinking-capable
    models) — the same google/gemma-4-12B-it thinking-channel failure
    mode inference.build_prompt() already documents and avoids for the
    local/adapter path, where training never demonstrates continuing from
    that channel, only from a bare PASS_0-4 continuation. Pass think=True
    only to deliberately compare against that behavior.

    Ollama's response is newline-delimited plain JSON (no "data: " prefix,
    no [DONE] sentinel — the final line instead carries "done": true), and
    that final line reports exact prompt/decode token counts and durations
    the serving stack actually measured (eval_count, eval_duration in
    nanoseconds) — used here for tokens_per_second directly instead of
    estimating it from chunk-arrival wall-clock timing the way generate_remote
    must, since this is ground truth from the server, not a client-side
    approximation. Falls back to the same wall-clock estimate generate_remote
    uses if an older Ollama build omits those fields.

    Raises urllib.error.HTTPError/URLError on failure, same as generate_remote.
    """
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True, "think": think}
    if max_tokens is not None:
        # Ollama's cap on generated tokens lives under options.num_predict,
        # not a top-level field the way OpenAI's shape has max_tokens.
        payload["options"] = {"num_predict": max_tokens}
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )

    start = time.monotonic()
    ttft: Optional[float] = None
    chunks: list[str] = []
    tokens_generated = 0
    eval_duration_seconds: Optional[float] = None

    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            event: dict[str, Any] = json.loads(line)
            delta = event.get("message", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = time.monotonic() - start
                chunks.append(delta)
            if event.get("done"):
                tokens_generated = event.get("eval_count", tokens_generated)
                eval_duration_ns = event.get("eval_duration")
                if eval_duration_ns:
                    eval_duration_seconds = eval_duration_ns / 1e9
                break

    total = time.monotonic() - start
    if ttft is None:
        ttft = total

    if eval_duration_seconds and tokens_generated:
        tokens_per_second = tokens_generated / eval_duration_seconds
    elif tokens_generated > 1 and total > ttft:
        tokens_per_second = (tokens_generated - 1) / (total - ttft)
    else:
        tokens_per_second = 0.0

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
