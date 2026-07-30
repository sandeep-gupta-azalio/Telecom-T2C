"""Tests for src/remote_client.py's stdlib-only SSE client.

Monkeypatches urllib.request.urlopen with a fake context manager yielding
canned SSE byte-lines, and src.remote_client.time.monotonic with a fixed
sequence, so TTFT/total/tokens-per-second are deterministic without a real
server or real wall-clock timing.
"""

import json

import pytest

from src.remote_client import RemoteGenerationMetrics, generate_ollama, generate_remote, summarize_remote_metrics


class _FakeSSEResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return iter(self._lines)

    def __exit__(self, exc_type, exc, tb):
        return False


def _sse_lines(*events) -> list[bytes]:
    lines = []
    for event in events:
        if event == "[DONE]":
            lines.append(b"data: [DONE]\n")
        else:
            lines.append(("data: " + json.dumps(event) + "\n").encode("utf-8"))
    return lines


class TestGenerateRemote:
    def test_assembles_text_from_content_deltas_in_order(self, monkeypatch):
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}}], "usage": {"completion_tokens": 2}},
            "[DONE]",
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        text, metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])
        assert text == "Hello"
        assert metrics.tokens_generated == 2

    def test_ttft_total_and_tokens_per_second_from_controlled_clock(self, monkeypatch):
        # generate_remote calls time.monotonic() exactly 3 times: once at
        # start, once when the first content delta arrives (TTFT), once
        # after the loop ends (total). start=100.0, ttft=0.2, total=0.9.
        times = iter([100.0, 100.2, 100.9])
        monkeypatch.setattr("src.remote_client.time.monotonic", lambda: next(times))
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {"content": "b"}}]},
            {"choices": [{"delta": {}}], "usage": {"completion_tokens": 3}},
            "[DONE]",
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        _text, metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])

        assert metrics.ttft_seconds == pytest.approx(0.2)
        assert metrics.total_seconds == pytest.approx(0.9)
        # (tokens_generated - 1) / (total - ttft) = (3 - 1) / (0.9 - 0.2)
        assert metrics.tokens_per_second == pytest.approx(2 / 0.7)

    def test_ttft_falls_back_to_total_when_no_content_ever_arrives(self, monkeypatch):
        times = iter([100.0, 100.5])  # start, total (no ttft-triggering delta call)
        monkeypatch.setattr("src.remote_client.time.monotonic", lambda: next(times))
        lines = _sse_lines({"choices": [{"delta": {}}], "usage": {"completion_tokens": 0}}, "[DONE]")
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        text, metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])

        assert text == ""
        assert metrics.ttft_seconds == pytest.approx(0.5)
        assert metrics.tokens_per_second == 0.0

    def test_single_token_gives_zero_tokens_per_second_not_a_division_error(self, monkeypatch):
        times = iter([100.0, 100.1, 100.1])
        monkeypatch.setattr("src.remote_client.time.monotonic", lambda: next(times))
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "a"}}]},
            {"choices": [{"delta": {}}], "usage": {"completion_tokens": 1}},
            "[DONE]",
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        _text, metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])
        assert metrics.tokens_per_second == 0.0

    def test_ignores_lines_that_are_not_sse_data_events(self, monkeypatch):
        lines = [
            b"\n",
            b": comment\n",
            b"data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}).encode("utf-8") + b"\n",
            b"data: [DONE]\n",
        ]
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        text, _metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])
        assert text == "ok"

    def test_stops_consuming_at_done_even_if_more_lines_follow(self, monkeypatch):
        lines = _sse_lines(
            {"choices": [{"delta": {"content": "a"}}]},
            "[DONE]",
        ) + _sse_lines({"choices": [{"delta": {"content": "should not be read"}}]})
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        text, _metrics = generate_remote("http://fake", "token", [{"role": "user", "content": "hi"}])
        assert text == "a"


def _ollama_lines(*events) -> list[bytes]:
    return [(json.dumps(event) + "\n").encode("utf-8") for event in events]


class TestGenerateOllama:
    def test_assembles_text_from_message_content_in_order(self, monkeypatch):
        lines = _ollama_lines(
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 2, "eval_duration": 500_000_000},
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        text, metrics = generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert text == "Hello"
        assert metrics.tokens_generated == 2

    def test_uses_exact_eval_count_and_eval_duration_for_tokens_per_second(self, monkeypatch):
        # 10 tokens / 2s (eval_duration in nanoseconds) = 5 tok/s exactly,
        # server-reported — not the client-side wall-clock estimate.
        lines = _ollama_lines(
            {"message": {"content": "a"}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 10, "eval_duration": 2_000_000_000},
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        _text, metrics = generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert metrics.tokens_per_second == pytest.approx(5.0)

    def test_falls_back_to_wallclock_estimate_when_eval_duration_missing(self, monkeypatch):
        times = iter([100.0, 100.2, 100.9])
        monkeypatch.setattr("src.remote_client.time.monotonic", lambda: next(times))
        lines = _ollama_lines(
            {"message": {"content": "a"}, "done": False},
            {"message": {"content": "b"}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 3},
        )
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        _text, metrics = generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert metrics.ttft_seconds == pytest.approx(0.2)
        assert metrics.tokens_per_second == pytest.approx(2 / 0.7)

    def test_missing_done_line_fields_give_zero_tokens_not_a_crash(self, monkeypatch):
        lines = _ollama_lines({"message": {"content": "a"}, "done": True})
        monkeypatch.setattr(
            "src.remote_client.urllib.request.urlopen", lambda request, timeout: _FakeSSEResponse(lines)
        )
        _text, metrics = generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert metrics.tokens_generated == 0
        assert metrics.tokens_per_second == 0.0

    def test_includes_num_predict_option_when_max_tokens_given(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeSSEResponse(_ollama_lines({"message": {"content": ""}, "done": True}))

        monkeypatch.setattr("src.remote_client.urllib.request.urlopen", fake_urlopen)
        generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}], max_tokens=128)
        assert captured["body"]["options"] == {"num_predict": 128}

    def test_omits_options_when_max_tokens_not_given(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeSSEResponse(_ollama_lines({"message": {"content": ""}, "done": True}))

        monkeypatch.setattr("src.remote_client.urllib.request.urlopen", fake_urlopen)
        generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert "options" not in captured["body"]

    def test_sends_no_authorization_header(self, monkeypatch):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["headers"] = dict(request.header_items())
            return _FakeSSEResponse(_ollama_lines({"message": {"content": ""}, "done": True}))

        monkeypatch.setattr("src.remote_client.urllib.request.urlopen", fake_urlopen)
        generate_ollama("http://localhost:11434", [{"role": "user", "content": "hi"}])
        assert "Authorization" not in captured["headers"]


class TestSummarizeRemoteMetrics:
    def test_empty_list_returns_empty_dict(self):
        assert summarize_remote_metrics([]) == {}

    def test_averages_across_multiple_queries(self):
        metrics = [
            RemoteGenerationMetrics(ttft_seconds=0.2, total_seconds=1.0, tokens_generated=10, tokens_per_second=5.0),
            RemoteGenerationMetrics(ttft_seconds=0.4, total_seconds=2.0, tokens_generated=20, tokens_per_second=15.0),
        ]
        summary = summarize_remote_metrics(metrics)
        assert summary["num_queries"] == 2
        assert summary["avg_ttft_seconds"] == pytest.approx(0.3)
        assert summary["avg_total_seconds"] == pytest.approx(1.5)
        assert summary["avg_tokens_per_second"] == pytest.approx(10.0)
        assert summary["total_tokens_generated"] == 30
