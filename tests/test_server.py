"""Smoke tests for src/server.py's FastAPI app: auth gating and the generate endpoint.

Uses FastAPI's TestClient (no real uvicorn/ngrok/model) and monkeypatches
src.inference.generate so no real model/tokenizer is needed.
"""

import json

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed in this environment")

from src import inference, server


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(inference, "generate", lambda model, tokenizer, messages, max_new_tokens=512: "fake reply")

    app = server.build_app(model=object(), tokenizer=object(), api_token="secret-token")
    return TestClient(app)


class TestGenerateApiToken:
    def test_returns_nonempty_random_string(self):
        a = server.generate_api_token()
        b = server.generate_api_token()
        assert a and b
        assert a != b


class TestHealth:
    def test_health_ok_without_auth(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestGenerateEndpoint:
    def test_rejects_missing_authorization(self, client):
        response = client.post("/generate", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 401

    def test_rejects_wrong_token(self, client):
        response = client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_rejects_empty_messages(self, client):
        response = client.post(
            "/generate", json={"messages": []}, headers={"Authorization": "Bearer secret-token"}
        )
        assert response.status_code == 400

    def test_generates_with_valid_token(self, client):
        response = client.post(
            "/generate",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["generated_text"] == "fake reply"
        assert body["elapsed_seconds"] >= 0


class TestChatCompletionsEndpoint:
    """/chat/completions speaks OpenAI's shape so t2c's execute_openai() can
    talk to this server unmodified — LlmConfig(provider="openai",
    api_base=<this server's ngrok URL>), with OPENAI_API_KEY set to this
    server's own bearer token (execute_openai sends it as
    `Authorization: Bearer <OPENAI_API_KEY>`, matching _check_auth exactly).
    """

    def test_rejects_missing_authorization(self, client):
        response = client.post("/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 401

    def test_rejects_wrong_token(self, client):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_rejects_empty_messages(self, client):
        response = client.post(
            "/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer secret-token"}
        )
        assert response.status_code == 400

    def test_returns_openai_shaped_response(self, client):
        response = client.post(
            "/chat/completions",
            json={"model": "t2c-gemma4", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        body = response.json()
        # Exactly what src.t2c.ontology.llm._extract_openai_content reads —
        # this shape is the actual compatibility contract, not incidental.
        assert body["choices"][0]["message"]["content"] == "fake reply"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["model"] == "t2c-gemma4"

    def test_defaults_model_name_when_not_supplied(self, client):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.json()["model"] == "t2c-gemma4"

    def test_uses_max_tokens_field_not_max_new_tokens(self, monkeypatch, client):
        captured = {}

        def fake_generate(model, tokenizer, messages, max_new_tokens=512):
            captured["max_new_tokens"] = max_new_tokens
            return "fake reply"

        monkeypatch.setattr(inference, "generate", fake_generate)
        client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 128},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert captured["max_new_tokens"] == 128


class TestChatCompletionsStreamingEndpoint:
    """stream=True path: SSE chunks via inference.generate_stream, ending in
    a usage-bearing chunk + [DONE] (see remote_client.generate_remote, the
    client this shape exists for)."""

    def test_rejects_missing_authorization_before_streaming_starts(self, client):
        response = client.post(
            "/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "stream": True}
        )
        assert response.status_code == 401

    def test_rejects_empty_messages_before_streaming_starts(self, client):
        response = client.post(
            "/chat/completions",
            json={"messages": [], "stream": True},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 400

    def test_streams_sse_chunks_ending_in_usage_and_done(self, monkeypatch, client):
        def fake_generate_stream(model, tokenizer, messages, max_new_tokens=512):
            yield "Hel", 1
            yield "lo", 2
            yield "", 3  # stop-token step: empty delta, still carries the final token_index

        monkeypatch.setattr(inference, "generate_stream", fake_generate_stream)
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        blocks = [b for b in response.text.split("\n\n") if b.strip()]
        assert blocks[-1] == "data: [DONE]"

        events = [json.loads(b[len("data: ") :]) for b in blocks[:-1]]
        contents = [e["choices"][0]["delta"].get("content", "") for e in events]
        assert "".join(contents) == "Hello"

        final = events[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["usage"]["completion_tokens"] == 3

    def test_non_streaming_path_is_unaffected_when_stream_is_false(self, client):
        response = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
            headers={"Authorization": "Bearer secret-token"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["choices"][0]["message"]["content"] == "fake reply"
