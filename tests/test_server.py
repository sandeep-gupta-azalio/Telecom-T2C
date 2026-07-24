"""Smoke tests for src/server.py's FastAPI app: auth gating and the generate endpoint.

Uses FastAPI's TestClient (no real uvicorn/ngrok/model) and monkeypatches
src.inference.generate so no real model/tokenizer is needed.
"""

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
