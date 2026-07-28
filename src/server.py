"""Minimal HTTP inference server, meant to be tunneled out via ngrok for local testing.

Not part of the training path — this exists purely so a Colab-hosted GPU can
serve the fine-tuned adapter to a request made from the developer's own PC,
which has no GPU capable of running a 12B model (see README "Testing the
adapter locally"). Bearer-token-gated since ngrok URLs are public: anyone
with the URL can otherwise reach /generate.
"""

import json
import secrets
import time
from typing import Any, Iterator, Optional

from src import utils

logger = utils.get_logger("server")

# fastapi/pydantic imported at module level, not lazily like trl/unsloth
# elsewhere in src/ — needed here because Message/GenerateRequest/
# GenerateResponse must be real module-level classes for FastAPI's request
# parsing to work at all: FastAPI resolves route parameter annotations via
# typing.get_type_hints() against the endpoint function's __globals__, which
# can't see a class defined inside another function's local scope. Defining
# them locally inside build_app() (an earlier version of this file did)
# silently broke body parsing — FastAPI fell back to treating the body
# model as a query parameter instead of erroring loudly. Both packages are
# also serving-only dependencies already required at notebook runtime by
# the time this module is imported (see the inference-server notebook's
# Install section), so there's no CPU-test-collection cost to paying for
# them at import time the way there would be for trl/unsloth.
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: Optional[int] = None


class GenerateResponse(BaseModel):
    generated_text: str
    elapsed_seconds: float


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    max_tokens: Optional[int] = None
    # Accepted for OpenAI-client compatibility but otherwise ignored —
    # inference.generate always greedy-decodes (see its own docstring for
    # why: a documented Gemma+PEFT model.generate() workaround), so there
    # is no sampling temperature to honor here.
    temperature: Optional[float] = None
    # When true, respond as an SSE stream of chat.completion.chunk events
    # (OpenAI's streaming shape) via inference.generate_stream, so a client
    # can measure real time-to-first-token instead of only total latency.
    stream: Optional[bool] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatCompletionChoice]


def generate_api_token() -> str:
    """Return a fresh random bearer token (printed once in the notebook, not stored)."""
    return secrets.token_urlsafe(24)


def build_app(model: Any, tokenizer: Any, api_token: str, default_max_new_tokens: int = 512) -> Any:
    """Build a FastAPI app exposing GET /health, POST /generate, and POST /chat/completions.

    Both generation routes require `Authorization: Bearer <api_token>`.

    /generate takes this project's own shape:
    `{"messages": [{"role": ..., "content": ...}, ...], "max_new_tokens": int?}`
    — the same prompt-turns list shape used throughout src/ (see
    inference.build_prompt).

    /chat/completions speaks OpenAI's chat-completions request/response
    shape instead, so an unmodified OpenAI-compatible client — notably the
    sibling t2c project's `execute_openai()` (LlmConfig(provider="openai",
    api_base=<this server's ngrok URL>)) — can talk to this server without
    any t2c-side changes. `OPENAI_API_KEY` on the client side must be set
    to this server's own bearer token (printed once when the notebook
    starts this server) — execute_openai() sends it as
    `Authorization: Bearer <OPENAI_API_KEY>`, which is exactly the header
    _check_auth() below expects; there's no real OpenAI account involved.

    Generation runs synchronously via inference.generate on whatever thread
    FastAPI dispatches the request to; this server is for one developer's
    manual testing, not concurrent production load, so no request
    queue/batching is implemented.
    """
    from src import inference

    app = FastAPI(title="Telecom-T2C-Trainer Inference")

    def _check_auth(authorization: Optional[str]) -> None:
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    def _run_generate(messages: list[Message], max_new_tokens: int) -> tuple[str, float]:
        if not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        message_dicts = [m.model_dump() for m in messages]
        start = time.monotonic()
        text = inference.generate(model, tokenizer, message_dicts, max_new_tokens=max_new_tokens)
        elapsed = time.monotonic() - start
        logger.info("Generated %d chars in %.1fs", len(text), elapsed)
        return text, elapsed

    def _stream_chat_completion_chunks(
        messages: list[Message], max_new_tokens: int, response_model_name: str
    ) -> Iterator[str]:
        """SSE body for stream=True: one chat.completion.chunk per non-empty
        delta, a final chunk carrying usage.completion_tokens, then [DONE] —
        mirrors OpenAI's streaming shape closely enough for an unmodified
        OpenAI-compatible client to consume (see build_app's own docstring
        for why /chat/completions exists at all).

        `messages` must already be validated non-empty by the caller — this
        function is a generator, so raising HTTPException from inside it
        would happen only once the ASGI framework starts iterating (after
        the 200 status/headers are already committed), which is too late
        for FastAPI to turn into a proper error response.
        """
        message_dicts = [m.model_dump() for m in messages]
        completion_id = f"t2c-{secrets.token_hex(8)}"
        start = time.monotonic()
        last_token_index = 0
        for delta, token_index in inference.generate_stream(
            model, tokenizer, message_dicts, max_new_tokens=max_new_tokens
        ):
            last_token_index = token_index
            if delta:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": response_model_name,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        elapsed = time.monotonic() - start
        logger.info("Streamed %d tokens in %.1fs", last_token_index, elapsed)
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": response_model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": last_token_index},
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate_endpoint(
        request: GenerateRequest, authorization: Optional[str] = Header(None)
    ) -> GenerateResponse:
        _check_auth(authorization)
        text, elapsed = _run_generate(request.messages, request.max_new_tokens or default_max_new_tokens)
        return GenerateResponse(generated_text=text, elapsed_seconds=elapsed)

    @app.post("/chat/completions", response_model=None)
    def chat_completions_endpoint(
        request: ChatCompletionRequest, authorization: Optional[str] = Header(None)
    ):
        _check_auth(authorization)
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")
        max_new_tokens = request.max_tokens or default_max_new_tokens
        response_model_name = request.model or "t2c-gemma4"

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion_chunks(request.messages, max_new_tokens, response_model_name),
                media_type="text/event-stream",
            )

        text, _elapsed = _run_generate(request.messages, max_new_tokens)
        return ChatCompletionResponse(
            id=f"t2c-{secrets.token_hex(8)}",
            model=response_model_name,
            choices=[
                ChatCompletionChoice(index=0, message=Message(role="assistant", content=text))
            ],
        )

    return app


def start_server(app: Any, port: int, ngrok_authtoken: Optional[str] = None, timeout_seconds: float = 15.0):
    """Start `app` with uvicorn on a background thread and open an ngrok tunnel to it.

    Returns (server, tunnel) — pass both to stop_server() to tear down
    cleanly. Runs uvicorn on its own thread (not the notebook's main thread)
    so the notebook cell returns immediately instead of blocking; uvicorn's
    own asyncio loop lives entirely on that thread, so it doesn't conflict
    with Colab/IPython's main-thread event loop.
    """
    import threading

    import uvicorn
    from pyngrok import ngrok

    if ngrok_authtoken:
        ngrok.set_auth_token(ngrok_authtoken)

    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout_seconds
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"uvicorn server did not start within {timeout_seconds}s")
        time.sleep(0.1)

    tunnel = ngrok.connect(port, "http")
    logger.info("Server live at %s (requires the printed bearer token)", tunnel.public_url)
    return server, tunnel


def stop_server(server: Any, tunnel: Any) -> None:
    """Tear down the ngrok tunnel and signal uvicorn's background thread to stop."""
    from pyngrok import ngrok

    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
    server.should_exit = True
