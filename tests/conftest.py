"""Shared fixtures.

Every HTTP test goes through :class:`MockHTTP`, which replaces
``urllib.request.urlopen``.  No test in this repository opens a socket: the real
``urllib`` code path in ``jev_decide._http`` is exercised, but nothing leaves the
process.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest


class FakeResponse:
    """The slice of ``http.client.HTTPResponse`` that ``_http.post_json`` touches."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, amount: int | None = None) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class RecordedRequest:
    """What a backend actually put on the wire."""

    def __init__(self, request: urllib.request.Request, timeout: float | None) -> None:
        self.url = request.full_url
        self.method = request.get_method()
        self.headers = {key.lower(): value for key, value in request.headers.items()}
        self.timeout = timeout
        raw = request.data or b""
        self.body: dict[str, Any] = json.loads(raw.decode("utf-8")) if raw else {}


class MockHTTP:
    """A scripted stand-in for ``urllib.request.urlopen``.

    Handlers are consumed in order; the last one is reused once the queue runs dry, so
    a test that only cares about one response does not have to enqueue three.
    """

    def __init__(self) -> None:
        self.requests: list[RecordedRequest] = []
        self._handlers: list[Callable[[RecordedRequest], Any]] = []

    # -- scripting ---------------------------------------------------------------

    def push(self, handler: Callable[[RecordedRequest], Any]) -> MockHTTP:
        self._handlers.append(handler)
        return self

    def json(self, body: dict[str, Any], status: int = 200) -> MockHTTP:
        return self.push(lambda _req: FakeResponse(status, json.dumps(body).encode("utf-8")))

    def text(self, body: str, status: int = 200) -> MockHTTP:
        return self.push(lambda _req: FakeResponse(status, body.encode("utf-8")))

    def http_error(self, code: int, body: str = "") -> MockHTTP:
        def handler(_req: RecordedRequest) -> Any:
            raise urllib.error.HTTPError(
                _req.url, code, "error", {}, io.BytesIO(body.encode("utf-8"))
            )

        return self.push(handler)

    def unreachable(self, reason: str = "connection refused") -> MockHTTP:
        def handler(_req: RecordedRequest) -> Any:
            raise urllib.error.URLError(reason)

        return self.push(handler)

    def hangs(self) -> MockHTTP:
        """Sleep for the whole timeout, then fail the way a real socket timeout does."""

        def handler(req: RecordedRequest) -> Any:
            time.sleep(max(req.timeout or 0.0, 0.0))
            raise TimeoutError("timed out")

        return self.push(handler)

    # -- the urlopen replacement ---------------------------------------------------

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        recorded = RecordedRequest(request, timeout)
        self.requests.append(recorded)
        if not self._handlers:
            raise AssertionError(f"unexpected HTTP call to {recorded.url}")
        handler = self._handlers.pop(0) if len(self._handlers) > 1 else self._handlers[0]
        return handler(recorded)

    # -- assertions ----------------------------------------------------------------

    @property
    def last(self) -> RecordedRequest:
        assert self.requests, "no HTTP request was made"
        return self.requests[-1]

    @property
    def count(self) -> int:
        return len(self.requests)


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> MockHTTP:
    """Patch ``urllib.request.urlopen`` for the duration of a test."""
    mock = MockHTTP()
    monkeypatch.setattr(urllib.request, "urlopen", mock)
    return mock


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every test independent of whatever keys the developer has exported."""
    for name in (
        "TYPESAFE_API_KEY",
        "TYPESAFE_BASE_URL",
        "TYPESAFE_DEFAULT_MODEL",
        "JEV_DECIDE_LLM_API_KEY",
        "JEV_DECIDE_LLM_BASE_URL",
        "JEV_DECIDE_LLM_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def jev_choice_body(
    choice: str,
    probabilities: dict[str, float],
    confidence: float | None = None,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """A TypeSafe System One choice response, shaped as docs.typesafe.ai/api documents."""
    answer: dict[str, Any] = {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities,
    }
    if confidence is not None:
        answer["confidence"] = confidence
    return {
        "model": model,
        "answers": {"decision": answer},
        "usage": {"input_tokens": 318, "output_tokens": 34},
    }


def jev_score_body(
    score: float,
    probabilities: dict[str, float],
    confidence: float | None = None,
) -> dict[str, Any]:
    """A TypeSafe System One score response."""
    answer: dict[str, Any] = {
        "type": "score",
        "score": score,
        "legend": {"0": "Not at all", "4": "Completely"},
        "probabilities": probabilities,
    }
    if confidence is not None:
        answer["confidence"] = confidence
    return {"model": "jev-1.13.0", "answers": {"decision": answer}, "usage": {}}


def llm_body(payload: dict[str, Any], model: str = "gpt-4o-mini") -> dict[str, Any]:
    """An OpenAI-compatible chat-completions response carrying JSON content."""
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(payload)},
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30},
    }
