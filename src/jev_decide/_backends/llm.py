"""The ``llm`` backend: any OpenAI-compatible ``/chat/completions`` endpoint.

The contract for this backend is "schema in, schema out": the request pins
``response_format`` to a JSON schema so the model cannot answer in prose, and the
reply is parsed and validated before it becomes a :class:`Choice` or :class:`Score`.
If validation fails the request is repaired and retried once inside the same timeout
budget -- the schema/validate/retry loop that `instructor
<https://github.com/567-labs/instructor>`_ made the standard shape for this.

Confidence here is derived from the distribution the model returns, not from a
self-reported number: a model's own "I am 95% sure" is unanchored, while the shape of
a distribution over the options is at least the same quantity the ``jev`` backend
reports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any

from .._http import HttpFailure, post_json
from .._math import clamp, confidence_from_probs, normalise_probs
from .base import BackendUnavailable, ChoiceRequest, ChoiceResult, ScoreRequest, ScoreResult

__all__ = ["LlmBackend"]

_log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
API_KEY_ENV = "JEV_DECIDE_LLM_API_KEY"
API_KEY_ENV_FALLBACK = "OPENAI_API_KEY"
BASE_URL_ENV = "JEV_DECIDE_LLM_BASE_URL"
BASE_URL_ENV_FALLBACK = "OPENAI_BASE_URL"
MODEL_ENV = "JEV_DECIDE_LLM_MODEL"

# Mirrors the jev backend: a Score is a distribution over ordered levels, and the
# reported value is the probability-weighted index mapped onto the caller's [lo, hi].
SCORE_LEVELS: tuple[str, ...] = (
    "Not at all",
    "Slightly",
    "Moderately",
    "Largely",
    "Completely",
)

_SYSTEM_PROMPT = (
    "You are a typed decision function inside a program. You answer exactly one "
    "question about the given state and you reply with JSON matching the provided "
    "schema. No prose, no explanation, no markdown. Probabilities must be "
    "non-negative and sum to 1. Spread the probability mass honestly: if the state "
    "does not settle the question, say so with a flat distribution instead of "
    "picking a winner."
)

_MIN_RETRY_BUDGET_S = 0.05


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class LlmBackend:
    """OpenAI-compatible chat completions with a JSON schema pinned to the answer."""

    name = "llm"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        env_key = os.environ.get(API_KEY_ENV) or os.environ.get(API_KEY_ENV_FALLBACK)
        env_base = os.environ.get(BASE_URL_ENV) or os.environ.get(BASE_URL_ENV_FALLBACK)
        self._api_key = (api_key or env_key or "").strip()
        self._base_url = (base_url or env_base or DEFAULT_BASE_URL).strip()
        self._model = (model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL).strip()

    @property
    def endpoint(self) -> str:
        """The resolved chat-completions URL this backend will POST to."""
        return _endpoint(self._base_url)

    @property
    def model(self) -> str:
        """The resolved model name."""
        return self._model

    def availability(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, (
                f"no API key (pass api_key=... or set {API_KEY_ENV} / {API_KEY_ENV_FALLBACK})"
            )
        return True, f"configured for {self.endpoint} model={self._model}"

    # -- requests ---------------------------------------------------------------

    def _complete(
        self,
        user_content: str,
        schema: dict[str, Any],
        schema_name: str,
        timeout_s: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call the endpoint, retrying once if the reply does not validate."""
        usable, reason = self.availability()
        if not usable:
            raise BackendUnavailable(reason)

        started = time.monotonic()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        last_problem = ""

        for attempt in (1, 2):
            remaining = timeout_s - (time.monotonic() - started)
            if remaining <= 0.0 or (attempt == 2 and remaining < _MIN_RETRY_BUDGET_S):
                break

            payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
            }
            try:
                body = post_json(
                    self.endpoint,
                    payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout_s=remaining,
                )
            except HttpFailure as exc:
                raise BackendUnavailable(str(exc)) from exc

            content = _extract_content(body)
            if content is None:
                last_problem = "the response carried no message content"
            else:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    last_problem = f"the reply was not valid JSON ({exc})"
                else:
                    if isinstance(parsed, dict):
                        return parsed, body
                    kind = type(parsed).__name__
                    last_problem = f"the reply was a JSON {kind}, expected an object"

            _log.debug("llm attempt %s rejected: %s", attempt, last_problem)
            messages = [
                *messages,
                {"role": "assistant", "content": content or ""},
                {
                    "role": "user",
                    "content": (
                        f"That reply was rejected: {last_problem}. "
                        "Reply again with JSON only, matching the schema exactly."
                    ),
                },
            ]

        raise BackendUnavailable(f"model did not return a usable JSON answer: {last_problem}")

    def choice(self, request: ChoiceRequest, timeout_s: float) -> ChoiceResult:
        options = list(request.options)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["choice", "probabilities"],
            "properties": {
                "choice": {"type": "string", "enum": options},
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": options,
                    "properties": {option: {"type": "number"} for option in options},
                },
            },
        }
        user_content = _render(
            {"state": request.state, "question": request.question, "options": options}
        )
        parsed, body = self._complete(user_content, schema, "jev_decide_choice", timeout_s)

        probs, usable = normalise_probs(parsed.get("probabilities"), request.options)
        picked = parsed.get("choice")
        note = ""
        degraded = False

        if not isinstance(picked, str) or picked not in request.options:
            if not usable:
                raise BackendUnavailable(
                    f"model returned choice={picked!r}, which is not one of {options}, "
                    "and no usable probabilities"
                )
            picked = max(probs, key=lambda option: probs[option])
            note = "model returned an unknown option; fell back to the highest probability"
            degraded = True

        if not usable:
            probs = {option: (1.0 if option == picked else 0.0) for option in request.options}
            note = "model returned no usable probability distribution"
            degraded = True
            confidence = 0.0
        else:
            confidence = confidence_from_probs(probs)

        return ChoiceResult(
            value=picked,
            probs=probs,
            confidence=confidence,
            raw=_raw(parsed, body),
            note=note,
            degraded=degraded,
        )

    def score(self, request: ScoreRequest, timeout_s: float) -> ScoreResult:
        level_keys = [str(index) for index in range(len(SCORE_LEVELS))]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["probabilities"],
            "properties": {
                "probabilities": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": level_keys,
                    "properties": {key: {"type": "number"} for key in level_keys},
                }
            },
        }
        user_content = _render(
            {
                "state": request.state,
                "rubric": request.rubric,
                "levels": {key: SCORE_LEVELS[int(key)] for key in level_keys},
                "instruction": (
                    "Give the probability that each level is the right description of "
                    "the state with respect to `rubric`."
                ),
            }
        )
        parsed, body = self._complete(user_content, schema, "jev_decide_score", timeout_s)

        probs, usable = normalise_probs(parsed.get("probabilities"), level_keys)
        if not usable:
            raise BackendUnavailable("model returned no usable probability distribution")

        top = len(SCORE_LEVELS) - 1
        expected_index = math.fsum(int(key) * probs[key] for key in level_keys)
        fraction = clamp(expected_index / top, 0.0, 1.0)
        value = request.lo + fraction * (request.hi - request.lo)

        return ScoreResult(
            value=clamp(value, request.lo, request.hi),
            confidence=confidence_from_probs(probs),
            raw=_raw(parsed, body),
        )


def _render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=False)


def _extract_content(body: dict[str, Any]) -> str | None:
    """Pull the assistant message text out of an OpenAI-compatible response."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some gateways return content as a list of typed parts.
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        joined = "".join(parts)
        return joined or None
    return None


def _raw(parsed: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {"parsed": parsed}
    for key in ("model", "id", "usage"):
        if body.get(key) is not None:
            raw[key] = body[key]
    return raw
