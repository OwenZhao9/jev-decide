"""The ``jev`` backend: TypeSafe's System One HTTP API.

Implemented against the public documentation at https://docs.typesafe.ai (API
reference: https://docs.typesafe.ai/api), which specifies the endpoint, auth header,
request body and answer shapes used below::

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <API_KEY>
    Content-Type: application/json

    {"state": ..., "model": "jev-latest",
     "questions": {"<id>": {"type": "choice", "instructions": ..., "criteria": {...}}}}

Nothing here is guessed.  Without an API key the backend reports itself unavailable
and ``auto`` skips it -- it never fabricates an endpoint or an answer.
"""

from __future__ import annotations

import math
import os
from typing import Any

from .._http import HttpFailure, post_json
from .._math import clamp, confidence_from_probs, normalise_probs
from .base import BackendUnavailable, ChoiceRequest, ChoiceResult, ScoreRequest, ScoreResult

__all__ = ["JevBackend"]

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"
API_KEY_ENV = "TYPESAFE_API_KEY"
BASE_URL_ENV = "TYPESAFE_BASE_URL"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"

_QUESTION_ID = "decision"

# TypeSafe Score takes an ordered array of level descriptions (2..10 levels) and answers
# with a probability-weighted index across them.  Five generic levels cover an arbitrary
# caller rubric; the index is mapped linearly onto the caller's [lo, hi].
SCORE_LEVELS: tuple[str, ...] = (
    "Not at all",
    "Slightly",
    "Moderately",
    "Largely",
    "Completely",
)


def _endpoint(base_url: str) -> str:
    """Build the System One URL, tolerating a base that already carries the path."""
    base = base_url.rstrip("/")
    if base.endswith("/v1/systemone"):
        return base
    if base.endswith("/v1"):
        return f"{base}/systemone"
    return f"{base}/v1/systemone"


class JevBackend:
    """TypeSafe System One over HTTPS.  Blocks up to the timeout it is handed."""

    name = "jev"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        self._base_url = (base_url or os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).strip()
        self._model = (model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL).strip()

    @property
    def endpoint(self) -> str:
        """The resolved System One URL this backend will POST to."""
        return _endpoint(self._base_url)

    @property
    def model(self) -> str:
        """The resolved TypeSafe model name."""
        return self._model

    def availability(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, f"no API key (pass api_key=... or set {API_KEY_ENV})"
        return True, f"configured for {self.endpoint} model={self._model}"

    # -- requests ---------------------------------------------------------------

    def _ask(self, question: dict[str, Any], state: Any, timeout_s: float) -> dict[str, Any]:
        usable, reason = self.availability()
        if not usable:
            raise BackendUnavailable(reason)

        payload = {"state": state, "model": self._model, "questions": {_QUESTION_ID: question}}
        try:
            body = post_json(
                self.endpoint,
                payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout_s=timeout_s,
            )
        except HttpFailure as exc:
            raise BackendUnavailable(str(exc)) from exc

        answers = body.get("answers")
        if not isinstance(answers, dict) or _QUESTION_ID not in answers:
            raise BackendUnavailable("TypeSafe response carried no answer for the question")
        answer = answers[_QUESTION_ID]
        if not isinstance(answer, dict):
            raise BackendUnavailable("TypeSafe answer was not a JSON object")
        return {"answer": answer, "model": body.get("model"), "usage": body.get("usage")}

    def choice(self, request: ChoiceRequest, timeout_s: float) -> ChoiceResult:
        question = {
            "type": "choice",
            "instructions": request.question,
            # Documented: "a map of option to rubric description; use null when an option
            # needs no extra detail."  The caller gave us bare option names, so: null.
            "criteria": dict.fromkeys(request.options),
        }
        envelope = self._ask(question, request.state, timeout_s)
        answer = envelope["answer"]

        probs, usable = normalise_probs(answer.get("probabilities"), request.options)
        picked = answer.get("choice")
        note = ""
        degraded = False

        if not isinstance(picked, str) or picked not in request.options:
            if not usable:
                raise BackendUnavailable(
                    f"TypeSafe returned choice={picked!r}, which is not one of "
                    f"{list(request.options)}, and no usable probabilities"
                )
            picked = max(probs, key=lambda option: probs[option])
            note = "TypeSafe returned an unknown option; fell back to the highest probability"
            degraded = True

        if not usable:
            probs = {option: (1.0 if option == picked else 0.0) for option in request.options}
            note = "TypeSafe returned no usable probability distribution"
            degraded = True
            confidence = 0.0
        else:
            confidence = _confidence(answer, probs)

        return ChoiceResult(
            value=picked,
            probs=probs,
            confidence=confidence,
            raw=_raw(envelope),
            note=note,
            degraded=degraded,
        )

    def score(self, request: ScoreRequest, timeout_s: float) -> ScoreResult:
        question = {
            "type": "score",
            "instructions": request.rubric,
            "criteria": list(SCORE_LEVELS),
        }
        envelope = self._ask(question, request.state, timeout_s)
        answer = envelope["answer"]

        level = answer.get("score")
        numeric = not isinstance(level, bool) and isinstance(level, (int, float))
        if not numeric or not math.isfinite(level):
            raise BackendUnavailable(f"TypeSafe returned score={level!r}, expected a number")

        top = len(SCORE_LEVELS) - 1
        fraction = clamp(float(level) / top, 0.0, 1.0)
        value = request.lo + fraction * (request.hi - request.lo)

        level_keys = [str(index) for index in range(len(SCORE_LEVELS))]
        probs, usable = normalise_probs(answer.get("probabilities"), level_keys)
        confidence = _confidence(answer, probs) if usable else 0.0
        note = "" if usable else "TypeSafe returned no usable probability distribution"

        return ScoreResult(
            value=clamp(value, request.lo, request.hi),
            confidence=confidence,
            raw=_raw(envelope),
            note=note,
            degraded=not usable,
        )


def _confidence(answer: dict[str, Any], probs: dict[str, float]) -> float:
    """Prefer TypeSafe's own ``confidence``; fall back to the distribution's shape."""
    reported = answer.get("confidence")
    if not isinstance(reported, bool) and isinstance(reported, (int, float)):
        number = float(reported)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            return number
    return confidence_from_probs(probs)


def _raw(envelope: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {"answer": envelope["answer"]}
    if envelope.get("model") is not None:
        raw["model"] = envelope["model"]
    if envelope.get("usage") is not None:
        raw["usage"] = envelope["usage"]
    return raw
