"""The ``rules`` backend: the caller's own pure function.

Zero dependencies, no IO, no network, microseconds.  This is the backend that is
*always* available, which is what makes ``auto`` safe to call with no API key at all.

It is also the honest floor of the library: when there is no rules function to call,
it does not invent an answer -- it returns the first option with a flat distribution,
``confidence=0.0`` and ``degraded=True``, so that :meth:`Decider.gate` blocks by
default and nothing downstream moves on a guess.
"""

from __future__ import annotations

import logging
import math

from .._math import clamp, confidence_from_probs, uniform_probs
from .base import BackendUnavailable, ChoiceRequest, ChoiceResult, ScoreRequest, ScoreResult

__all__ = ["RulesBackend"]

_log = logging.getLogger(__name__)

NO_RULES_NOTE = (
    "no rules function was supplied, so the answer is a placeholder with "
    "confidence 0.0 -- gate() will refuse to act on it"
)


class RulesBackend:
    """Evaluate the caller-supplied pure function.  Never blocks, never uses the network."""

    name = "rules"

    def availability(self) -> tuple[bool, str]:
        return True, "always available (pure local function, no IO)"

    def choice(self, request: ChoiceRequest, timeout_s: float) -> ChoiceResult:
        options = request.options
        if request.rules is None:
            return ChoiceResult(
                value=options[0],
                probs=uniform_probs(options),
                confidence=0.0,
                raw=None,
                note=NO_RULES_NOTE,
                degraded=True,
            )

        try:
            picked = request.rules(request.state)
        except Exception as exc:
            _log.warning("rules function raised %s: %s", type(exc).__name__, exc)
            raise BackendUnavailable(
                f"rules function raised {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(picked, str) or picked not in options:
            raise BackendUnavailable(
                f"rules function returned {picked!r}, which is not one of {list(options)}"
            )

        probs = {option: (1.0 if option == picked else 0.0) for option in options}
        return ChoiceResult(
            value=picked,
            probs=probs,
            confidence=confidence_from_probs(probs),
            raw=None,
        )

    def score(self, request: ScoreRequest, timeout_s: float) -> ScoreResult:
        lo, hi = request.lo, request.hi
        if request.rules is None:
            return ScoreResult(
                value=(lo + hi) / 2.0,
                confidence=0.0,
                raw=None,
                note=NO_RULES_NOTE,
                degraded=True,
            )

        try:
            raw_value = request.rules(request.state)
        except Exception as exc:
            _log.warning("rules function raised %s: %s", type(exc).__name__, exc)
            raise BackendUnavailable(
                f"rules function raised {type(exc).__name__}: {exc}"
            ) from exc

        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise BackendUnavailable(
                f"rules function returned {raw_value!r}, expected a number in [{lo}, {hi}]"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise BackendUnavailable(f"rules function returned a non-finite value {raw_value!r}")

        clamped = clamp(value, lo, hi)
        note = ""
        if clamped != value:
            note = f"rules function returned {value}, clamped into [{lo}, {hi}]"
        return ScoreResult(value=clamped, confidence=1.0, raw=None, note=note)
