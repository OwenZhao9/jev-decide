"""``Decider`` -- ask one typed question, get an answer you can gate on.

.. warning::

   Every ``Decider`` method **blocks**: it does network IO and can take up to
   ``2 * timeout_s``.  Never call it from a real-time control loop.  Call it on a task
   boundary, or from a worker thread, and let the loop read the last answer.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from ._backends import (
    Backend,
    BackendUnavailable,
    ChoiceRequest,
    ChoiceResult,
    JevBackend,
    LlmBackend,
    RulesBackend,
    ScoreRequest,
    ScoreResult,
)
from ._errors import ConfigError
from ._jsonsafe import ensure_json_safe
from ._math import clamp, percentiles
from .types import Choice, Score

__all__ = ["Decider"]

_log = logging.getLogger(__name__)

BackendName = Literal["auto", "jev", "llm", "rules"]

_BACKEND_NAMES: tuple[str, ...] = ("jev", "llm", "rules")
_TERMINAL_BACKEND = "rules"

#: The whole call -- first attempt plus every fallback -- is bounded by
#: ``_TOTAL_BUDGET_FACTOR * timeout_s``.  The contract fixes this at 2x.
_TOTAL_BUDGET_FACTOR = 2.0

#: Each network attempt is capped at ``timeout_s`` and at this share of what is left of
#: the total budget, so the fallbacks cannot add up past the 2x ceiling.
_SLICE_FACTOR = 0.9

#: Below this there is no point opening a socket; skip straight to the next backend.
_MIN_NETWORK_SLICE_S = 0.005

_LATENCY_WINDOW = 256


class _BackendStats:
    """Rolling counters for one backend, surfaced by :meth:`Decider.health`."""

    __slots__ = ("attempts", "failures", "last_error", "latencies", "ok")

    def __init__(self) -> None:
        self.attempts = 0
        self.ok = 0
        self.failures = 0
        self.last_error: str | None = None
        self.latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)


class Decider:
    """Typed decisions with calibrated confidence, and a fallback chain behind them.

    ``Decider`` instances are **not thread safe** and hold no lock on purpose: share
    one per thread rather than one across threads.
    """

    def __init__(
        self,
        backend: BackendName = "auto",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 2.0,
        fallback_chain: Sequence[str] = ("jev", "llm", "rules"),
        on_decision: Callable[[Choice | Score], None] | None = None,
    ) -> None:
        if backend not in ("auto", *_BACKEND_NAMES):
            raise ConfigError(
                f"backend must be one of 'auto', {_BACKEND_NAMES}, got {backend!r}"
            )
        self._timeout_s = _check_timeout(timeout_s)
        for name, value in (("api_key", api_key), ("base_url", base_url), ("model", model)):
            if value is not None and not isinstance(value, str):
                raise ConfigError(f"{name} must be a str or None, got {type(value).__name__}")
        if on_decision is not None and not callable(on_decision):
            raise ConfigError("on_decision must be callable or None")

        self._backend_name: str = backend
        self._fallback_chain = _check_chain(fallback_chain)
        self._on_decision = on_decision

        self._backends: dict[str, Backend] = {
            "jev": JevBackend(api_key=api_key, base_url=base_url, model=model),
            "llm": LlmBackend(api_key=api_key, base_url=base_url, model=model),
            "rules": RulesBackend(),
        }
        self._chain: tuple[str, ...] = self._resolve_chain(backend, self._fallback_chain)
        self._stats: dict[str, _BackendStats] = {n: _BackendStats() for n in _BACKEND_NAMES}
        self._decisions = 0

    # -- configuration ----------------------------------------------------------

    @staticmethod
    def _resolve_chain(backend: str, fallback_chain: tuple[str, ...]) -> tuple[str, ...]:
        """The order backends are tried in, always ending at the always-available one.

        ``rules`` is appended when it is missing: the contract forbids raising at run
        time, so there must be a backend that cannot fail to answer.
        """
        chain = fallback_chain if backend == "auto" else (backend,)
        if _TERMINAL_BACKEND not in chain:
            chain = (*chain, _TERMINAL_BACKEND)
        return chain

    @property
    def backend(self) -> str:
        """The backend this ``Decider`` was configured with."""
        return self._backend_name

    @property
    def chain(self) -> tuple[str, ...]:
        """The resolved order backends are attempted in."""
        return self._chain

    @property
    def timeout_s(self) -> float:
        """Per-backend timeout.  A whole call is bounded by twice this."""
        return self._timeout_s

    # -- decisions --------------------------------------------------------------

    def choice(
        self,
        state: Mapping[str, Any],
        question: str,
        options: Sequence[str],
        *,
        rules: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> Choice:
        """Pick one of ``options`` given ``state``.

        Parameters
        ----------
        state:
            A JSON-serialisable mapping.  Anything that is not (a custom object, a
            set, ``NaN``, a non-string dict key) raises :class:`ValueError` -- it is
            never silently ``str()``-ed into the request.
        question:
            The question to answer, in plain words.
        options:
            The allowed answers.  The returned ``value`` is always one of these, and
            ``probs`` is keyed by them in this order.
        rules:
            A pure function ``state -> option`` used by the ``rules`` backend.
            Supplying one is what makes an offline fallback meaningful.

        Returns
        -------
        Choice
            Never raises at run time; check ``degraded`` and ``note``.
        """
        state_dict = _check_state(state)
        question = _check_text("question", question)
        option_tuple = _check_options(options)
        if rules is not None and not callable(rules):
            raise ConfigError("rules must be callable or None")

        request = ChoiceRequest(
            state=state_dict, question=question, options=option_tuple, rules=rules
        )
        result, backend_name, notes, degraded, latency_ms = self._run(
            request, lambda backend, budget: backend.choice(request, budget)
        )

        if result is None:  # pragma: no cover - the rules backend cannot fail this way
            result = ChoiceResult(
                value=option_tuple[0],
                probs={option: 1.0 / len(option_tuple) for option in option_tuple},
                confidence=0.0,
                note="every backend failed",
                degraded=True,
            )
            backend_name = _TERMINAL_BACKEND
            degraded = True

        decision = Choice(
            value=result.value,
            # Clamped defensively: a float-rounding artefact in a backend must degrade
            # the number, never raise out of a decision call.
            probs={option: clamp(p, 0.0, 1.0) for option, p in result.probs.items()},
            confidence=clamp(result.confidence, 0.0, 1.0),
            latency_ms=latency_ms,
            backend=backend_name,
            degraded=degraded or result.degraded,
            note=_join_notes(notes, result.note),
            raw=result.raw,
        )
        return self._emit(decision)

    def score(
        self,
        state: Mapping[str, Any],
        rubric: str,
        lo: float = 0.0,
        hi: float = 10.0,
        *,
        rules: Callable[[Mapping[str, Any]], float] | None = None,
    ) -> Score:
        """Rate ``state`` against ``rubric`` on the scale ``[lo, hi]``.

        Parameters
        ----------
        state:
            JSON-serialisable mapping; same rules as :meth:`choice`.
        rubric:
            What is being rated, in plain words.
        lo, hi:
            The scale.  ``lo`` must be strictly below ``hi``; the returned ``value``
            is always inside the range.
        rules:
            Pure function ``state -> float`` for the ``rules`` backend.  A value
            outside ``[lo, hi]`` is clamped and noted.

        Returns
        -------
        Score
            Never raises at run time; check ``degraded`` and ``note``.
        """
        state_dict = _check_state(state)
        rubric = _check_text("rubric", rubric)
        lo_f, hi_f = _check_range(lo, hi)
        if rules is not None and not callable(rules):
            raise ConfigError("rules must be callable or None")

        request = ScoreRequest(state=state_dict, rubric=rubric, lo=lo_f, hi=hi_f, rules=rules)
        result, backend_name, notes, degraded, latency_ms = self._run(
            request, lambda backend, budget: backend.score(request, budget)
        )

        if result is None:  # pragma: no cover - the rules backend cannot fail this way
            result = ScoreResult(
                value=(lo_f + hi_f) / 2.0,
                confidence=0.0,
                note="every backend failed",
                degraded=True,
            )
            backend_name = _TERMINAL_BACKEND
            degraded = True

        decision = Score(
            value=clamp(result.value, lo_f, hi_f),
            lo=lo_f,
            hi=hi_f,
            confidence=clamp(result.confidence, 0.0, 1.0),
            latency_ms=latency_ms,
            backend=backend_name,
            degraded=degraded or result.degraded,
            note=_join_notes(notes, result.note),
        )
        return self._emit(decision)

    # -- the fallback loop ------------------------------------------------------

    def _run(
        self,
        request: ChoiceRequest | ScoreRequest,
        call: Callable[[Backend, float], Any],
    ) -> tuple[Any, str, list[str], bool, float]:
        """Walk the chain until one backend answers, inside the 2x total budget."""
        started = time.monotonic()
        deadline = started + _TOTAL_BUDGET_FACTOR * self._timeout_s
        notes: list[str] = []

        for position, name in enumerate(self._chain):
            backend = self._backends[name]
            stats = self._stats[name]

            if name != _TERMINAL_BACKEND:
                remaining = deadline - time.monotonic()
                budget = min(self._timeout_s, remaining * _SLICE_FACTOR)
                if budget < _MIN_NETWORK_SLICE_S:
                    notes.append(f"{name}: skipped, no time budget left")
                    continue
            else:
                # Pure local computation: no IO, so nothing to time-box.
                budget = max(deadline - time.monotonic(), 0.0)

            attempt_started = time.monotonic()
            stats.attempts += 1
            try:
                result = call(backend, budget)
            except BackendUnavailable as exc:
                stats.failures += 1
                stats.last_error = str(exc)
                stats.latencies.append((time.monotonic() - attempt_started) * 1000.0)
                notes.append(f"{name}: {exc}")
                _log.info("backend %s unavailable: %s", name, exc)
                continue
            except Exception as exc:  # defensive: a backend must never break the caller
                stats.failures += 1
                stats.last_error = f"unexpected {type(exc).__name__}: {exc}"
                notes.append(f"{name}: unexpected {type(exc).__name__}: {exc}")
                _log.warning("backend %s raised unexpectedly", name, exc_info=True)
                continue

            stats.ok += 1
            stats.latencies.append((time.monotonic() - attempt_started) * 1000.0)
            latency_ms = (time.monotonic() - started) * 1000.0
            return result, name, notes, position > 0, latency_ms

        return None, _TERMINAL_BACKEND, notes, True, (time.monotonic() - started) * 1000.0

    def _emit(self, decision: Choice | Score) -> Choice | Score:
        self._decisions += 1
        if self._on_decision is not None:
            try:
                self._on_decision(decision)
            except Exception:  # a broken audit hook must not break the decision
                _log.warning("on_decision callback raised", exc_info=True)
        return decision

    # -- gating -----------------------------------------------------------------

    @staticmethod
    def gate(
        c: Choice,
        *,
        min_confidence: float,
        on_low: Literal["keep", "first", "raise"] = "keep",
        current: str | None = None,
    ) -> str:
        """Act on ``c`` only if it is confident enough -- otherwise do not move.

        This is *selective prediction*: trading coverage for risk by declining to
        answer when the model is not sure (arXiv:2607.03528).

        Parameters
        ----------
        c:
            The :class:`Choice` to gate.
        min_confidence:
            0..1 threshold.  ``c.confidence >= min_confidence`` returns ``c.value``.
        on_low:
            What to do below the threshold.  ``"keep"`` returns ``current`` (the state
            you are already in -- i.e. change nothing), ``"first"`` returns the first
            option, ``"raise"`` raises :class:`ValueError`.
        current:
            Required when ``on_low="keep"``: there is nothing to keep without it.

        Returns
        -------
        str
            The option to act on.
        """
        if not isinstance(c, Choice):
            raise ConfigError(f"gate() expects a Choice, got {type(c).__name__}")
        threshold = _check_unit_arg("min_confidence", min_confidence)
        if on_low not in ("keep", "first", "raise"):
            raise ConfigError(
                f"on_low must be 'keep', 'first' or 'raise', got {on_low!r}"
            )
        if current is not None and not isinstance(current, str):
            raise ConfigError(f"current must be a str or None, got {type(current).__name__}")
        if on_low == "keep" and current is None:
            raise ConfigError(
                "on_low='keep' needs current=<the option you are already on>; "
                "pass one, or use on_low='first' / on_low='raise'"
            )

        if c.confidence >= threshold:
            return c.value

        if on_low == "raise":
            raise ValueError(
                f"confidence {c.confidence:.3f} is below min_confidence {threshold:.3f} "
                f"for choice {c.value!r} from backend {c.backend!r}"
            )
        if on_low == "first":
            if not c.probs:
                raise ConfigError("on_low='first' needs a Choice with a non-empty probs map")
            return next(iter(c.probs))
        # current is not None: guaranteed by the on_low='keep' check above.
        return current

    # -- introspection ----------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Per-backend availability and recent latency percentiles.

        Cheap and non-blocking: it reports what this instance has already observed and
        how each backend is configured.  It does **not** probe the network, so calling
        it on a dashboard tick is safe.
        """
        backends: dict[str, Any] = {}
        for name in _BACKEND_NAMES:
            backend = self._backends[name]
            stats = self._stats[name]
            available, reason = backend.availability()
            entry: dict[str, Any] = {
                "available": available,
                "reason": reason,
                "in_chain": name in self._chain,
                "attempts": stats.attempts,
                "ok": stats.ok,
                "failures": stats.failures,
                "last_error": stats.last_error,
                "latency_ms": percentiles(list(stats.latencies)),
            }
            endpoint = getattr(backend, "endpoint", None)
            if endpoint is not None:
                entry["endpoint"] = endpoint
            model = getattr(backend, "model", None)
            if model is not None:
                entry["model"] = model
            backends[name] = entry

        return {
            "backend": self._backend_name,
            "chain": list(self._chain),
            "timeout_s": self._timeout_s,
            "total_budget_s": _TOTAL_BUDGET_FACTOR * self._timeout_s,
            "decisions": self._decisions,
            "backends": backends,
        }


# -- argument checking ----------------------------------------------------------


def _check_timeout(timeout_s: object) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
        raise ConfigError(f"timeout_s must be a number, got {type(timeout_s).__name__}")
    value = float(timeout_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ConfigError(f"timeout_s must be finite and > 0, got {timeout_s!r}")
    return value


def _check_chain(fallback_chain: Sequence[str]) -> tuple[str, ...]:
    if isinstance(fallback_chain, str) or not isinstance(fallback_chain, Sequence):
        raise ConfigError("fallback_chain must be a sequence of backend names")
    chain = tuple(fallback_chain)
    if not chain:
        raise ConfigError("fallback_chain must not be empty")
    for name in chain:
        if name not in _BACKEND_NAMES:
            raise ConfigError(
                f"fallback_chain entry {name!r} is not one of {_BACKEND_NAMES} "
                "('auto' is not a backend, it is what selects this chain)"
            )
    if len(set(chain)) != len(chain):
        raise ConfigError(f"fallback_chain has duplicate entries: {list(chain)}")
    return chain


def _check_state(state: object) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ConfigError(
            f"state must be a mapping, got {type(state).__name__}; "
            "wrap a bare value, e.g. {'text': value}"
        )
    as_dict = dict(state)
    ensure_json_safe(as_dict)
    return as_dict


def _check_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ConfigError(f"{name} must not be empty")
    return stripped


def _check_options(options: object) -> tuple[str, ...]:
    if isinstance(options, str) or not isinstance(options, Sequence):
        raise ConfigError("options must be a sequence of strings, not a bare string")
    tuple_options = tuple(options)
    if not tuple_options:
        raise ConfigError("options must not be empty")
    for option in tuple_options:
        if not isinstance(option, str):
            raise ConfigError(f"every option must be a str, got {type(option).__name__}")
        if not option.strip():
            raise ConfigError("options must not contain empty strings")
    if len(set(tuple_options)) != len(tuple_options):
        raise ConfigError(f"options must be unique, got {list(tuple_options)}")
    return tuple_options


def _check_range(lo: object, hi: object) -> tuple[float, float]:
    numbers: list[float] = []
    for name, value in (("lo", lo), ("hi", hi)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{name} must be a number, got {type(value).__name__}")
        number = float(value)
        if not math.isfinite(number):
            raise ConfigError(f"{name} must be finite, got {value!r}")
        numbers.append(number)
    if not numbers[0] < numbers[1]:
        raise ConfigError(f"lo ({numbers[0]}) must be strictly less than hi ({numbers[1]})")
    return numbers[0], numbers[1]


def _check_unit_arg(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number in 0..1, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number) or not (0.0 <= number <= 1.0):
        raise ConfigError(f"{name} must be in 0..1, got {value!r}")
    return number


def _join_notes(notes: Sequence[str], tail: str) -> str:
    parts = [note for note in (*notes, tail) if note]
    return "; ".join(parts)
