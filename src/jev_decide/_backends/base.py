"""Shared vocabulary for the three backends.

A backend is deliberately dumb: it answers one request within one timeout, or it
raises :class:`BackendUnavailable`.  Choosing what to do about that -- fall back, note
the degradation, keep the latency budget -- is the ``Decider``'s job, in one place.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "Backend",
    "BackendUnavailable",
    "ChoiceRequest",
    "ChoiceResult",
    "ScoreRequest",
    "ScoreResult",
]


class BackendUnavailable(Exception):
    """This backend cannot answer right now -- try the next link in the chain.

    Internal control flow.  It never reaches the caller: the ``Decider`` catches it and
    degrades.
    """


@dataclass(frozen=True)
class ChoiceRequest:
    """One typed multiple-choice question."""

    state: Any
    question: str
    options: tuple[str, ...]
    rules: Callable[[Mapping[str, Any]], str] | None = None


@dataclass(frozen=True)
class ScoreRequest:
    """One typed rating question on the scale ``[lo, hi]``."""

    state: Any
    rubric: str
    lo: float
    hi: float
    rules: Callable[[Mapping[str, Any]], float] | None = None


@dataclass(frozen=True)
class ChoiceResult:
    """A backend's raw answer to a :class:`ChoiceRequest`, before packaging."""

    value: str
    probs: dict[str, float]
    confidence: float
    raw: dict[str, Any] | None = None
    note: str = ""
    degraded: bool = False


@dataclass(frozen=True)
class ScoreResult:
    """A backend's raw answer to a :class:`ScoreRequest`, before packaging."""

    value: float
    confidence: float
    raw: dict[str, Any] | None = None
    note: str = ""
    degraded: bool = False


class Backend(Protocol):
    """What the ``Decider`` needs from a backend."""

    name: str

    def availability(self) -> tuple[bool, str]:
        """``(usable, reason)``.  ``reason`` explains a ``False`` in plain words."""
        ...

    def choice(self, request: ChoiceRequest, timeout_s: float) -> ChoiceResult:
        """Answer ``request`` within ``timeout_s`` or raise :class:`BackendUnavailable`."""
        ...

    def score(self, request: ScoreRequest, timeout_s: float) -> ScoreResult:
        """Answer ``request`` within ``timeout_s`` or raise :class:`BackendUnavailable`."""
        ...

