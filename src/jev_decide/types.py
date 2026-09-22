"""The two public result types: :class:`Choice` and :class:`Score`.

Both are ``@dataclass(frozen=True)`` and round-trip through ``to_dict()`` /
``from_dict()`` as plain JSON, so a caller can drop them straight into a log line, a
dashboard payload or an experience store without writing an encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ._errors import ConfigError

__all__ = ["Choice", "Score"]


def _check_unit(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number in 0..1, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or not (0.0 <= number <= 1.0):
        raise ConfigError(f"{name} must be a finite number in 0..1, got {number!r}")
    return number


def _check_latency(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"latency_ms must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ConfigError(f"latency_ms must be finite and >= 0, got {number!r}")
    return number


def _check_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a str, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Choice:
    """One option picked out of a fixed set, with the full distribution behind it.

    Attributes
    ----------
    value:
        The selected option.  Always one of the ``options`` that were asked about.
    probs:
        Probability per option, normalised so the values sum to 1.
    confidence:
        0..1.  How peaked ``probs`` is -- **not** the probability of ``value``.
        Feed this to :meth:`jev_decide.Decider.gate`.
    latency_ms:
        Wall-clock milliseconds the whole decision took, fallbacks included.
    backend:
        Which backend actually produced this answer: ``"jev"``, ``"llm"`` or
        ``"rules"``.
    degraded:
        ``True`` when this is a fallback answer rather than the backend you asked for.
    note:
        Human-readable reason for the degradation (empty when ``degraded`` is False).
    raw:
        Backend-specific payload, kept for auditing.  ``None`` for the rules backend.
    """

    value: str
    probs: dict[str, float]
    confidence: float
    latency_ms: float
    backend: str
    degraded: bool = False
    note: str = ""
    raw: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _check_str("value", self.value))
        object.__setattr__(self, "backend", _check_str("backend", self.backend))
        object.__setattr__(self, "note", _check_str("note", self.note))
        object.__setattr__(self, "confidence", _check_unit("confidence", self.confidence))
        object.__setattr__(self, "latency_ms", _check_latency(self.latency_ms))
        object.__setattr__(self, "degraded", bool(self.degraded))

        if not isinstance(self.probs, dict):
            raise ConfigError(f"probs must be a dict, got {type(self.probs).__name__}")
        probs: dict[str, float] = {}
        for key, value in self.probs.items():
            probs[_check_str("probs key", key)] = _check_unit(f"probs[{key!r}]", value)
        # Defensive copy: a frozen dataclass whose dict a caller can mutate is a lie.
        object.__setattr__(self, "probs", probs)

        if self.raw is not None and not isinstance(self.raw, dict):
            raise ConfigError(f"raw must be a dict or None, got {type(self.raw).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict.  Round-trips through :meth:`from_dict`."""
        return {
            "value": self.value,
            "probs": dict(self.probs),
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "degraded": self.degraded,
            "note": self.note,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Choice:
        """Rebuild a :class:`Choice` from :meth:`to_dict` output."""
        if not isinstance(d, dict):
            raise ConfigError(f"Choice.from_dict expects a dict, got {type(d).__name__}")
        required = ("value", "probs", "confidence", "latency_ms", "backend")
        missing = [k for k in required if k not in d]
        if missing:
            raise ConfigError(f"Choice.from_dict is missing required key(s): {', '.join(missing)}")
        return cls(
            value=d["value"],
            probs=dict(d["probs"]) if isinstance(d["probs"], dict) else d["probs"],
            confidence=d["confidence"],
            latency_ms=d["latency_ms"],
            backend=d["backend"],
            degraded=bool(d.get("degraded", False)),
            note=d.get("note", ""),
            raw=d.get("raw"),
        )


@dataclass(frozen=True)
class Score:
    """A number on a caller-defined scale, with the confidence to act on it.

    Attributes
    ----------
    value:
        The score, always inside ``[lo, hi]``.
    lo, hi:
        The scale that was asked for, echoed back so a logged ``Score`` is
        self-describing.
    confidence:
        0..1, same meaning as on :class:`Choice`.
    latency_ms:
        Wall-clock milliseconds for the whole decision.
    backend:
        ``"jev"``, ``"llm"`` or ``"rules"``.
    degraded:
        ``True`` when this is a fallback answer.
    note:
        Human-readable reason for the degradation.
    """

    value: float
    lo: float
    hi: float
    confidence: float
    latency_ms: float
    backend: str
    degraded: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("value", "lo", "hi"):
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ConfigError(f"{name} must be a number, got {number!r}")
            number = float(number)
            if not math.isfinite(number):
                raise ConfigError(f"{name} must be finite, got {number!r}")
            object.__setattr__(self, name, number)
        if self.lo > self.hi:
            raise ConfigError(f"lo ({self.lo}) must be <= hi ({self.hi})")
        if not (self.lo <= self.value <= self.hi):
            raise ConfigError(f"value ({self.value}) must lie in [{self.lo}, {self.hi}]")
        object.__setattr__(self, "backend", _check_str("backend", self.backend))
        object.__setattr__(self, "note", _check_str("note", self.note))
        object.__setattr__(self, "confidence", _check_unit("confidence", self.confidence))
        object.__setattr__(self, "latency_ms", _check_latency(self.latency_ms))
        object.__setattr__(self, "degraded", bool(self.degraded))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict.  Round-trips through :meth:`from_dict`."""
        return {
            "value": self.value,
            "lo": self.lo,
            "hi": self.hi,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "degraded": self.degraded,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Score:
        """Rebuild a :class:`Score` from :meth:`to_dict` output."""
        if not isinstance(d, dict):
            raise ConfigError(f"Score.from_dict expects a dict, got {type(d).__name__}")
        required = ("value", "lo", "hi", "confidence", "latency_ms", "backend")
        missing = [k for k in required if k not in d]
        if missing:
            raise ConfigError(f"Score.from_dict is missing required key(s): {', '.join(missing)}")
        return cls(
            value=d["value"],
            lo=d["lo"],
            hi=d["hi"],
            confidence=d["confidence"],
            latency_ms=d["latency_ms"],
            backend=d["backend"],
            degraded=bool(d.get("degraded", False)),
            note=d.get("note", ""),
        )
