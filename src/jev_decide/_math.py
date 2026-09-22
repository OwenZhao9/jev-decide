"""Pure, dependency-free numerics: probability hygiene, confidence, percentiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "clamp",
    "confidence_from_probs",
    "normalise_probs",
    "percentiles",
    "uniform_probs",
]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]`` (order of the bounds does not matter)."""
    low, high = (lo, hi) if lo <= hi else (hi, lo)
    return low if value < low else high if value > high else value


def uniform_probs(options: Sequence[str]) -> dict[str, float]:
    """A flat distribution over ``options`` -- i.e. "I have no idea"."""
    if not options:
        return {}
    share = 1.0 / len(options)
    return {option: share for option in options}


def normalise_probs(
    raw: Mapping[str, object] | None, options: Sequence[str]
) -> tuple[dict[str, float], bool]:
    """Project ``raw`` onto ``options`` and normalise it to sum to 1.

    Returns ``(probs, usable)``.  ``usable`` is ``False`` when the input carried no
    positive mass for any known option -- the caller then knows the backend gave it
    nothing to go on and should say so rather than invent certainty.

    Unknown keys are dropped, missing options get 0.0, negative and non-finite values
    are treated as 0.0.
    """
    if not options:
        return {}, False

    cleaned: dict[str, float] = {}
    for option in options:
        value = raw.get(option) if raw else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            cleaned[option] = 0.0
            continue
        number = float(value)
        cleaned[option] = number if math.isfinite(number) and number > 0.0 else 0.0

    total = math.fsum(cleaned.values())
    if total <= 0.0:
        return uniform_probs(options), False
    return {option: value / total for option, value in cleaned.items()}, True


def confidence_from_probs(probs: Mapping[str, float]) -> float:
    """Collapse a distribution's *shape* into a single 0..1 number.

    Normalised negative entropy: ``1 - H(p) / ln(n)``.

    * one-hot (all mass on one outcome) -> ``1.0``
    * flat (every outcome equally likely) -> ``0.0``

    This is the same idea TypeSafe documents for its own ``confidence`` field ("a
    flatter distribution means lower confidence"), computed locally so that every
    backend reports confidence on a comparable scale.
    """
    values = [float(value) for value in probs.values() if value > 0.0]
    n = len(probs)
    if n <= 1:
        return 1.0 if values else 0.0
    total = math.fsum(values)
    if total <= 0.0:
        return 0.0
    entropy = -math.fsum((v / total) * math.log(v / total) for v in values)
    return clamp(1.0 - entropy / math.log(n), 0.0, 1.0)


def percentiles(samples: Sequence[float]) -> dict[str, float]:
    """p50 / p90 / p99 of ``samples`` (nearest-rank).  Empty input -> empty dict."""
    if not samples:
        return {}
    ordered = sorted(samples)
    out: dict[str, float] = {}
    for label, fraction in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        rank = math.ceil(fraction * len(ordered))
        index = min(max(rank - 1, 0), len(ordered) - 1)
        out[label] = round(ordered[index], 3)
    return out
