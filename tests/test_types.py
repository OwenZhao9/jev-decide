"""Choice and Score: JSON round-trip, immutability, and construction-time validation."""

from __future__ import annotations

import dataclasses
import json

import pytest

from jev_decide import Choice, ConfigError, Score


def make_choice(**overrides: object) -> Choice:
    kwargs: dict[str, object] = {
        "value": "high",
        "probs": {"off": 0.1, "low": 0.2, "high": 0.7},
        "confidence": 0.62,
        "latency_ms": 12.5,
        "backend": "jev",
    }
    kwargs.update(overrides)
    return Choice(**kwargs)  # type: ignore[arg-type]


def make_score(**overrides: object) -> Score:
    kwargs: dict[str, object] = {
        "value": 7.5,
        "lo": 0.0,
        "hi": 10.0,
        "confidence": 0.8,
        "latency_ms": 9.0,
        "backend": "llm",
    }
    kwargs.update(overrides)
    return Score(**kwargs)  # type: ignore[arg-type]


def test_choice_round_trips_through_json() -> None:
    original = make_choice(raw={"answer": {"type": "choice"}}, note="n", degraded=True)
    restored = Choice.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_score_round_trips_through_json() -> None:
    original = make_score(note="clamped", degraded=True)
    restored = Score.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_to_dict_is_json_serialisable_without_a_custom_encoder() -> None:
    json.dumps(make_choice().to_dict(), allow_nan=False)
    json.dumps(make_score().to_dict(), allow_nan=False)


def test_both_types_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_choice().value = "low"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_score().value = 1.0  # type: ignore[misc]


def test_probs_are_copied_so_the_caller_cannot_mutate_a_frozen_result() -> None:
    probs = {"a": 0.5, "b": 0.5}
    choice = make_choice(value="a", probs=probs)
    probs["a"] = 99.0
    assert choice.probs == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence": 1.5},
        {"confidence": float("nan")},
        {"confidence": "high"},
        {"latency_ms": -1.0},
        {"value": 3},
        {"probs": [("a", 1.0)]},
        {"probs": {"a": 2.0}},
        {"raw": "not a dict"},
    ],
)
def test_choice_rejects_impossible_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        make_choice(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"value": 11.0},
        {"lo": 10.0, "hi": 0.0},
        {"value": float("inf")},
        {"confidence": -0.1},
    ],
)
def test_score_rejects_impossible_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        make_score(**overrides)


def test_config_error_is_also_a_value_error() -> None:
    # Contract 0.4: constructor argument errors are ValueError.
    with pytest.raises(ValueError):
        make_choice(confidence=2.0)


@pytest.mark.parametrize("missing", ["value", "probs", "confidence", "latency_ms", "backend"])
def test_choice_from_dict_reports_missing_keys(missing: str) -> None:
    payload = make_choice().to_dict()
    payload.pop(missing)
    with pytest.raises(ConfigError, match=missing):
        Choice.from_dict(payload)


@pytest.mark.parametrize("missing", ["value", "lo", "hi", "confidence", "latency_ms", "backend"])
def test_score_from_dict_reports_missing_keys(missing: str) -> None:
    payload = make_score().to_dict()
    payload.pop(missing)
    with pytest.raises(ConfigError, match=missing):
        Score.from_dict(payload)
