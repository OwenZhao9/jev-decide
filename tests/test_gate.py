"""``gate()`` -- selective prediction: act only when the answer is confident enough."""

from __future__ import annotations

import pytest

from jev_decide import Choice, ConfigError, Decider


def choice(confidence: float, value: str = "high") -> Choice:
    return Choice(
        value=value,
        probs={"off": 0.1, "low": 0.2, "high": 0.7},
        confidence=confidence,
        latency_ms=1.0,
        backend="jev",
    )


def test_a_confident_answer_is_acted_on_whatever_the_strategy() -> None:
    confident = choice(0.9)
    assert Decider.gate(confident, min_confidence=0.7, on_low="keep", current="off") == "high"
    assert Decider.gate(confident, min_confidence=0.7, on_low="first") == "high"
    assert Decider.gate(confident, min_confidence=0.7, on_low="raise") == "high"


def test_keep_holds_the_current_state() -> None:
    assert Decider.gate(choice(0.3), min_confidence=0.8, on_low="keep", current="off") == "off"


def test_keep_is_the_default_strategy() -> None:
    assert Decider.gate(choice(0.3), min_confidence=0.8, current="low") == "low"


def test_first_falls_back_to_the_first_option() -> None:
    # probs is keyed in the order the options were asked about, so the first key is
    # options[0] -- by convention the safe / do-nothing option.
    assert Decider.gate(choice(0.3), min_confidence=0.8, on_low="first") == "off"


def test_raise_refuses_loudly() -> None:
    with pytest.raises(ValueError, match="below min_confidence"):
        Decider.gate(choice(0.31), min_confidence=0.8, on_low="raise")


def test_the_threshold_is_inclusive() -> None:
    assert Decider.gate(choice(0.8), min_confidence=0.8, on_low="raise") == "high"
    with pytest.raises(ValueError):
        Decider.gate(choice(0.7999999), min_confidence=0.8, on_low="raise")


def test_min_confidence_zero_always_passes() -> None:
    assert Decider.gate(choice(0.0), min_confidence=0.0, on_low="raise") == "high"


def test_keep_without_a_current_state_is_a_programming_error() -> None:
    # There is nothing to "keep" without it, so say so on every call rather than
    # silently acting on a low-confidence answer at 3am.
    with pytest.raises(ConfigError, match="needs current"):
        Decider.gate(choice(0.95), min_confidence=0.5, on_low="keep")


def test_gate_is_a_static_method_usable_without_an_instance() -> None:
    assert Decider.gate(choice(0.99), min_confidence=0.5, on_low="first") == "high"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_confidence": 1.5},
        {"min_confidence": -0.1},
        {"min_confidence": "high"},
        {"min_confidence": 0.5, "on_low": "shrug"},
        {"min_confidence": 0.5, "on_low": "keep", "current": 7},
    ],
)
def test_bad_gate_arguments_raise_value_error(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Decider.gate(choice(0.9), **kwargs)


def test_gate_rejects_something_that_is_not_a_choice() -> None:
    with pytest.raises(ConfigError, match="expects a Choice"):
        Decider.gate({"value": "high"}, min_confidence=0.5, on_low="first")  # type: ignore[arg-type]


def test_end_to_end_a_degraded_answer_does_not_move_anything() -> None:
    d = Decider("auto")  # no keys anywhere
    result = d.choice({"tilt_deg": 31.0}, "assist level?", ["off", "low", "high"])
    assert Decider.gate(result, min_confidence=0.6, on_low="keep", current="low") == "low"
