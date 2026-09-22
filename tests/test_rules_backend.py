"""The ``rules`` backend: zero dependencies, always available, never lies."""

from __future__ import annotations

import pytest

from jev_decide import Decider


def test_choice_uses_the_supplied_pure_function() -> None:
    d = Decider("rules")
    result = d.choice(
        {"battery_pct": 12},
        "assist level?",
        ["off", "low", "high"],
        rules=lambda s: "low" if s["battery_pct"] < 20 else "high",
    )
    assert result.value == "low"
    assert result.backend == "rules"
    assert result.degraded is False
    assert result.note == ""
    assert result.probs == {"off": 0.0, "low": 1.0, "high": 0.0}
    assert result.confidence == pytest.approx(1.0)
    assert result.raw is None


def test_score_uses_the_supplied_pure_function() -> None:
    d = Decider("rules")
    result = d.score({"tilt_deg": 31.0}, "how risky is this posture", 0.0, 10.0,
                     rules=lambda s: s["tilt_deg"] / 10.0)
    assert result.value == pytest.approx(3.1)
    assert (result.lo, result.hi) == (0.0, 10.0)
    assert result.backend == "rules"
    assert result.degraded is False
    assert result.confidence == pytest.approx(1.0)


def test_score_out_of_range_is_clamped_and_said_out_loud() -> None:
    d = Decider("rules")
    result = d.score({}, "rubric", 0.0, 5.0, rules=lambda _s: 99.0)
    assert result.value == 5.0
    assert "clamped" in result.note


def test_no_rules_function_gives_a_zero_confidence_placeholder() -> None:
    d = Decider("rules")
    result = d.choice({}, "assist level?", ["off", "low", "high"])
    assert result.degraded is True
    assert result.confidence == 0.0
    assert result.probs == {"off": pytest.approx(1 / 3), "low": pytest.approx(1 / 3),
                            "high": pytest.approx(1 / 3)}
    assert "no rules function" in result.note
    # And the whole point: gate() refuses to act on it.
    assert Decider.gate(result, min_confidence=0.5, on_low="keep", current="off") == "off"


def test_no_rules_function_score_sits_in_the_middle_with_no_confidence() -> None:
    d = Decider("rules")
    result = d.score({}, "rubric", 2.0, 6.0)
    assert result.value == 4.0
    assert result.confidence == 0.0
    assert result.degraded is True


def test_a_rules_function_that_raises_degrades_instead_of_propagating() -> None:
    def boom(_state: object) -> str:
        raise RuntimeError("bad sensor")

    result = Decider("rules").choice({}, "q", ["a", "b"], rules=boom)
    assert result.degraded is True
    assert "bad sensor" in result.note
    assert result.confidence == 0.0


def test_a_rules_function_returning_an_unknown_option_degrades() -> None:
    result = Decider("rules").choice({}, "q", ["a", "b"], rules=lambda _s: "c")
    assert result.value in ("a", "b")
    assert result.degraded is True
    assert "not one of" in result.note


def test_a_rules_function_returning_a_non_number_score_degrades() -> None:
    result = Decider("rules").score({}, "r", rules=lambda _s: "seven")  # type: ignore[arg-type,return-value]
    assert result.degraded is True
    assert "expected a number" in result.note


def test_rules_backend_never_touches_the_network(mock_http) -> None:
    Decider("rules").choice({}, "q", ["a"], rules=lambda _s: "a")
    assert mock_http.count == 0


def test_same_input_gives_the_same_answer() -> None:
    d = Decider("rules")
    state = {"x": 3, "y": "a"}
    picker = lambda s: "a" if s["x"] > 2 else "b"  # noqa: E731
    first = d.choice(state, "q", ["a", "b"], rules=picker)
    second = d.choice(state, "q", ["a", "b"], rules=picker)
    assert first.to_dict() | {"latency_ms": 0} == second.to_dict() | {"latency_ms": 0}
