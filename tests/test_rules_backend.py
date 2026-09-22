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


# ------------------------------------------------- rules may return weights, not just a pick

OPTIONS = ("zero", "resist", "assist")


def test_weights_become_probs_and_the_argmax_wins() -> None:
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS, rules=lambda s: {"assist": 9.0, "resist": 1.0, "zero": 0.0})
    assert c.value == "assist"
    assert c.probs["assist"] == pytest.approx(0.9)
    assert c.probs["resist"] == pytest.approx(0.1)
    assert c.probs["zero"] == 0.0
    assert sum(c.probs.values()) == pytest.approx(1.0)


def test_a_close_call_reports_low_confidence_and_the_gate_holds() -> None:
    """The whole point: offline, a one-hot answer would gate through every time."""
    d = Decider(backend="rules")
    clear = d.choice(
        {}, "which?", OPTIONS, rules=lambda s: {"assist": 9.0, "resist": 1.0, "zero": 0.2}
    )
    close = d.choice(
        {}, "which?", OPTIONS, rules=lambda s: {"assist": 5.0, "resist": 4.5, "zero": 0.1}
    )
    assert clear.confidence > close.confidence
    assert Decider.gate(clear, min_confidence=0.5, on_low="keep", current="zero") == "assist"
    assert Decider.gate(close, min_confidence=0.5, on_low="keep", current="zero") == "zero"


def test_flat_weights_mean_no_idea() -> None:
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS, rules=lambda s: {o: 1.0 for o in OPTIONS})
    assert c.confidence == pytest.approx(0.0)


def test_returning_a_plain_string_still_gives_one_hot() -> None:
    """Backwards compatible: the str form keeps its old meaning."""
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS, rules=lambda s: "resist")
    assert c.value == "resist"
    assert c.confidence == pytest.approx(1.0)
    assert c.probs == {"zero": 0.0, "resist": 1.0, "assist": 0.0}


def test_unknown_keys_are_dropped_and_negatives_are_zeroed() -> None:
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS,
                 rules=lambda s: {"assist": 3.0, "resist": -5.0, "nonsense": 100.0})
    assert c.value == "assist"
    assert c.probs == {"zero": 0.0, "resist": 0.0, "assist": 1.0}


def test_weights_with_no_positive_mass_are_not_an_answer() -> None:
    """All-zero weights mean the rules function had nothing to say.

    ``rules`` is the terminal backend, so there is nowhere left to fall through to:
    the result comes back degraded with confidence 0.0 rather than inventing
    certainty, and ``gate`` refuses to act on it.
    """
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS, rules=lambda s: {"assist": 0.0, "resist": 0.0})
    assert c.degraded is True
    assert c.confidence == pytest.approx(0.0)
    assert Decider.gate(c, min_confidence=0.1, on_low="keep", current="zero") == "zero"


def test_ties_resolve_to_the_first_option_given() -> None:
    d = Decider(backend="rules")
    c = d.choice({}, "which?", OPTIONS, rules=lambda s: {"resist": 2.0, "assist": 2.0})
    assert c.value == "resist"          # options order decides, deterministically
