"""``auto``: walk the fallback chain, degrade loudly, always come back with something."""

from __future__ import annotations

import pytest

from conftest import jev_choice_body, llm_body
from jev_decide import ConfigError, Decider


def test_with_no_keys_at_all_auto_still_works_and_says_it_is_degraded(mock_http) -> None:
    d = Decider("auto")
    result = d.choice(
        {"battery_pct": 12},
        "assist level?",
        ["off", "low", "high"],
        rules=lambda s: "low" if s["battery_pct"] < 20 else "high",
    )
    assert mock_http.count == 0, "with no keys nothing should hit the network"
    assert result.value == "low"
    assert result.backend == "rules"
    assert result.degraded is True
    assert "jev: no API key" in result.note
    assert "llm: no API key" in result.note


def test_with_no_keys_and_no_rules_auto_returns_a_refusal_not_a_guess(mock_http) -> None:
    result = Decider("auto").choice({}, "q", ["a", "b", "c"])
    assert result.degraded is True
    assert result.confidence == 0.0
    assert Decider.gate(result, min_confidence=0.01, on_low="keep", current="hold") == "hold"


def test_score_degrades_the_same_way(mock_http) -> None:
    result = Decider("auto").score({}, "how risky", 0.0, 1.0, rules=lambda _s: 0.25)
    assert result.backend == "rules"
    assert result.degraded is True
    assert result.value == pytest.approx(0.25)


def test_the_first_healthy_backend_wins_and_is_not_marked_degraded(mock_http) -> None:
    mock_http.json(jev_choice_body("high", {"off": 0.05, "low": 0.15, "high": 0.8}, 0.72))
    result = Decider("auto", api_key="sk-test").choice({}, "q", ["off", "low", "high"])
    assert result.backend == "jev"
    assert result.degraded is False
    assert result.note == ""
    assert mock_http.count == 1


def test_jev_failing_falls_through_to_llm(mock_http) -> None:
    mock_http.http_error(529, "overloaded")
    mock_http.json(llm_body({"choice": "b", "probabilities": {"a": 0.2, "b": 0.8}}))
    result = Decider("auto", api_key="sk-test", timeout_s=1.0).choice({}, "q", ["a", "b"])
    assert mock_http.count == 2
    assert mock_http.requests[0].url.endswith("/v1/systemone")
    assert mock_http.requests[1].url.endswith("/chat/completions")
    assert result.backend == "llm"
    assert result.degraded is True
    assert "jev: HTTP 529" in result.note


def test_everything_networked_failing_lands_on_rules(mock_http) -> None:
    mock_http.unreachable("no route to host")
    result = Decider("auto", api_key="sk-test", timeout_s=0.5).choice(
        {}, "q", ["a", "b"], rules=lambda _s: "b"
    )
    assert result.backend == "rules"
    assert result.value == "b"
    assert result.degraded is True
    assert result.note.count("no route to host") == 2


def test_the_chain_order_is_configurable(mock_http) -> None:
    mock_http.json(llm_body({"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}))
    d = Decider("auto", api_key="sk", timeout_s=1.0, fallback_chain=("llm", "jev", "rules"))
    result = d.choice({}, "q", ["a", "b"])
    assert mock_http.count == 1
    assert mock_http.last.url.endswith("/chat/completions")
    assert result.backend == "llm"
    assert result.degraded is False


def test_rules_is_appended_when_the_chain_forgets_it(mock_http) -> None:
    d = Decider("auto", fallback_chain=("jev", "llm"))
    assert d.chain == ("jev", "llm", "rules")
    assert d.choice({}, "q", ["a"], rules=lambda _s: "a").backend == "rules"


def test_an_explicit_backend_still_has_rules_behind_it(mock_http) -> None:
    # Contract 0.4: run-time failures degrade, they never raise into the caller.
    d = Decider("jev", api_key="sk", timeout_s=0.5)
    assert d.chain == ("jev", "rules")
    mock_http.unreachable()
    result = d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert result.degraded is True


def test_an_explicit_rules_backend_never_reaches_for_the_network(mock_http) -> None:
    d = Decider("rules", api_key="sk")
    assert d.chain == ("rules",)
    d.choice({}, "q", ["a"], rules=lambda _s: "a")
    assert mock_http.count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"backend": "nope"},
        {"timeout_s": 0},
        {"timeout_s": -1.0},
        {"timeout_s": float("inf")},
        {"timeout_s": "2"},
        {"fallback_chain": ()},
        {"fallback_chain": ("jev", "nope")},
        {"fallback_chain": ("auto",)},
        {"fallback_chain": ("jev", "jev")},
        {"fallback_chain": "jev"},
        {"on_decision": "not callable"},
        {"api_key": 123},
        {"model": object()},
    ],
)
def test_bad_construction_raises_value_error(kwargs: dict) -> None:
    backend = kwargs.pop("backend", "auto")
    with pytest.raises(ValueError):
        Decider(backend, **kwargs)


@pytest.mark.parametrize(
    "call",
    [
        lambda d: d.choice({}, "", ["a"]),
        lambda d: d.choice({}, "q", []),
        lambda d: d.choice({}, "q", "ab"),
        lambda d: d.choice({}, "q", ["a", "a"]),
        lambda d: d.choice({}, "q", ["a", ""]),
        lambda d: d.choice({}, "q", ["a", 2]),
        lambda d: d.choice({}, "q", ["a"], rules="nope"),
        lambda d: d.score({}, "r", 5.0, 1.0),
        lambda d: d.score({}, "r", 0.0, float("nan")),
        lambda d: d.score({}, "r", 0.0, 0.0),
        lambda d: d.score({}, ""),
        lambda d: d.score({}, "r", rules=7),
    ],
)
def test_bad_arguments_raise_value_error(call) -> None:
    d = Decider("rules")
    with pytest.raises(ConfigError):
        call(d)
