"""``on_decision`` auditing and the non-blocking ``health()`` report."""

from __future__ import annotations

import json

import pytest

from conftest import jev_choice_body
from jev_decide import Choice, Decider, Score


def test_on_decision_fires_for_every_choice() -> None:
    seen: list[Choice | Score] = []
    d = Decider("rules", on_decision=seen.append)
    first = d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    second = d.choice({}, "q", ["a", "b"], rules=lambda _s: "b")
    assert seen == [first, second]
    assert seen[0] is first


def test_on_decision_fires_for_scores_too() -> None:
    seen: list[Choice | Score] = []
    d = Decider("rules", on_decision=seen.append)
    result = d.score({}, "r", rules=lambda _s: 4.0)
    assert seen == [result]
    assert isinstance(seen[0], Score)


def test_on_decision_fires_on_degraded_answers_as_well(mock_http) -> None:
    seen: list[Choice | Score] = []
    mock_http.http_error(500)
    d = Decider("auto", api_key="sk", timeout_s=0.5, on_decision=seen.append)
    result = d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert len(seen) == 1
    assert seen[0] is result
    assert seen[0].degraded is True


def test_the_callback_sees_everything_a_dashboard_needs() -> None:
    seen: list[dict] = []
    d = Decider("rules", on_decision=lambda decision: seen.append(decision.to_dict()))
    d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    payload = seen[0]
    assert {"value", "probs", "confidence", "backend", "degraded", "latency_ms"} <= set(payload)
    json.dumps(payload)  # straight into a log line or a websocket frame


def test_a_broken_callback_cannot_break_the_decision() -> None:
    def explode(_decision: Choice | Score) -> None:
        raise RuntimeError("dashboard is down")

    result = Decider("rules", on_decision=explode).choice({}, "q", ["a"], rules=lambda _s: "a")
    assert result.value == "a"


def test_health_reports_every_backend_without_touching_the_network(mock_http) -> None:
    report = Decider("auto").health()
    assert mock_http.count == 0
    assert set(report["backends"]) == {"jev", "llm", "rules"}
    assert report["backend"] == "auto"
    assert report["chain"] == ["jev", "llm", "rules"]
    assert report["timeout_s"] == 2.0
    assert report["total_budget_s"] == 4.0
    assert report["decisions"] == 0
    json.dumps(report)  # must be dashboard-ready as-is


def test_health_marks_unkeyed_backends_unavailable_and_says_why() -> None:
    backends = Decider("auto").health()["backends"]
    assert backends["jev"]["available"] is False
    assert "TYPESAFE_API_KEY" in backends["jev"]["reason"]
    assert backends["llm"]["available"] is False
    assert backends["rules"]["available"] is True


def test_health_marks_a_keyed_backend_available_and_shows_its_endpoint() -> None:
    backends = Decider("auto", api_key="sk-test").health()["backends"]
    assert backends["jev"]["available"] is True
    assert backends["jev"]["endpoint"] == "https://api.typesafe.ai/v1/systemone"
    assert backends["jev"]["model"] == "jev-latest"


def test_health_accumulates_latency_percentiles(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {"a": 0.9, "b": 0.1}, 0.6))
    d = Decider("auto", api_key="sk-test", timeout_s=1.0)
    for _ in range(3):
        d.choice({}, "q", ["a", "b"])

    report = d.health()
    assert report["decisions"] == 3
    jev_stats = report["backends"]["jev"]
    assert jev_stats["attempts"] == 3
    assert jev_stats["ok"] == 3
    assert jev_stats["failures"] == 0
    assert jev_stats["last_error"] is None
    assert set(jev_stats["latency_ms"]) == {"p50", "p90", "p99"}
    assert all(value >= 0.0 for value in jev_stats["latency_ms"].values())
    # llm was never reached because jev answered first.
    assert report["backends"]["llm"]["attempts"] == 0
    assert report["backends"]["llm"]["latency_ms"] == {}


def test_health_records_the_last_error(mock_http) -> None:
    mock_http.http_error(401, "bad key")
    d = Decider("auto", api_key="sk-wrong", timeout_s=0.5)
    d.choice({}, "q", ["a"], rules=lambda _s: "a")
    jev_stats = d.health()["backends"]["jev"]
    assert jev_stats["failures"] == 1
    assert "HTTP 401" in (jev_stats["last_error"] or "")


def test_health_marks_backends_outside_the_chain() -> None:
    report = Decider("rules").health()
    assert report["backends"]["rules"]["in_chain"] is True
    assert report["backends"]["jev"]["in_chain"] is False


def test_two_deciders_keep_separate_state(mock_http) -> None:
    a = Decider("rules")
    b = Decider("rules")
    a.choice({}, "q", ["x"], rules=lambda _s: "x")
    assert a.health()["decisions"] == 1
    assert b.health()["decisions"] == 0


@pytest.mark.parametrize("backend", ["auto", "jev", "llm", "rules"])
def test_health_is_json_serialisable_for_every_backend(backend: str) -> None:
    json.dumps(Decider(backend).health(), allow_nan=False)
