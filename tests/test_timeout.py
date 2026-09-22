"""Timeouts degrade immediately, and ``auto`` never spends more than 2x ``timeout_s``."""

from __future__ import annotations

import time

from conftest import llm_body
from jev_decide import Decider

TIMEOUT_S = 0.4

#: Headroom for the interpreter itself: walking the fallback chain, two socket
#: timeouts and the bookkeeping around them all cost real milliseconds. Asserting
#: "budget + this" tests that the fallbacks do not multiply the budget; asserting
#: the bare budget would test that Python takes no time at all.
_OVERHEAD_MS = 20.0


def test_a_hanging_backend_degrades_to_the_next_one(mock_http) -> None:
    mock_http.hangs()
    mock_http.json(llm_body({"choice": "a", "probabilities": {"a": 0.8, "b": 0.2}}))
    result = Decider("auto", api_key="sk", timeout_s=0.2).choice({}, "q", ["a", "b"])
    assert result.backend == "llm"
    assert result.degraded is True
    assert "jev" in result.note


def test_the_whole_call_stays_inside_twice_the_timeout(mock_http) -> None:
    mock_http.hangs()  # every backend hangs until its own deadline
    d = Decider("auto", api_key="sk", timeout_s=TIMEOUT_S)

    started = time.monotonic()
    result = d.choice({}, "q", ["a", "b"], rules=lambda _s: "b")
    elapsed = time.monotonic() - started

    assert elapsed <= 2 * TIMEOUT_S + _OVERHEAD_MS / 1000.0, (
        f"auto took {elapsed:.3f}s, budget is {2 * TIMEOUT_S}s"
    )
    assert result.backend == "rules"
    assert result.value == "b"
    assert result.degraded is True


def test_no_single_attempt_exceeds_the_per_backend_timeout(mock_http) -> None:
    mock_http.hangs()
    d = Decider("auto", api_key="sk", timeout_s=TIMEOUT_S)
    d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")

    timeouts = [request.timeout for request in mock_http.requests]
    assert timeouts, "the network backends should have been attempted"
    assert all(t is not None and t <= TIMEOUT_S for t in timeouts)
    assert sum(timeouts) <= 2 * TIMEOUT_S


def test_score_honours_the_same_budget(mock_http) -> None:
    mock_http.hangs()
    d = Decider("auto", api_key="sk", timeout_s=TIMEOUT_S)

    started = time.monotonic()
    result = d.score({}, "how urgent", 0.0, 10.0, rules=lambda _s: 3.0)
    elapsed = time.monotonic() - started

    assert elapsed <= 2 * TIMEOUT_S + _OVERHEAD_MS / 1000.0
    assert result.backend == "rules"
    assert result.value == 3.0


def test_the_rules_backend_is_reached_even_with_no_budget_left(mock_http) -> None:
    mock_http.hangs()
    d = Decider("auto", api_key="sk", timeout_s=0.05, fallback_chain=("jev", "llm", "rules"))
    result = d.choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert result.value == "a"


def test_latency_ms_covers_the_whole_call_not_just_the_last_hop(mock_http) -> None:
    """``latency_ms`` covers the whole ``choice()`` call, not just the last hop.

    The ceiling is the documented ``2 * timeout_s`` budget plus ``_OVERHEAD_MS``;
    with a 0.5 s budget that allowance is about 4%.
    """
    mock_http.hangs()
    timeout_s = 0.5
    d = Decider("auto", api_key="sk", timeout_s=timeout_s)
    result = d.choice({}, "q", ["a"], rules=lambda _s: "a")
    assert result.latency_ms >= timeout_s * 1000.0
    assert result.latency_ms <= 2 * timeout_s * 1000.0 + _OVERHEAD_MS
