"""The ``jev`` backend against a mocked TypeSafe System One endpoint.

The request shape asserted here is the one documented at
https://docs.typesafe.ai/api -- endpoint, bearer auth, ``state`` / ``model`` /
``questions``, and the choice / score answer objects.
"""

from __future__ import annotations

import pytest

from conftest import jev_choice_body, jev_score_body
from jev_decide import Decider


def jev(**kwargs: object) -> Decider:
    options: dict[str, object] = {"api_key": "sk-test", "timeout_s": 1.0}
    options.update(kwargs)
    return Decider("jev", **options)  # type: ignore[arg-type]


def test_choice_posts_the_documented_request(mock_http) -> None:
    mock_http.json(
        jev_choice_body("billing", {"billing": 0.88, "technical": 0.12, "sales": 0.0}, 0.81)
    )
    result = jev().choice(
        {"ticket": "payouts failing"},
        "Which team should handle this?",
        ["billing", "technical", "sales"],
    )

    request = mock_http.last
    assert request.url == "https://api.typesafe.ai/v1/systemone"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert request.headers["content-type"] == "application/json"
    assert request.body["model"] == "jev-latest"
    assert request.body["state"] == {"ticket": "payouts failing"}

    question = next(iter(request.body["questions"].values()))
    assert question["type"] == "choice"
    assert question["instructions"] == "Which team should handle this?"
    assert question["criteria"] == {"billing": None, "technical": None, "sales": None}

    assert result.value == "billing"
    assert result.backend == "jev"
    assert result.degraded is False
    assert result.confidence == pytest.approx(0.81)
    assert result.probs == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert result.raw is not None and result.raw["model"] == "jev-1.13.0"


def test_probabilities_are_renormalised_and_projected_onto_the_options(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {"a": 6.0, "b": 2.0, "ghost": 92.0}))
    result = jev().choice({}, "q", ["a", "b"])
    assert result.probs == {"a": pytest.approx(0.75), "b": pytest.approx(0.25)}
    assert sum(result.probs.values()) == pytest.approx(1.0)


def test_confidence_falls_back_to_the_distribution_shape(mock_http) -> None:
    # Neither answer carries a ``confidence`` field, so it is derived from the shape.
    mock_http.json(jev_choice_body("a", {"a": 0.5, "b": 0.5}))
    mock_http.json(jev_choice_body("a", {"a": 1.0, "b": 0.0}))

    flat = jev().choice({}, "q", ["a", "b"])
    assert flat.confidence == pytest.approx(0.0, abs=1e-9)

    peaked = jev().choice({}, "q", ["a", "b"])
    assert peaked.confidence == pytest.approx(1.0)


def test_an_unknown_option_falls_back_to_the_argmax_and_says_so(mock_http) -> None:
    mock_http.json(jev_choice_body("sideways", {"a": 0.2, "b": 0.8}))
    result = jev().choice({}, "q", ["a", "b"])
    assert result.value == "b"
    assert result.degraded is True
    assert "unknown option" in result.note


def test_score_maps_the_level_index_onto_the_callers_scale(mock_http) -> None:
    # 5 levels -> index 0..4; index 2 is the midpoint of [0, 10].
    mock_http.json(jev_score_body(2.0, {"0": 0.0, "1": 0.1, "2": 0.8, "3": 0.1, "4": 0.0}, 0.7))
    result = jev().score({"text": "hello"}, "how frustrated is the customer", 0.0, 10.0)

    question = next(iter(mock_http.last.body["questions"].values()))
    assert question["type"] == "score"
    assert question["instructions"] == "how frustrated is the customer"
    assert isinstance(question["criteria"], list) and len(question["criteria"]) == 5

    assert result.value == pytest.approx(5.0)
    assert result.confidence == pytest.approx(0.7)
    assert result.backend == "jev"
    assert result.degraded is False


@pytest.mark.parametrize(
    ("level", "expected"),
    [(0.0, 0.0), (1.0, 2.5), (4.0, 10.0), (99.0, 10.0), (-3.0, 0.0)],
)
def test_score_index_mapping_is_linear_and_clamped(
    mock_http, level: float, expected: float
) -> None:
    mock_http.json(jev_score_body(level, {"0": 1.0}))
    assert jev().score({}, "r", 0.0, 10.0).value == pytest.approx(expected)


def test_score_respects_a_custom_range(mock_http) -> None:
    mock_http.json(jev_score_body(4.0, {"4": 1.0}))
    assert jev().score({}, "r", -1.0, 1.0).value == pytest.approx(1.0)


def test_a_custom_base_url_is_honoured(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {"a": 1.0}))
    jev(base_url="https://gateway.internal/typesafe").choice({}, "q", ["a"])
    assert mock_http.last.url == "https://gateway.internal/typesafe/v1/systemone"


def test_a_base_url_that_already_carries_the_path_is_not_doubled(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {"a": 1.0}))
    jev(base_url="https://gw.internal/v1/systemone").choice({}, "q", ["a"])
    assert mock_http.last.url == "https://gw.internal/v1/systemone"


def test_a_custom_model_is_sent(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {"a": 1.0}))
    jev(model="jev-1.13.0").choice({}, "q", ["a"])
    assert mock_http.last.body["model"] == "jev-1.13.0"


def test_the_api_key_can_come_from_the_environment(mock_http, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-from-env")
    mock_http.json(jev_choice_body("a", {"a": 1.0}))
    Decider("jev").choice({}, "q", ["a"])
    assert mock_http.last.headers["authorization"] == "Bearer sk-from-env"


def test_without_a_key_the_backend_is_skipped_entirely(mock_http) -> None:
    result = Decider("jev").choice({}, "q", ["a", "b"], rules=lambda _s: "b")
    assert mock_http.count == 0, "a backend with no key must not open a connection"
    assert result.backend == "rules"
    assert result.degraded is True
    assert "no API key" in result.note


@pytest.mark.parametrize("status", [401, 422, 429, 529, 500])
def test_http_errors_degrade_instead_of_raising(mock_http, status: int) -> None:
    mock_http.http_error(status, '{"detail":"nope"}')
    result = jev().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert result.degraded is True
    assert f"HTTP {status}" in result.note


def test_an_unreachable_host_degrades(mock_http) -> None:
    mock_http.unreachable("name resolution failed")
    result = jev().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert "could not reach" in result.note


@pytest.mark.parametrize(
    "body",
    [
        {"model": "jev", "answers": {}},
        {"model": "jev", "answers": {"other": {"type": "choice"}}},
        {"model": "jev"},
        {"model": "jev", "answers": {"decision": "not an object"}},
    ],
)
def test_a_malformed_answer_degrades(mock_http, body: dict) -> None:
    mock_http.json(body)
    result = jev().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert result.degraded is True


def test_a_non_json_body_degrades(mock_http) -> None:
    mock_http.text("<html>502 Bad Gateway</html>")
    result = jev().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert "non-JSON" in result.note


def test_a_choice_with_no_usable_probabilities_reports_zero_confidence(mock_http) -> None:
    mock_http.json(jev_choice_body("a", {}, confidence=0.99))
    result = jev().choice({}, "q", ["a", "b"])
    assert result.value == "a"
    assert result.confidence == 0.0
    assert result.degraded is True
    assert "no usable probability" in result.note


def test_a_non_numeric_score_degrades(mock_http) -> None:
    mock_http.json(jev_score_body("high", {"0": 1.0}))  # type: ignore[arg-type]
    result = jev().score({}, "r", rules=lambda _s: 1.0)
    assert result.backend == "rules"
    assert result.degraded is True
