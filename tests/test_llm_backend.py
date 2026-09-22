"""The ``llm`` backend against a mocked OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import math

import pytest

from conftest import llm_body
from jev_decide import Decider


def llm(**kwargs: object) -> Decider:
    options: dict[str, object] = {"api_key": "sk-llm", "timeout_s": 1.0}
    options.update(kwargs)
    return Decider("llm", **options)  # type: ignore[arg-type]


def test_choice_pins_the_answer_to_a_json_schema(mock_http) -> None:
    mock_http.json(
        llm_body({"choice": "low", "probabilities": {"off": 0.1, "low": 0.7, "high": 0.2}})
    )
    result = llm().choice({"battery_pct": 18}, "assist level?", ["off", "low", "high"])

    request = mock_http.last
    assert request.url == "https://api.openai.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-llm"
    assert request.body["model"] == "gpt-4o-mini"
    assert request.body["temperature"] == 0

    schema = request.body["response_format"]["json_schema"]
    assert request.body["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["choice"]["enum"] == ["off", "low", "high"]
    assert schema["schema"]["additionalProperties"] is False
    assert sorted(schema["schema"]["properties"]["probabilities"]["required"]) == [
        "high",
        "low",
        "off",
    ]

    # The state travels as JSON inside the user message, not as a stringified object.
    user_message = request.body["messages"][-1]["content"]
    assert json.loads(user_message)["state"] == {"battery_pct": 18}

    assert result.value == "low"
    assert result.backend == "llm"
    assert result.degraded is False
    assert result.probs["low"] == pytest.approx(0.7)
    assert 0.0 < result.confidence < 1.0
    assert result.raw is not None and result.raw["parsed"]["choice"] == "low"


def test_confidence_comes_from_the_distribution_not_the_model_boasting(mock_http) -> None:
    mock_http.json(
        llm_body({"choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.99})
    )
    result = llm().choice({}, "q", ["a", "b"])
    assert result.confidence == pytest.approx(0.0, abs=1e-9)


def test_a_flat_distribution_is_reported_as_no_confidence(mock_http) -> None:
    mock_http.json(llm_body({"choice": "a", "probabilities": {"a": 0.25, "b": 0.25,
                                                              "c": 0.25, "d": 0.25}}))
    result = llm().choice({}, "q", ["a", "b", "c", "d"])
    assert result.confidence == pytest.approx(0.0, abs=1e-9)
    assert Decider.gate(result, min_confidence=0.6, on_low="first") == "a"


def test_score_asks_for_a_level_distribution_and_maps_it(mock_http) -> None:
    mock_http.json(llm_body({"probabilities": {"0": 0.0, "1": 0.0, "2": 1.0, "3": 0.0, "4": 0.0}}))
    result = llm().score({"text": "hi"}, "how urgent", 0.0, 10.0)

    schema = mock_http.last.body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["probabilities"]["required"] == ["0", "1", "2", "3", "4"]

    assert result.value == pytest.approx(5.0)
    assert result.confidence == pytest.approx(1.0)
    assert result.backend == "llm"


def test_score_is_the_probability_weighted_index(mock_http) -> None:
    # Split evenly between the two extremes: the value lands in the middle, and the
    # confidence reflects a distribution that is spread over two of five levels.
    mock_http.json(llm_body({"probabilities": {"0": 0.5, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.5}}))
    result = llm().score({}, "r", 0.0, 4.0)
    assert result.value == pytest.approx(2.0)
    assert result.confidence == pytest.approx(1.0 - math.log(2) / math.log(5))


def test_a_uniform_score_distribution_has_no_confidence(mock_http) -> None:
    flat = dict.fromkeys(("0", "1", "2", "3", "4"), 0.2)
    mock_http.json(llm_body({"probabilities": flat}))
    result = llm().score({}, "r", 0.0, 10.0)
    assert result.value == pytest.approx(5.0)
    assert result.confidence == pytest.approx(0.0, abs=1e-9)


PROSE_REPLY = {"choices": [{"message": {"role": "assistant", "content": "sure thing!"}}]}


def test_a_malformed_reply_is_repaired_and_retried_once(mock_http) -> None:
    mock_http.json(PROSE_REPLY).json(
        llm_body({"choice": "b", "probabilities": {"a": 0.1, "b": 0.9}})
    )
    result = llm().choice({}, "q", ["a", "b"])

    assert mock_http.count == 2, "the backend should repair and retry exactly once"
    assert result.value == "b"
    assert result.degraded is False

    messages = mock_http.last.body["messages"]
    assert len(messages) == 4
    assert messages[-2]["role"] == "assistant"
    assert "rejected" in messages[-1]["content"]


def test_two_bad_replies_degrade_to_rules(mock_http) -> None:
    mock_http.json(PROSE_REPLY)
    result = llm().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert mock_http.count == 2
    assert result.backend == "rules"
    assert result.degraded is True
    assert "not valid JSON" in result.note


def test_without_a_key_the_backend_is_skipped(mock_http) -> None:
    result = Decider("llm").choice({}, "q", ["a", "b"], rules=lambda _s: "b")
    assert mock_http.count == 0
    assert result.backend == "rules"
    assert "no API key" in result.note


def test_the_key_and_base_url_can_come_from_the_environment(mock_http, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evomap.example/v1")
    mock_http.json(llm_body({"choice": "a", "probabilities": {"a": 1.0}}))
    Decider("llm").choice({}, "q", ["a"])
    assert mock_http.last.url == "https://evomap.example/v1/chat/completions"
    assert mock_http.last.headers["authorization"] == "Bearer sk-env"


def test_any_openai_compatible_gateway_works(mock_http) -> None:
    mock_http.json(llm_body({"choice": "a", "probabilities": {"a": 1.0}}))
    llm(base_url="http://localhost:11434/v1", model="qwen3").choice({}, "q", ["a"])
    assert mock_http.last.url == "http://localhost:11434/v1/chat/completions"
    assert mock_http.last.body["model"] == "qwen3"


def test_content_returned_as_typed_parts_is_understood(mock_http) -> None:
    payload = json.dumps({"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}})
    mock_http.json(
        {
            "model": "m",
            "choices": [{"message": {"role": "assistant",
                                     "content": [{"type": "text", "text": payload}]}}],
        }
    )
    assert llm().choice({}, "q", ["a", "b"]).value == "a"


def test_an_http_error_degrades(mock_http) -> None:
    mock_http.http_error(429, "slow down")
    result = llm().choice({}, "q", ["a", "b"], rules=lambda _s: "a")
    assert result.backend == "rules"
    assert "HTTP 429" in result.note


def test_a_choice_outside_the_option_set_falls_back_to_the_argmax(mock_http) -> None:
    mock_http.json(llm_body({"choice": "banana", "probabilities": {"a": 0.3, "b": 0.7}}))
    result = llm().choice({}, "q", ["a", "b"])
    assert result.value == "b"
    assert result.degraded is True
