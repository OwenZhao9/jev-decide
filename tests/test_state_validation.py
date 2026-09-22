"""``state`` must be JSON serialisable -- and must never be silently ``str()``-ed."""

from __future__ import annotations

import pytest

from jev_decide import ConfigError, Decider


class Opaque:
    def __repr__(self) -> str:  # pragma: no cover - only reached if we stringify
        return "Opaque()"

    def __str__(self) -> str:  # pragma: no cover
        return "Opaque()"


@pytest.fixture
def decider() -> Decider:
    return Decider("rules")


@pytest.mark.parametrize(
    "state",
    [
        {"obj": Opaque()},
        {"nested": {"deep": [1, 2, Opaque()]}},
        {"set": {1, 2, 3}},
        {"bytes": b"raw"},
        {"nan": float("nan")},
        {"inf": float("inf")},
        {1: "int key"},
        {"inner": {2: "int key"}},
    ],
)
def test_non_json_state_raises_value_error(decider: Decider, state: dict) -> None:
    with pytest.raises(ValueError):
        decider.choice(state, "q", ["a", "b"], rules=lambda _s: "a")


def test_the_object_is_not_stringified_into_the_request(decider: Decider) -> None:
    seen: list[object] = []
    with pytest.raises(ConfigError):
        decider.choice({"obj": Opaque()}, "q", ["a", "b"], rules=lambda s: seen.append(s) or "a")
    assert seen == [], "the rules function must never see a state we refused to accept"


def test_error_message_points_at_the_offending_path(decider: Decider) -> None:
    with pytest.raises(ConfigError) as info:
        decider.choice({"a": {"b": [0, Opaque()]}}, "q", ["x"], rules=lambda _s: "x")
    message = str(info.value)
    assert "'a'" in message and "'b'" in message and "[1]" in message
    assert "Opaque" in message


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"a": 1, "b": 2.5, "c": "text", "d": True, "e": None},
        {"list": [1, "two", {"three": 3}], "nested": {"deep": {"deeper": [True, None]}}},
        {"unicode": "倾角 31.0 度"},
    ],
)
def test_json_safe_state_is_accepted(decider: Decider, state: dict) -> None:
    result = decider.choice(state, "q", ["a", "b"], rules=lambda _s: "b")
    assert result.value == "b"


def test_score_validates_state_too(decider: Decider) -> None:
    with pytest.raises(ValueError):
        decider.score({"obj": Opaque()}, "rubric", rules=lambda _s: 1.0)


def test_state_must_be_a_mapping(decider: Decider) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        decider.choice("a bare string", "q", ["a"], rules=lambda _s: "a")  # type: ignore[arg-type]
