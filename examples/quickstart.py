"""Run me: ``uv run python examples/quickstart.py``

No API key needed.  With no key configured, ``auto`` walks past the network backends
and lands on your own rules function, telling you it degraded.  Export
``TYPESAFE_API_KEY`` (or ``OPENAI_API_KEY``) and the same code starts using the model
instead -- nothing else changes.
"""

from __future__ import annotations

import json

from jev_decide import Choice, Decider, Score

OPTIONS = ["off", "low", "high"]


def house_rules(state: dict) -> str:
    """The offline floor: a pure function the caller owns."""
    if state["battery_pct"] < 20 or state["tilt_deg"] > 40:
        return "off"
    return "low" if state["tilt_deg"] > 20 else "high"


def audit(decision: Choice | Score) -> None:
    """Every decision, on its way to a log line or a dashboard panel."""
    print("  audit:", json.dumps(decision.to_dict(), ensure_ascii=False)[:160])


def main() -> None:
    decider = Decider("auto", timeout_s=1.5, on_decision=audit)
    state = {"battery_pct": 62, "tilt_deg": 31.0, "gait": "stairs_up"}

    print("choice:")
    picked = decider.choice(state, "What assist level fits this state?", OPTIONS,
                            rules=house_rules)
    print(f"  value={picked.value} confidence={picked.confidence:.2f} "
          f"backend={picked.backend} degraded={picked.degraded}")
    if picked.note:
        print(f"  note={picked.note}")

    # The whole point: do not move unless the answer is confident enough.
    action = Decider.gate(picked, min_confidence=0.7, on_low="keep", current="off")
    print(f"  gated action -> {action}")

    print("score:")
    risk = decider.score(state, "How risky is this posture right now?", 0.0, 10.0,
                         rules=lambda s: s["tilt_deg"] / 5.0)
    print(f"  value={risk.value:.2f} in [{risk.lo}, {risk.hi}] "
          f"confidence={risk.confidence:.2f} backend={risk.backend}")

    print("health:")
    for name, info in decider.health()["backends"].items():
        print(f"  {name}: available={info['available']} ({info['reason']})")


if __name__ == "__main__":
    main()
