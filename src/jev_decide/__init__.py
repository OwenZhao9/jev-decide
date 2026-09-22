"""jev-decide -- typed decisions with calibrated confidence.

Replace "make the model write a paragraph and parse it" with "ask one typed question
and get a choice, a probability distribution and a confidence back".  The point of the
library is the last of those three: :meth:`Decider.gate` refuses to act when the
answer is not confident enough, so an uncertain model does not move anything.

.. warning::

   ``Decider`` methods **block** (up to ``2 * timeout_s``) and must never be called
   from a real-time control loop.  Instances are not thread safe and take no locks.

Example
-------
::

    from jev_decide import Decider

    d = Decider()  # no key needed: falls back to your own rules function
    c = d.choice(
        {"tilt_deg": 31.0, "battery_pct": 18},
        "What should the assist level be?",
        ["off", "low", "high"],
        rules=lambda s: "low" if s["battery_pct"] < 20 else "high",
    )
    action = Decider.gate(c, min_confidence=0.7, on_low="keep", current="off")
"""

from __future__ import annotations

import logging

from ._errors import ConfigError, JevDecideError
from .decider import Decider
from .types import Choice, Score

__all__ = ["Choice", "ConfigError", "Decider", "JevDecideError", "Score"]

__version__ = "0.1.1"

# A library configures no handlers; the application does.
logging.getLogger(__name__).addHandler(logging.NullHandler())
