"""Strict JSON-serialisability checking for caller-supplied ``state``.

The contract is explicit: ``state`` must be JSON serialisable (numbers, strings,
booleans, lists, dicts).  Anything else raises ``ValueError`` -- we must **never**
silently ``str()`` an object into the payload, because that would quietly change what
the decision was made on.

``json.dumps`` alone is not strict enough for this:

* ``json.dumps({1: "a"})`` silently coerces the ``int`` key to ``"1"``;
* ``json.dumps(float("nan"))`` emits bare ``NaN``, which is not valid JSON.

So we walk the structure ourselves and report the exact path of the offending value.
"""

from __future__ import annotations

import math
from typing import Any

from ._errors import ConfigError

__all__ = ["ensure_json_safe"]

_MAX_DEPTH = 64


def _type_name(value: object) -> str:
    return type(value).__name__


def _walk(value: Any, path: str, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise ConfigError(
            f"state is nested deeper than {_MAX_DEPTH} levels at {path}; "
            "flatten it before passing it in"
        )

    if value is None or isinstance(value, (str, bool, int)):
        # bool is a subclass of int; both are JSON scalars.
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(
                f"state{path} is {value!r}, which is not valid JSON; "
                "NaN and Infinity cannot be serialised"
            )
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(
                    f"state{path} has a non-string key {key!r} ({_type_name(key)}); "
                    "JSON object keys must be strings -- convert it yourself rather than "
                    "letting it be coerced silently"
                )
            _walk(item, f"{path}[{key!r}]", depth + 1)
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", depth + 1)
        return

    raise ConfigError(
        f"state{path} is a {_type_name(value)}, which is not JSON serialisable. "
        "state must contain only numbers, strings, booleans, None, lists and dicts. "
        "Convert it explicitly -- jev_decide will not str() it for you."
    )


def ensure_json_safe(state: Any, *, name: str = "state") -> None:
    """Raise :class:`ConfigError` (a ``ValueError``) if ``state`` is not JSON safe.

    Parameters
    ----------
    state:
        The value to validate.
    name:
        Name used in the error message (defaults to ``"state"``).
    """
    try:
        _walk(state, "", 0)
    except ConfigError as exc:
        if name != "state":
            raise ConfigError(str(exc).replace("state", name, 1)) from None
        raise exc from None
