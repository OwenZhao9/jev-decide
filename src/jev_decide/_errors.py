"""Construction-time error types.

Per the shared four-library contract (section 0.4) every library defines its own
``<Lib>Error(Exception)`` base class, and it is used **only** for construction-time /
argument validation.  Nothing in this package raises at run time: a backend that is
unreachable degrades instead (see :mod:`jev_decide.decider`).

``ConfigError`` deliberately inherits from both :class:`JevDecideError` and
:class:`ValueError` so that the contract's "constructor argument errors raise
``ValueError``" rule and the "library defines its own error base class" rule are both
satisfied by a single exception type.
"""

from __future__ import annotations

__all__ = ["ConfigError", "JevDecideError"]


class JevDecideError(Exception):
    """Base class for every error raised by ``jev_decide``.

    Only construction-time / argument validation raises.  Run-time failures (timeouts,
    HTTP errors, unavailable backends) never raise: they degrade and are reported via
    ``degraded=True`` and ``note`` on the returned :class:`~jev_decide.Choice` /
    :class:`~jev_decide.Score`.
    """


class ConfigError(JevDecideError, ValueError):
    """Invalid argument passed to a ``jev_decide`` entry point.

    Also a :class:`ValueError`, so ``except ValueError`` catches it.
    """
