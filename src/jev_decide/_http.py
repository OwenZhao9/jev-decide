"""Minimal JSON-over-HTTP client built on the standard library.

Deliberately tiny and dependency-free: ``urllib.request`` is enough for a single
``POST`` with a hard timeout, and adding ``httpx`` would pull a dependency tree into a
library whose whole point is being safe to drop into a control-adjacent process.

Every failure funnels into :class:`HttpFailure`, which the backend layer turns into a
degraded result.  Nothing here ever escapes to the caller.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

__all__ = ["HttpFailure", "post_json"]

_log = logging.getLogger(__name__)

_MAX_BODY_BYTES = 4 * 1024 * 1024


class HttpFailure(Exception):
    """An HTTP call did not produce a usable JSON body.

    Internal control-flow only -- callers of :mod:`jev_decide` never see this.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response body.

    ``timeout_s`` is passed straight to ``urlopen`` and bounds connect + read.
    """
    try:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:  # pragma: no cover - state is pre-validated
        raise HttpFailure(f"request payload is not JSON serialisable: {exc}") from exc

    # The URL comes from the caller's own configuration, never from model output.
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", 200) or 200)
            raw = response.read(_MAX_BODY_BYTES)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(2048).decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - body already consumed
            detail = ""
        raise HttpFailure(
            f"HTTP {exc.code} from {_host(url)}{': ' + detail if detail else ''}",
            status=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise HttpFailure(f"could not reach {_host(url)}: {reason}") from exc

    if status >= 400:
        raise HttpFailure(f"HTTP {status} from {_host(url)}", status=status)

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HttpFailure(f"{_host(url)} returned a non-JSON body: {exc}", status=status) from exc

    if not isinstance(decoded, dict):
        raise HttpFailure(
            f"{_host(url)} returned a JSON {type(decoded).__name__}, expected an object",
            status=status,
        )
    _log.debug("POST %s -> %s", url, status)
    return decoded


def _host(url: str) -> str:
    """Host portion of ``url``, so error messages never leak a key in a query string."""
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError:  # pragma: no cover - urlsplit is very forgiving
        return url
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}" if parsed.netloc else url
