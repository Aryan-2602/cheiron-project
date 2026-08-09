"""Logging setup: JSON lines on stdout, one line per pipeline stage boundary.

The goal is legibility when running or debugging the service locally -- being
able to answer "which stage was slow" or "why did this request 500" from the
log alone. It is not production observability infrastructure, and deliberately
uses nothing beyond the standard library.

Two conventions make the output useful:

* **Messages are short constants; variable data goes in ``extra``.** That keeps
  lines greppable (``grep 'ctgov search completed'``) and machine-readable, and
  it means no payload is ever interpolated into prose.
* **Only counts, ids, durations, and truncated query text are logged.** Never a
  prompt, a completion, a raw API response, a trial record, or anything from
  config -- every ``extra`` payload is an explicit allowlist of fields, so a
  secret cannot leak by accident.
"""

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings

#: Correlates every line emitted while handling one request. Without it,
#: concurrent requests interleave into a single unreadable stream.
request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Attributes the logging module puts on every record. Anything outside this set
#: arrived via ``extra=`` and belongs in the JSON payload. Built once from a
#: throwaway record so it tracks the standard library rather than a hand-copied
#: list that drifts.
_RESERVED: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

TRUNCATION_MARKER = "...[truncated]"


def truncate(text: str, limit: int = 200) -> str:
    """Clip user-supplied text so a long query cannot dominate the log.

    The marker is explicit so a reader never mistakes a clipped value for the
    whole thing.
    """
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + TRUNCATION_MARKER


class JsonFormatter(logging.Formatter):
    """One JSON object per line, including any ``extra`` fields.

    Merging ``extra`` is what makes the logging structured rather than prose:
    ``logger.info("ctgov search completed", extra={"records": 600})`` emits
    ``records`` as a top-level key that can be filtered on.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # Second precision with a trailing Z, which is what the README
            # documents and what reads cleanly in a terminal. isoformat()
            # emitted "2026-08-09T23:41:02.123456+00:00" -- six digits of
            # precision nothing here needs, in a shape the docs did not match.
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        current_request = request_id.get()
        if current_request:
            payload["request_id"] = current_request

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # default=str so an unexpected non-serializable value degrades to its
        # repr instead of raising inside the logging call and losing the line.
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
