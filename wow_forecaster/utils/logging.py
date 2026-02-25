"""
Structured logging setup for the WoW Economy Forecaster.

Call ``configure_logging(config)`` once at CLI entry (before any pipeline work)
to set up the root logger with the configured level and optional file handler.

All internal modules use ``logging.getLogger(__name__)`` — never call
``configure_logging`` or ``basicConfig`` from within library code.

JSON format (set ``json_format = true`` in config/default.toml [logging]):
  Emits one JSON object per line, suitable for log aggregation tools
  (Datadog, CloudWatch, Loki, etc.)::

    {"ts": "2026-02-24T15:00:00Z", "level": "INFO", "logger": "...", "msg": "..."}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_forecaster.config import LoggingConfig

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line.

    Fields: ``ts``, ``level``, ``logger``, ``msg``.
    Extra fields from ``extra=`` kwargs are included at the top level.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
                LOG_DATE_FORMAT
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Include any extra= kwargs passed to the logger call
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


def configure_logging(config: "LoggingConfig") -> None:
    """Configure the root logger from a ``LoggingConfig`` instance.

    Sets up:
      - StreamHandler (stdout) at the configured level.
      - Optional FileHandler if ``config.log_file`` is set.
      - JSON line format if ``config.json_format`` is ``True``.

    Args:
        config: Logging configuration section from ``AppConfig``.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)

    formatter: logging.Formatter
    if config.json_format:
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    handlers: list[logging.Handler] = []

    # Console handler (stdout)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    handlers.append(console)

    # File handler (create parent dirs if needed)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    logging.getLogger("pyarrow").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
