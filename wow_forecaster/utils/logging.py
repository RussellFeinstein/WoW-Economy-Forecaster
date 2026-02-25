"""
Structured logging setup for the WoW Economy Forecaster.

Call ``configure_logging(config)`` once at CLI entry (before any pipeline work)
to set up the root logger with the configured level and optional file handler.

All internal modules use ``logging.getLogger(__name__)`` — never call
``configure_logging`` or ``basicConfig`` from within library code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_forecaster.config import LoggingConfig

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def configure_logging(config: "LoggingConfig") -> None:
    """Configure the root logger from a ``LoggingConfig`` instance.

    Sets up:
      - StreamHandler (stdout) at the configured level.
      - Optional FileHandler if ``config.log_file`` is set.

    Args:
        config: Logging configuration section from ``AppConfig``.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    handlers.append(console)

    # File handler (create parent dirs if needed)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Quieten noisy third-party loggers
    logging.getLogger("pyarrow").setLevel(logging.WARNING)
