"""Application logging configuration."""

import logging
import sys


def setup_logging(log_level: str) -> None:
    """Configure structured application logging.

    Configures the root 'app' logger with a console handler that includes
    timestamps, log level, and logger name. Avoids duplicate handlers on
    repeated calls (e.g. during testing or hot-reload).

    Args:
        log_level: Logging level string (e.g. "INFO", "DEBUG").
    """
    logger = logging.getLogger("app")

    # Avoid adding duplicate handlers on restart / re-import
    if logger.handlers:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False
