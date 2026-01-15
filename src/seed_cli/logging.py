

"""seed_cli.logging

Centralized logging configuration for seed-cli.

Design goals:
- One entry point: setup_logging()
- Respect --verbose / --debug flags
- Human-friendly by default
- Library-safe (does not clobber existing loggers)
"""

import logging
from typing import Optional


_LOGGER_NAME = "seed_cli"


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging for seed-cli.

    Levels:
    - default: WARNING
    - --verbose: INFO
    - --debug: DEBUG
    """
    level = logging.WARNING
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Make seed_cli logs propagate only within its namespace
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a namespaced seed-cli logger."""
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)
