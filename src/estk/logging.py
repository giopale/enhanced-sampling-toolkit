"""Logging for estk.

Library modules call ``get_logger(__name__)`` and never add handlers, so importing estk does not
change the logging set-up of the program that uses it. Only the CLI calls
``setup_cli_logging``, which attaches a single handler to the ``estk`` logger.
"""

from __future__ import annotations

import logging

ROOT_LOGGER_NAME = "estk"

_VERBOSITY_LEVELS = {-1: logging.WARNING, 0: logging.INFO, 1: logging.DEBUG}


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a logger in the ``estk`` hierarchy."""
    if name != ROOT_LOGGER_NAME and not name.startswith(ROOT_LOGGER_NAME + "."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def setup_cli_logging(verbosity: int = 0) -> logging.Logger:
    """Send ``estk`` log records to the terminal.

    ``verbosity`` is -1 (warnings only), 0 (info, the default) or 1 (debug). Calling this again
    replaces the handler instead of adding a second one.
    """
    from rich.logging import RichHandler

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    level = _VERBOSITY_LEVELS[max(-1, min(1, verbosity))]
    for handler in [h for h in logger.handlers if getattr(h, "_estk_cli", False)]:
        logger.removeHandler(handler)

    handler = RichHandler(show_time=False, show_path=verbosity > 0, markup=False)
    handler._estk_cli = True
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
