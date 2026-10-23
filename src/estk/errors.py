"""Exceptions raised by estk.

Library code raises these instead of calling ``sys.exit()``; the CLI turns them into a short
error message and a non-zero exit code.
"""

from __future__ import annotations

from pathlib import Path


class EstError(Exception):
    """Base class for all errors raised by estk."""


class ConfigError(EstError):
    """The project configuration file is missing, unreadable or invalid."""

    def __init__(self, message: str, problems: list[str] | None = None, path: Path | None = None):
        self.message = message
        self.problems = list(problems or [])
        self.path = path
        super().__init__(str(self))

    def __str__(self) -> str:
        where = f"{self.path}: " if self.path is not None else ""
        lines = [f"{where}{self.message}"]
        lines += [f"  - {p}" for p in self.problems]
        return "\n".join(lines)


class MissingFileError(EstError):
    """A file needed for the analysis does not exist."""


class MissingLimitsError(EstError):
    """The grid limits of a CV are not set and can't be inferred."""
