"""What the estk text files (surfaces, paths) share: the ``#! FIELDS`` / ``#! SET`` header,
metadata as ``#! SET estk.<key> <JSON>``, provenance, and gzip compression for names ending in
``.gz``."""

from __future__ import annotations

import gzip
import json
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np

from estk.errors import EstError, MissingFileError

PREFIX = "estk."
# Written by every save from the current session; values read from a file are replaced.
PROVENANCE_KEYS = ("version", "command")

_command: str | None = None


def set_command(argv: Sequence[str] | None) -> None:
    """Record the command line that is running (set by the CLI), written to every file."""
    global _command
    _command = shlex.join(argv) if argv else None


def provenance() -> dict[str, Any]:
    from estk import __version__

    meta: dict[str, Any] = {"version": __version__}
    if _command:
        meta["command"] = _command
    return meta


def open_text(path: Path, mode: str = "r") -> IO[str]:
    """Open a text file, compressed with gzip when its name ends in ``.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t")  # type: ignore[return-value]
    return open(path, mode, errors="replace" if mode == "r" else "strict")


def base_name(path: Path) -> str:
    """The name without ``.gz`` and the extension: ``combined.dat.gz`` -> ``combined``."""
    name = path.name.removesuffix(".gz")
    return Path(name).stem


def read_header(path: Path) -> tuple[list[str], dict[str, str]]:
    """``FIELDS`` names and raw ``SET`` values (``#!`` or plain ``#`` headers)."""
    if not path.is_file():
        raise MissingFileError(f"file not found: {path}")
    fields: list[str] = []
    sets: dict[str, str] = {}
    with open_text(path) as f:
        for line in f:
            if not line.startswith("#"):
                if line.strip():
                    break  # the header is over
                continue
            words = line.lstrip("#!").split(maxsplit=2)
            if len(words) >= 2 and words[0] == "FIELDS":
                fields = line.lstrip("#!").split()[1:]
            elif len(words) == 3 and words[0] == "SET":
                sets[words[1]] = words[2].strip()
    if not fields:
        raise EstError(f"{path}: no FIELDS header")
    return fields, sets


def metadata_lines(metadata: dict[str, Any]) -> list[str]:
    """``#! SET estk.<key> <JSON>`` lines, current provenance first."""
    meta = {**provenance()}
    meta.update({k: v for k, v in metadata.items() if k not in PROVENANCE_KEYS})
    return [f"#! SET {PREFIX}{k} {json.dumps(jsonable(v))}" for k, v in meta.items()]


def decode(raw: str) -> Any:
    return _restore(json.loads(raw))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)  # "inf", "-inf", "nan": strict JSON has no such numbers
    return value


def _restore(value: Any) -> Any:
    if isinstance(value, str) and value in ("inf", "-inf", "nan"):
        return float(value)
    if isinstance(value, dict):
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value
