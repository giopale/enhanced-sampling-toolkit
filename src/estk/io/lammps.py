"""Reading simulation settings from a LAMMPS log file."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from estk.errors import MissingFileError
from estk.logging import get_logger

log = get_logger(__name__)

# Thermostat fix styles whose arguments start with Tstart Tstop (no "temp" keyword).
_TEMP_FIX_STYLES = ("langevin", "temp/csvr", "temp/csld", "temp/berendsen", "temp/rescale")
_VAR_REF = re.compile(r"\$\{(\w+)\}|\$(\w)")


@dataclass
class LammpsMetadata:
    """Settings in effect for the last ``run`` command of a LAMMPS log.

    ``timestep`` is in the time unit of ``units`` (ps for ``metal``); ``None`` means not found.
    """

    units: str | None = None
    timestep: float | None = None
    temperature: float | None = None
    warnings: list[str] = field(default_factory=list)


def read_log_metadata(path: Path) -> LammpsMetadata:
    """Read units, timestep and thermostat temperature from ``log.lammps``.

    Commands are read by keyword, not by column position, and simple ``variable`` definitions
    (``equal``, ``index``, ``string``) are substituted. When the input has several ``run``
    commands, the values in effect at the last one are returned.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"LAMMPS log not found: {path}")

    state = _State()
    at_last_run: dict[str, object] | None = None
    with open(path, errors="replace") as f:
        for raw in f:
            # Thermo output and other non-command lines: skip quickly.
            line = raw.lstrip()
            if not line[:1].isalpha():
                continue
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            words = line.split()
            command = words[0].lower()
            if command == "variable":
                state.define_variable(words)
            elif command == "units" and len(words) > 1:
                state.units = words[1]
            elif command == "timestep" and len(words) > 1:
                value = state.number(words[1])
                if value is not None:
                    state.timestep = value
            elif command == "fix":
                state.read_fix(words)
            elif command == "unfix" and len(words) > 1:
                state.temperatures.pop(words[1], None)
            elif command == "run":
                at_last_run = state.snapshot()

    values = at_last_run if at_last_run is not None else state.snapshot()
    meta = LammpsMetadata(
        units=values["units"],  # type: ignore[arg-type]
        timestep=values["timestep"],  # type: ignore[arg-type]
    )
    temps: dict[str, tuple[float, float]] = values["temperatures"]  # type: ignore[assignment]
    if len(temps) > 1:
        meta.warnings.append(
            f"several thermostats active ({', '.join(temps)}); using fix {next(iter(temps))}"
        )
    if temps:
        t_start, t_stop = next(iter(temps.values()))
        if t_start != t_stop:
            meta.warnings.append(f"thermostat ramps from {t_start} to {t_stop} K; using {t_start}")
        meta.temperature = t_start
    for warning in meta.warnings:
        log.warning("%s: %s", path, warning)
    return meta


class _State:
    def __init__(self) -> None:
        self.units: str | None = None
        self.timestep: float | None = None
        self.temperatures: dict[str, tuple[float, float]] = {}
        self.variables: dict[str, str] = {}

    def snapshot(self) -> dict[str, object]:
        return {
            "units": self.units,
            "timestep": self.timestep,
            "temperatures": dict(self.temperatures),
        }

    def define_variable(self, words: list[str]) -> None:
        if len(words) >= 4 and words[2] in ("equal", "index", "string"):
            value = words[3]
            if words[2] == "equal":
                value = value.strip("\"'")
            # LAMMPS keeps the first definition of index/string variables.
            if words[2] == "equal" or words[1] not in self.variables:
                self.variables[words[1]] = value

    def substitute(self, word: str) -> str:
        return _VAR_REF.sub(
            lambda m: self.variables.get(m.group(1) or m.group(2), m.group(0)), word
        )

    def number(self, word: str) -> float | None:
        word = self.substitute(word)
        if word.startswith("$(") and word.endswith(")"):
            word = word[2:-1]
        try:
            return float(word)
        except ValueError:
            return None

    def read_fix(self, words: list[str]) -> None:
        if len(words) < 4:
            return
        fix_id, style = words[1], words[3].lower()
        args = words[4:]
        if "temp" in args and style in ("nvt", "npt", "nvt/sllod", "rigid/nvt", "rigid/npt"):
            i = args.index("temp")
            values = args[i + 1 : i + 3]
        elif style in _TEMP_FIX_STYLES:
            values = args[0:2]
        else:
            return
        numbers = [self.number(v) for v in values]
        if len(numbers) == 2 and None not in numbers:
            self.temperatures[fix_id] = (numbers[0], numbers[1])  # type: ignore[assignment]
