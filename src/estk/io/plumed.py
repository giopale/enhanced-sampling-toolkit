"""Reading settings from a PLUMED input file and OPES/METAD state files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from estk.errors import EstError, MissingFileError
from estk.logging import get_logger

log = get_logger(__name__)

BIAS_ACTIONS = ("OPES_METAD", "OPES_METAD_EXPLORE", "METAD", "PBMETAD", "OPES_EXPANDED")
WALL_ACTIONS = ("UPPER_WALLS", "LOWER_WALLS")


@dataclass
class Action:
    """One PLUMED action: ``label: NAME KEY=value FLAG ...``."""

    name: str
    label: str | None
    keywords: dict[str, str]
    flags: list[str]

    def get(self, key: str) -> str | None:
        return self.keywords.get(key.upper())

    def get_list(self, key: str) -> list[str]:
        value = self.get(key)
        return [] if value is None else [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Bias:
    """A biasing action (OPES, METAD, ...)."""

    action: str
    label: str
    args: list[str]
    bias_factor: float | None = None
    temperature: float | None = None
    state_file: str | None = None

    @property
    def bias_column(self) -> str:
        return f"{self.label}.bias"


@dataclass
class PlumedInput:
    """The settings of a PLUMED input that matter for analysis.

    ``walls`` maps a CV name to ``(lower, upper)`` positions of its LOWER/UPPER_WALLS (None when
    that side has no wall). Units are PLUMED's (``time_units``, ``energy_units``).
    """

    path: Path
    actions: list[Action]
    time_units: str = "ps"
    energy_units: str = "kj/mol"
    biases: list[Bias] = field(default_factory=list)
    walls: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    wall_bias_columns: list[str] = field(default_factory=list)

    @property
    def bias(self) -> Bias | None:
        """The main bias: the only one, or the first if there are several."""
        return self.biases[0] if self.biases else None


def read_input(path: Path) -> PlumedInput:
    """Parse a PLUMED input file (``INCLUDE`` and multi-line ``...`` blocks are followed)."""
    path = Path(path)
    actions = parse_actions(path)
    plumed = PlumedInput(path=path, actions=actions)

    for action in actions:
        if action.name == "UNITS":
            if "NATURAL" in action.flags:
                raise EstError(f"{path}: UNITS NATURAL is not supported")
            plumed.time_units = action.get("TIME") or plumed.time_units
            plumed.energy_units = action.get("ENERGY") or plumed.energy_units
        elif action.name in BIAS_ACTIONS:
            plumed.biases.append(_read_bias(action, path))
        elif action.name in WALL_ACTIONS:
            _read_walls(action, plumed, path)

    if len(plumed.biases) > 1:
        names = ", ".join(b.label for b in plumed.biases)
        log.warning("%s: several biases (%s); using %s", path, names, plumed.biases[0].label)
    return plumed


def _read_bias(action: Action, path: Path) -> Bias:
    label = action.label or action.name.lower()
    bias = Bias(action=action.name, label=label, args=action.get_list("ARG"))
    bias.bias_factor = _to_float(action.get("BIASFACTOR"), path, "BIASFACTOR")
    bias.temperature = _to_float(action.get("TEMP"), path, "TEMP")
    bias.state_file = action.get("STATE_WFILE") or action.get("STATE_RFILE")
    if bias.bias_factor is None and action.name == "OPES_METAD_EXPLORE":
        log.warning("%s: OPES_METAD_EXPLORE without BIASFACTOR", path)
    if bias.bias_factor is None and bias.state_file:
        bias.bias_factor = read_state_bias_factor(path.parent / bias.state_file)
    return bias


def _read_walls(action: Action, plumed: PlumedInput, path: Path) -> None:
    args, at = action.get_list("ARG"), action.get_list("AT")
    if len(args) != len(at):
        raise EstError(f"{path}: {action.name} has {len(args)} ARG but {len(at)} AT values")
    side = 0 if action.name == "LOWER_WALLS" else 1
    for arg, value in zip(args, at, strict=True):
        limits = list(plumed.walls.get(arg, (None, None)))
        limits[side] = _to_float(value, path, "AT")
        plumed.walls[arg] = (limits[0], limits[1])
    if action.label:
        plumed.wall_bias_columns.append(f"{action.label}.bias")


def read_state_bias_factor(path: Path) -> float | None:
    """Read ``#! SET biasfactor`` from an OPES/METAD state file; None if missing."""
    if not path.is_file():
        log.debug("state file %s not found", path)
        return None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            words = line.split()
            if len(words) >= 4 and words[1] == "SET" and words[2].lower() == "biasfactor":
                return _to_float(words[3], path, "biasfactor")
    return None


def parse_actions(path: Path, _seen: set[Path] | None = None) -> list[Action]:
    """Split a PLUMED input into actions."""
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"PLUMED input not found: {path}")
    seen = _seen if _seen is not None else set()
    if path.resolve() in seen:
        raise EstError(f"{path}: INCLUDE loop")
    seen.add(path.resolve())

    actions: list[Action] = []
    for words in _statements(path):
        action = _make_action(words)
        if action.name == "ENDPLUMED":
            break
        if action.name == "INCLUDE":
            target = action.get("FILE")
            if target is None:
                raise EstError(f"{path}: INCLUDE without FILE")
            actions += parse_actions(path.parent / target, seen)
        else:
            actions.append(action)
    return actions


def _statements(path: Path) -> list[list[str]]:
    """Return the words of each statement, joining multi-line ``...`` blocks."""
    statements: list[list[str]] = []
    block: list[str] | None = None
    with open(path, errors="replace") as f:
        for raw in f:
            words = _split(raw.split("#", 1)[0])
            if not words:
                continue
            if block is not None:
                if words[0] == "...":  # end of block, optionally followed by the action name
                    statements.append(block)
                    block = None
                else:
                    block += words
            elif words[-1] == "...":
                block = words[:-1]
            else:
                statements.append(words)
    if block is not None:
        raise EstError(f"{path}: unterminated '...' block")
    return statements


def _split(text: str) -> list[str]:
    """Split on whitespace, keeping ``{...}`` groups together."""
    words, current, depth = [], "", 0
    for char in text:
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        if char.isspace() and depth == 0:
            if current:
                words.append(current)
            current = ""
        else:
            current += char
    if current:
        words.append(current)
    return words


def _make_action(words: list[str]) -> Action:
    label = None
    if words[0].endswith(":"):
        label, words = words[0][:-1], words[1:]
    name = words[0].upper() if words else ""
    keywords: dict[str, str] = {}
    flags: list[str] = []
    for word in words[1:]:
        if "=" in word:
            key, value = word.split("=", 1)
            keywords[key.upper()] = value.strip("{}")
        else:
            flags.append(word.upper())
    if label is None:
        label = keywords.pop("LABEL", None)
    return Action(name=name, label=label, keywords=keywords, flags=flags)


def _to_float(value: str | None, path: Path, what: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)  # also reads "inf"
    except ValueError:
        raise EstError(f"{path}: can't read {what}={value!r} as a number") from None
