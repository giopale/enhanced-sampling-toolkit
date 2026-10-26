"""Project configuration (``estk.yaml``).

The file is optional. It only holds facts about the system being studied: where the runs are,
how their files are named, the collective variables (CVs) and atom groups. Choices for a single
analysis are function arguments / CLI flags, and plot cosmetics are plot styles.

Example::

    runs:
      root: .
      pattern: "00*"

    cv:
      d0:
        label: "d(Ti–O) (Å)"
        limits: [1.5, 2.5]
        bandwidth: 0.0216
      cn:
        limits: [0.75, null]   # null: taken from the PLUMED walls, then the data range

    reaction:
      start: {cn: 2.0}         # paths start at the minimum nearest this point

    atoms:
      type_map: {1: H, 2: O, 3: Ti}
      groups:
        water_oxygens: &water [153, 154, 155]
        site_A: [311]
        site_liquid: *water
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from estk import _yaml
from estk.errors import ConfigError

CONFIG_FILENAME = "estk.yaml"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunsConfig(_Section):
    """Where the simulation runs are: one folder per run, matched by ``pattern`` in ``root``."""

    root: Path = Path(".")
    pattern: str = "*"


class FilesConfig(_Section):
    """Names of the files inside each run folder."""

    lammps_log: str = "log.lammps"
    plumed_input: str = "plumed.plumed"
    colvar: str = "COLVAR"
    model_deviation: str = "model_devi.out"


class SimulationConfig(_Section):
    """Overrides for settings normally read from the simulation files.

    Leave these out unless a value can't be read (or is read wrongly) from ``log.lammps``, the
    PLUMED input or the bias state file.
    """

    temperature: float | None = Field(default=None, gt=0)  # K
    timestep: float | None = Field(default=None, gt=0)  # ps
    bias_factor: float | None = Field(default=None, gt=1)
    bias_column: str | None = None  # COLVAR column with the bias, e.g. "opes.bias"
    time_units: str | None = None  # PLUMED time units of COLVAR, e.g. "ps"
    energy_units: str | None = None  # PLUMED energy units of the bias, e.g. "kj/mol"


class CVConfig(_Section):
    """Settings for one collective variable.

    ``limits: [lo, hi]`` is a shorthand for ``min: lo`` and ``max: hi``; either end can be null.
    Limits left unset are taken from the PLUMED walls, then from the data range.
    """

    label: str | None = None
    units: str | None = None
    min: float | None = None
    max: float | None = None
    bandwidth: float | None = Field(default=None, gt=0)
    scale: float = 1.0
    source: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_limits(cls, data: Any) -> Any:
        if data is None:  # a CV listed with no settings
            return {}
        if not isinstance(data, dict) or "limits" not in data:
            return data
        data = dict(data)
        limits = data.pop("limits")
        if "min" in data or "max" in data:
            raise ValueError("use either 'limits' or 'min'/'max', not both")
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"'limits' must be a list of two numbers, got {limits!r}")
        data["min"], data["max"] = limits
        return data

    @model_validator(mode="after")
    def _check_order(self) -> CVConfig:
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(f"min ({self.min}) must be smaller than max ({self.max})")
        return self

    @property
    def limits(self) -> tuple[float | None, float | None]:
        return (self.min, self.max)


class AtomsConfig(_Section):
    """Atom types and named groups of atoms.

    Atom indices are 1-based, as in LAMMPS and PLUMED.
    """

    type_map: dict[int, str] = Field(default_factory=dict)
    groups: dict[str, list[int]] = Field(default_factory=dict)

    @field_validator("groups", mode="before")
    @classmethod
    def _single_index_to_list(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: [v] if isinstance(v, int) else v for k, v in value.items()}
        return value

    @field_validator("groups")
    @classmethod
    def _check_indices(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        for name, indices in value.items():
            bad = [i for i in indices if i < 1]
            if bad:
                raise ValueError(f"group '{name}': atom indices are 1-based, got {bad}")
        return value


class ReactionConfig(_Section):
    """Which way the reaction goes: the path starts at the minimum nearest ``start``.

    ``start`` gives values of some CVs, e.g. ``{cn: 2.0}``; only the CVs of the surface count.
    The barrier is measured from that end.
    """

    start: dict[str, float] = Field(default_factory=dict)


class PlotConfig(_Section):
    """Default plot settings."""

    style: str = "paper"


class ProjectConfig(_Section):
    """Contents of ``estk.yaml``."""

    runs: RunsConfig = Field(default_factory=RunsConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    cv: dict[str, CVConfig] = Field(default_factory=dict)
    atoms: AtomsConfig = Field(default_factory=AtomsConfig)
    reaction: ReactionConfig = Field(default_factory=ReactionConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)

    _path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _empty_sections(cls, data: Any) -> Any:
        # A section whose keys are all commented out reads as null: treat it as "use defaults".
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    @property
    def path(self) -> Path | None:
        """The file this configuration was loaded from, or None when using defaults."""
        return self._path

    @property
    def base_dir(self) -> Path:
        """Folder that relative paths are resolved against."""
        return self._path.parent if self._path is not None else Path.cwd()

    @property
    def runs_root(self) -> Path:
        """``runs.root`` as an absolute path."""
        root = self.runs.root.expanduser()
        return root if root.is_absolute() else (self.base_dir / root).resolve()

    def explain(self) -> list[tuple[str, Any, str]]:
        """List ``(key, value, source)`` for every setting in effect.

        ``source`` is ``"estk.yaml"`` for values set in the file and ``"default"`` otherwise.
        Entries the user defines (CVs, atom groups) only show the fields that were set.
        """
        rows: list[tuple[str, Any, str]] = []
        _explain_model(self, "", rows, only_set=False)
        return rows


def _explain_model(
    model: BaseModel, prefix: str, rows: list[tuple[str, Any, str]], only_set: bool
) -> None:
    for name in type(model).model_fields:
        value = getattr(model, name)
        key = f"{prefix}{name}"
        is_set = name in model.model_fields_set
        if isinstance(value, BaseModel):
            _explain_model(value, key + ".", rows, only_set=only_set)
        elif (
            isinstance(value, dict)
            and value
            and all(isinstance(v, BaseModel) for v in value.values())
        ):
            for entry, sub in value.items():
                if sub.model_fields_set:
                    _explain_model(sub, f"{key}.{entry}.", rows, only_set=True)
                else:
                    rows.append((f"{key}.{entry}", "(all defaults)", CONFIG_FILENAME))
        elif is_set or not only_set:
            rows.append((key, value, CONFIG_FILENAME if is_set else "default"))


def find_config(start: Path | None = None) -> Path | None:
    """Return the nearest ``estk.yaml`` in ``start`` (default: cwd) or its parents."""
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        candidate = folder / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """Load and validate a project configuration.

    With ``path=None`` the nearest ``estk.yaml`` is used (see ``find_config``); if there is none,
    all defaults are returned. An explicit ``path`` that does not exist is an error.
    """
    if path is None:
        path = find_config()
        if path is None:
            return ProjectConfig()
    path = Path(path)
    if not path.is_file():
        raise ConfigError("configuration file not found", path=path)

    try:
        data, _ = _yaml.load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}", path=path) from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("expected settings (key: value) at the top level", path=path)

    try:
        config = ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("invalid configuration", _format_errors(exc), path=path) from None
    config._path = path.resolve()
    return config


def set_cv_bandwidths(path: Path | str, values: dict[str, float]) -> None:
    """Write ``cv.<name>.bandwidth`` into an existing estk.yaml, changing only those lines.

    Comments, anchors and the layout of the rest of the file are kept. The result is loaded
    again to check it; on any problem the file is left as it was.
    """
    path = Path(path)
    original = path.read_text()
    lines = original.splitlines()
    for name, value in values.items():
        lines = _set_cv_key(lines, name, "bandwidth", f"{value:.4g}")
    path.write_text("\n".join(lines) + "\n")
    try:
        config = load_config(path)
        for name, value in values.items():
            if config.cv[name].bandwidth != float(f"{value:.4g}"):
                raise ConfigError(f"could not set cv.{name}.bandwidth", path=path)
    except Exception:
        path.write_text(original)
        raise


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_content(line: str) -> bool:
    return bool(line.strip()) and not line.lstrip().startswith("#")


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """Index after the last content line indented deeper than ``indent``, from ``start + 1``."""
    end = start + 1
    for i in range(start + 1, len(lines)):
        if _is_content(lines[i]):
            if _indent(lines[i]) <= indent:
                break
            end = i + 1
    return end


def _set_cv_key(lines: list[str], name: str, key: str, value: str) -> list[str]:
    lines = list(lines)
    header = re.compile(r"cv:\s*(\{\s*\})?\s*(#.*)?$")  # "cv:", "cv: {}", with a comment
    section = next((i for i, line in enumerate(lines) if header.match(line)), None)
    if section is None:
        return [*lines, "", "cv:", f"  {name}:", f"    {key}: {value}"]
    lines[section] = "cv:"
    end = _block_end(lines, section, 0)
    entries = [i for i in range(section + 1, end) if _is_content(lines[i])]
    indent = _indent(lines[entries[0]]) if entries else 2
    for i in entries:
        if _indent(lines[i]) != indent:
            continue
        key_text, _, rest = lines[i].strip().partition(":")
        if key_text.strip("'\"") != name:
            continue
        rest = rest.split(" #")[0].strip()
        if rest:  # flow mapping, e.g. "d0: {limits: [1, 2]}"
            if "&" in rest or "*" in rest or not rest.startswith("{"):
                raise ConfigError(f"cannot edit cv.{name} ({rest}); set {key} by hand")
            mapping = yaml.safe_load(rest) or {}
            mapping[key] = float(value)
            flow = yaml.safe_dump(mapping, default_flow_style=True, sort_keys=False).strip()
            lines[i] = f"{' ' * indent}{key_text}: {flow}"
            return lines
        entry_end = _block_end(lines, i, indent)
        children = [j for j in range(i + 1, entry_end) if _is_content(lines[j])]
        child = _indent(lines[children[0]]) if children else indent + 2
        for j in children:
            if _indent(lines[j]) == child and lines[j].strip().startswith(f"{key}:"):
                lines[j] = f"{' ' * child}{key}: {value}"
                return lines
        lines.insert(entry_end, f"{' ' * child}{key}: {value}")
        return lines
    lines[end:end] = [f"{' ' * indent}{name}:", f"{' ' * (indent + 2)}{key}: {value}"]
    return lines


def _format_errors(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            msg = "unknown setting (check the spelling)"
        problems.append(f"{loc}: {msg}" if loc else msg)
    return problems


def is_bias_column(name: str) -> bool:
    """True for COLVAR columns that are not CVs: time, biases and walls (opes.*, *.bias, ...)."""
    lowered = name.lower()
    return (
        lowered == "time"
        or lowered.endswith(".bias")
        or lowered.startswith(("opes.", "metad.", "uwall", "lwall"))
    )


def config_template(*, root: str = ".", pattern: str = "*", cvs: list[str] | None = None) -> str:
    """Return a commented ``estk.yaml`` with a stub entry for each CV in ``cvs``."""
    lines = [
        "# estk project file.",
        "# Everything is optional: settings left out use defaults or are read from the",
        "# simulation files (temperature, timestep, bias factor, PLUMED walls).",
        "",
        "runs:",
        "  # Folder containing the run folders, relative to this file.",
        f"  root: {_yaml.scalar(root)}",
        "  # Glob pattern matching one folder per simulation run.",
        f"  pattern: {_yaml.scalar(pattern)}",
        "",
        "files:",
        "  # File names inside each run folder.",
        "  lammps_log: log.lammps",
        "  plumed_input: plumed.plumed",
        "  colvar: COLVAR",
        "  model_deviation: model_devi.out",
        "",
        "# Only needed when a value can't be read from the simulation files:",
        "# simulation:",
        "#   temperature: 350     # K",
        "#   timestep: 0.0005     # ps",
        "#   bias_factor: 10",
        "#   bias_column: opes.bias",
        "",
        "# One entry per collective variable. All keys are optional:",
        "#   label      axis label used in plots",
        "#   units      units shown next to the label",
        "#   limits     grid range [min, max]; either end can be null. Unset limits come from",
        "#              the PLUMED walls, then from the data range",
        "#   bandwidth  KDE bandwidth; default: Silverman's rule",
        "#   scale      factor applied when loading (e.g. 57.29578 for radians -> degrees)",
        "#   source     file holding this CV when it is not in COLVAR (default: CV-<name>)",
        "cv:",
    ]
    for name in cvs or ["d0"]:
        prefix = "  " if cvs else "  # "
        lines += [
            f"{prefix}{_yaml.key(name)}:",
            f"  #   label: {_yaml.scalar(name)}",
            "  #   limits: [null, null]",
            "  #   bandwidth: 0.05",
        ]
    lines += [
        "",
        "atoms:",
        "  # Atom type numbers to element symbols, and named groups of 1-based atom indices.",
        "  # type_map: {1: H, 2: O, 3: Ti}",
        "  # groups:",
        "  #   site_A: [311]",
        "  #   site_B: [313, 218]",
        "",
        "# Which way the reaction goes: minimum energy paths start at the minimum nearest this",
        "# point (only the CVs of the surface count) and the barrier is measured from there.",
        "# Default: the minimum with the smaller value of the first CV.",
        "# reaction:",
        "#   start: {cn: 2.0}",
        "",
        "plot:",
        "  style: paper",
        "",
    ]
    return "\n".join(lines)


def read_colvar_fields(path: Path) -> list[str]:
    """Return the column names from the ``#! FIELDS`` header line of a PLUMED COLVAR file."""
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                return line.split()[2:]
            if not line.startswith("#"):
                break
    return []
