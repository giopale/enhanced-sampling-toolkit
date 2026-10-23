"""Project configuration (``estk.toml``).

The file is optional. It only holds facts about the system being studied: where the runs are,
how their files are named, the collective variables (CVs) and atom groups. Choices for a single
analysis are function arguments / CLI flags, and plot cosmetics are plot styles.

Example::

    [runs]
    root = "."
    pattern = "00*"

    [cv.d0]
    label = "d(Ti–O) (Å)"
    limits = [1.5, 2.5]
    bandwidth = 0.0216

    [cv.cn]
    min = 0.75          # one-sided: the upper limit comes from the PLUMED walls or the data

    [atoms]
    type_map = { 1 = "H", 2 = "O", 3 = "Ti" }
    groups = { site_A = [311], site_B = [313, 218] }
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from estk.errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

CONFIG_FILENAME = "estk.toml"


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


class CVConfig(_Section):
    """Settings for one collective variable.

    ``limits = [lo, hi]`` is a shorthand for ``min = lo`` and ``max = hi``. Limits left unset are
    taken from the PLUMED walls, then from the data range.
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


class PlotConfig(_Section):
    """Default plot settings."""

    style: str = "paper"


class ProjectConfig(_Section):
    """Contents of ``estk.toml``."""

    runs: RunsConfig = Field(default_factory=RunsConfig)
    files: FilesConfig = Field(default_factory=FilesConfig)
    cv: dict[str, CVConfig] = Field(default_factory=dict)
    atoms: AtomsConfig = Field(default_factory=AtomsConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)

    _path: Path | None = PrivateAttr(default=None)

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

        ``source`` is ``"estk.toml"`` for values set in the file and ``"default"`` otherwise.
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
    """Return the nearest ``estk.toml`` in ``start`` (default: cwd) or its parents."""
    start = (start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        candidate = folder / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(path: Path | str | None = None) -> ProjectConfig:
    """Load and validate a project configuration.

    With ``path=None`` the nearest ``estk.toml`` is used (see ``find_config``); if there is none,
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
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML: {exc}", path=path) from None

    try:
        config = ProjectConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("invalid configuration", _format_errors(exc), path=path) from None
    config._path = path.resolve()
    return config


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
    """Return a commented ``estk.toml`` with a stub section for each CV in ``cvs``."""
    lines = [
        "# estk project file.",
        "# Everything is optional: settings left out use defaults or are read from the",
        "# simulation files (temperature, timestep, bias factor, PLUMED walls).",
        "",
        "[runs]",
        "# Folder containing the run folders, relative to this file.",
        f"root = {_toml_string(root)}",
        "# Glob pattern matching one folder per simulation run.",
        f"pattern = {_toml_string(pattern)}",
        "",
        "[files]",
        "# File names inside each run folder.",
        'lammps_log = "log.lammps"',
        'plumed_input = "plumed.plumed"',
        'colvar = "COLVAR"',
        "",
        "# One section per collective variable. All keys are optional:",
        "#   label     axis label used in plots",
        "#   units     units shown next to the label",
        "#   limits    grid range [min, max]; or set only `min` or `max`. Unset limits come",
        "#             from the PLUMED walls, then from the data range",
        "#   bandwidth KDE bandwidth; default: Silverman's rule",
        "#   scale     factor applied when loading (e.g. 57.29578 for radians -> degrees)",
        "#   source    file holding this CV when it is not in COLVAR (default: CV-<name>)",
    ]
    for name in cvs or ["d0"]:
        prefix = "" if cvs else "# "
        lines += [
            "",
            f"{prefix}[cv.{_toml_key(name)}]",
            f'# label = "{name}"',
            "# limits = [0.0, 1.0]",
            "# bandwidth = 0.05",
        ]
    lines += [
        "",
        "[atoms]",
        "# Atom type numbers to element symbols, and named groups of 1-based atom indices.",
        '# type_map = { 1 = "H", 2 = "O", 3 = "Ti" }',
        "# groups = { site_A = [311], site_B = [313, 218] }",
        "",
        "[plot]",
        'style = "paper"',
        "",
    ]
    return "\n".join(lines)


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_key(key: str) -> str:
    bare = key.replace("_", "").replace("-", "").isalnum() and key.isascii()
    return key if bare else _toml_string(key)


def read_colvar_fields(path: Path) -> list[str]:
    """Return the column names from the ``#! FIELDS`` header line of a PLUMED COLVAR file."""
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                return line.split()[2:]
            if not line.startswith("#"):
                break
    return []
