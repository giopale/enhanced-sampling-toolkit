"""A project: a set of simulation runs plus the settings in ``estk.yaml``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from estk import units
from estk.config import ProjectConfig, SimulationConfig, load_config
from estk.errors import EstError, MissingFileError
from estk.io import lammps, plumed
from estk.io.colvar import CVTable, load_cv_table
from estk.logging import get_logger

log = get_logger(__name__)


@dataclass
class RunMetadata:
    """Simulation settings of one run.

    ``sources`` says where each value came from (a file name or ``estk.yaml``).
    """

    temperature: float | None = None  # K
    timestep: float | None = None  # ps
    lammps_units: str | None = None
    bias_label: str | None = None
    bias_column: str | None = None
    bias_factor: float | None = None
    time_units: str = "ps"  # PLUMED units of COLVAR time
    energy_units: str = "kj/mol"  # PLUMED units of the bias
    walls: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    wall_bias_columns: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def time_factor(self) -> float:
        """ps per PLUMED time unit."""
        return units.time_factor(self.time_units)

    @property
    def kbt(self) -> float:
        """k_B T in the energy units of the bias."""
        if self.temperature is None:
            raise EstError("temperature unknown; set simulation.temperature in estk.yaml")
        return units.kbt(self.temperature, self.energy_units)


def read_run_metadata(
    lammps_log: Path, plumed_input: Path, overrides: SimulationConfig | None = None
) -> RunMetadata:
    """Collect the settings of a run from its LAMMPS log and PLUMED input.

    Values in ``overrides`` (the ``simulation`` section of estk.yaml) take precedence. Missing
    files only produce a warning: the values they would give stay None.
    """
    meta = RunMetadata()

    if lammps_log.is_file():
        info = lammps.read_log_metadata(lammps_log)
        meta.lammps_units = info.units
        _set(meta, "temperature", info.temperature, lammps_log.name)
        if info.timestep is not None:
            try:
                dt = info.timestep * units.lammps_time_factor(info.units or "lj")
                _set(meta, "timestep", dt, lammps_log.name)
            except EstError as exc:
                log.warning("%s: %s", lammps_log, exc)
    else:
        log.warning("%s not found", lammps_log)

    if plumed_input.is_file():
        info_p = plumed.read_input(plumed_input)
        meta.time_units = info_p.time_units
        meta.energy_units = info_p.energy_units
        meta.walls = info_p.walls
        meta.wall_bias_columns = info_p.wall_bias_columns
        bias = info_p.bias
        if bias is not None:
            meta.bias_label = bias.label
            _set(meta, "bias_column", bias.bias_column, plumed_input.name)
            _set(meta, "bias_factor", bias.bias_factor, plumed_input.name)
            if meta.temperature is None:
                _set(meta, "temperature", bias.temperature, plumed_input.name)
    else:
        log.warning("%s not found", plumed_input)

    if overrides is not None:
        for name in type(overrides).model_fields:
            value = getattr(overrides, name)
            if value is not None:
                setattr(meta, name, value)
                meta.sources[name] = "estk.yaml"
    return meta


def _set(meta: RunMetadata, name: str, value: object, source: str) -> None:
    if value is not None:
        setattr(meta, name, value)
        meta.sources[name] = source


@dataclass(eq=False)
class Run:
    """One simulation run: a folder with COLVAR, the LAMMPS log and the PLUMED input."""

    path: Path
    config: ProjectConfig

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def colvar_path(self) -> Path:
        return self.path / self.config.files.colvar

    @property
    def lammps_log_path(self) -> Path:
        return self.path / self.config.files.lammps_log

    @property
    def plumed_path(self) -> Path:
        return self.path / self.config.files.plumed_input

    @cached_property
    def metadata(self) -> RunMetadata:
        return read_run_metadata(self.lammps_log_path, self.plumed_path, self.config.simulation)

    def load_cvs(
        self,
        cvs: Sequence[str],
        *,
        time: tuple[float | None, float | None] | None = None,
        extra: Mapping[str, str | Path] | None = None,
        filters: Mapping[str, tuple[float | None, float | None]] | None = None,
        bias: bool = True,
    ) -> CVTable:
        """Load CVs joined on time; see ``estk.io.colvar.load_cv_table``."""
        return load_cv_table(self, cvs, time=time, extra=extra, filters=filters, bias=bias)

    def __repr__(self) -> str:
        return f"Run({self.name!r})"


class Project:
    """The runs of a study and their settings.

    ``config`` is a ``ProjectConfig``, a path to an estk.yaml file, or None to use the nearest
    estk.yaml (or defaults if there is none).
    """

    def __init__(self, config: ProjectConfig | str | Path | None = None):
        self.config = config if isinstance(config, ProjectConfig) else load_config(config)

    @cached_property
    def runs(self) -> list[Run]:
        """Run folders matching ``runs.pattern`` in ``runs.root`` that contain a COLVAR file."""
        root, pattern = self.config.runs_root, self.config.runs.pattern
        if not root.is_dir():
            raise MissingFileError(f"runs root folder not found: {root}")
        folders = sorted((p for p in root.glob(pattern) if p.is_dir()), key=_natural_key)
        runs, skipped = [], []
        for folder in folders:
            run = Run(folder, self.config)
            (runs if run.colvar_path.is_file() else skipped).append(run)
        if skipped:
            log.info(
                "skipping folders without %s: %s",
                self.config.files.colvar,
                ", ".join(r.name for r in skipped),
            )
        if not runs:
            raise EstError(
                f"no run folders matching '{pattern}' with a {self.config.files.colvar} file "
                f"in {root}"
            )
        return runs

    def run(self, name: str) -> Run:
        for run in self.runs:
            if run.name == name:
                return run
        raise EstError(f"no run named '{name}' (runs: {', '.join(r.name for r in self.runs)})")

    def load_cvs(
        self,
        cvs: Sequence[str],
        *,
        time: tuple[float | None, float | None] | None = None,
        extra: Mapping[str, str | Path] | None = None,
        filters: Mapping[str, tuple[float | None, float | None]] | None = None,
        bias: bool = True,
    ) -> list[CVTable]:
        """Load the CVs of every run; see ``estk.io.colvar.load_cv_table``."""
        return [
            run.load_cvs(cvs, time=time, extra=extra, filters=filters, bias=bias)
            for run in self.runs
        ]

    def __repr__(self) -> str:
        where = self.config.path or "defaults"
        return f"Project({where})"


def _natural_key(path: Path) -> list[object]:
    """Sort key so that run-2 comes before run-10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]
