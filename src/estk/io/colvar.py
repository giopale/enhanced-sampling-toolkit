"""Reading COLVAR-like files and assembling time-aligned CV tables.

A run's data can be spread over several files: COLVAR (written during the simulation),
``CV-<name>`` files (CVs computed afterwards with the PLUMED driver) and property files such as
``model_devi.out`` (indexed by MD step). ``load_cv_table`` reads the columns it needs from each
file and joins them **on time**, so a missing or repeated row in one file only drops that row.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from estk.errors import EstError, MissingFileError
from estk.logging import get_logger

if TYPE_CHECKING:
    from estk.config import ProjectConfig
    from estk.project import Run

log = get_logger(__name__)

# Times closer than this (ps) are treated as the same frame when joining files.
TIME_TOLERANCE_PS = 1e-6


@dataclass
class Table:
    """The numeric content of a COLVAR-like file."""

    path: Path
    fields: list[str]
    data: np.ndarray  # shape (rows, len(fields))
    bad_lines: int = 0

    def has(self, name: str) -> bool:
        return name in self.fields

    def column(self, name: str) -> np.ndarray:
        try:
            return self.data[:, self.fields.index(name)]
        except ValueError:
            raise EstError(
                f"{self.path}: no column '{name}' (columns: {', '.join(self.fields)})"
            ) from None


def read_header(path: Path) -> list[str]:
    """Column names from the first line: ``#! FIELDS a b c`` (PLUMED) or ``# a b c``."""
    with open(path, errors="replace") as f:
        first = f.readline().strip()
    if first.startswith("#! FIELDS"):
        return first.split()[2:]
    if first.startswith("#"):
        return first[1:].split()
    raise EstError(f"{path}: first line must be a header ('#! FIELDS ...' or '# ...')")


def read_table(path: Path) -> Table:
    """Read a whitespace-separated table with a header line; comment lines are skipped.

    Lines with the wrong number of columns (e.g. a last line cut off when the simulation was
    killed) are dropped with a warning.
    """
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"file not found: {path}")
    fields = read_header(path)
    bad = 0
    try:
        data = np.loadtxt(path, comments="#", ndmin=2, dtype=float)
    except ValueError:
        data, bad = _read_tolerant(path, len(fields))
    if data.size == 0:
        data = np.empty((0, len(fields)))

    if data.shape[1] < len(fields):
        fields = fields[: data.shape[1]]
    elif data.shape[1] > len(fields):
        fields = fields + [f"col{j}" for j in range(len(fields), data.shape[1])]
    if bad:
        log.warning("%s: skipped %d malformed line(s)", path, bad)
    return Table(path=path, fields=fields, data=data, bad_lines=bad)


def _read_tolerant(path: Path, ncols: int) -> tuple[np.ndarray, int]:
    rows, bad = [], 0
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            words = line.split()
            try:
                values = [float(w) for w in words]
            except ValueError:
                bad += 1
                continue
            if len(values) != ncols:
                bad += 1
                continue
            rows.append(values)
    return np.array(rows, dtype=float).reshape(-1, ncols), bad


@dataclass
class LoadReport:
    """What happened while loading a run's CV table."""

    sources: dict[str, Path] = field(default_factory=dict)  # column -> file it came from
    rows_colvar: int = 0
    duplicates: int = 0  # repeated times (restarts); the last occurrence is kept
    missing: dict[str, int] = field(default_factory=dict)  # file -> COLVAR rows it lacks
    outside_time: int = 0
    nan: dict[str, int] = field(default_factory=dict)  # column -> rows dropped for NaN
    filtered: int = 0
    rows: int = 0
    # "<cv> upper/lower" -> (frames past the wall, largest wall energy in k_BT or None)
    walls: dict[str, tuple[int, float | None]] = field(default_factory=dict)
    walls_in_bias: bool = False

    def wall_summary(self) -> str:
        """Frames past each wall, e.g. ``d0 upper: 12 frames, max 3.1 kT``."""
        parts = []
        for name, (n, max_kbt) in self.walls.items():
            energy = f", max {max_kbt:.3g} kT" if max_kbt is not None and n else ""
            parts.append(f"{name}: {n} frames{energy}")
        return "; ".join(parts)

    def summary(self) -> str:
        parts = [f"{self.rows}/{self.rows_colvar} rows kept"]
        if self.duplicates:
            parts.append(f"{self.duplicates} duplicated times")
        for name, n in self.missing.items():
            parts.append(f"{n} missing in {name}")
        if self.outside_time:
            parts.append(f"{self.outside_time} outside the time window")
        for name, n in self.nan.items():
            parts.append(f"{n} NaN in {name}")
        if self.filtered:
            parts.append(f"{self.filtered} filtered out")
        return ", ".join(parts)


@dataclass
class CVTable:
    """Time-aligned data of one run: time (ps), the bias, and one array per CV/quantity."""

    run_name: str
    time: np.ndarray
    columns: dict[str, np.ndarray]
    bias: np.ndarray | None = None
    kbt: float | None = None  # in the energy units of the bias
    energy_units: str = "kj/mol"
    report: LoadReport = field(default_factory=LoadReport)

    def __len__(self) -> int:
        return len(self.time)

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(f"'{name}' not loaded (loaded: {', '.join(self.columns)})") from None

    @property
    def names(self) -> list[str]:
        return list(self.columns)

    def array(self, names: Sequence[str] | None = None) -> np.ndarray:
        """Stack the given columns (default: all) into an array of shape (rows, len(names))."""
        names = list(names) if names is not None else self.names
        return np.column_stack([self[n] for n in names]) if names else np.empty((len(self), 0))


def load_cv_table(
    run: Run,
    cvs: Sequence[str],
    *,
    time: tuple[float | None, float | None] | None = None,
    extra: Mapping[str, str | Path] | None = None,
    filters: Mapping[str, tuple[float | None, float | None]] | None = None,
    bias: bool = True,
    include_walls: bool = False,
) -> CVTable:
    """Load the CVs of one run, joined on time.

    Parameters
    ----------
    run:
        The run to read.
    cvs:
        CV names. Each is read from COLVAR if it has that column, otherwise from the file set
        by ``cv.<name>.source`` in estk.yaml, otherwise from ``CV-<name>`` (for ``x.y``,
        ``CV-x``).
    time:
        ``(start, end)`` in ns; either end can be None.
    extra:
        Other quantities to load, as ``{column: file}`` (e.g. ``{"max_devi_f": "model_devi.out"}``).
        Files whose first column is ``step`` are converted to time with the run's timestep.
    filters:
        ``{column: (min, max)}``: rows outside these ranges are dropped. Columns that are not
        in ``cvs`` or ``extra`` are loaded for filtering only.
    bias:
        Also load the bias column (needed for reweighting).
    include_walls:
        Add the energy of the PLUMED walls to the bias. It is computed from the wall
        definitions (AT, KAPPA, EXP, EPS, OFFSET) and the values of the walled CVs, which are
        loaded for this purpose if needed.

    The frames past each wall of a loaded CV are counted in ``report.walls`` in any case.
    """
    meta = run.metadata
    config = run.config
    report = LoadReport()
    time_factor = meta.time_factor

    colvar = read_table(run.colvar_path)
    if not colvar.fields or colvar.fields[0].lower() != "time":
        raise EstError(f"{colvar.path}: first column must be 'time'")
    report.rows_colvar = len(colvar.data)
    keys, order, report.duplicates = _keys(colvar.column("time") * time_factor)
    time_ps = colvar.column("time")[order] * time_factor
    columns: dict[str, np.ndarray] = {}

    def take(table: Table, name: str, column: str, idx: np.ndarray) -> None:
        columns[name] = table.column(column)[idx]
        report.sources[name] = table.path

    # Which file each requested column comes from.
    wanted: dict[str, Path | None] = {}
    for name in cvs:
        wanted[name] = None if colvar.has(name) else run.path / _cv_source(name, config)
    for name, path in (extra or {}).items():
        wanted[name] = run.path / Path(path)
    hidden = set()  # columns loaded only for filters or walls
    for name in filters or {}:
        if name not in wanted:
            wanted[name] = None if colvar.has(name) else run.path / _cv_source(name, config)
            hidden.add(name)
    walls = meta.wall_terms
    if include_walls and bias:
        for wall in walls:
            if wall.cv not in wanted:
                wanted[wall.cv] = (
                    None if colvar.has(wall.cv) else run.path / _cv_source(wall.cv, config)
                )
                hidden.add(wall.cv)

    bias_column = meta.bias_column
    if bias:
        if bias_column is None:
            raise EstError(
                f"{run.name}: no bias found in {run.plumed_path.name}; "
                "set simulation.bias_column in estk.yaml"
            )
        if not colvar.has(bias_column):
            raise EstError(
                f"{colvar.path}: no bias column '{bias_column}' "
                f"(columns: {', '.join(colvar.fields)})"
            )

    # Read the other files and join them on time.
    other_tables: dict[Path, Table] = {}
    for path in {p for p in wanted.values() if p is not None}:
        try:
            other_tables[path] = read_table(path)
        except MissingFileError:
            needed_by = [n for n, p in wanted.items() if p == path]
            if any(n in hidden and n not in (filters or {}) for n in needed_by):
                raise MissingFileError(
                    f"{path} not found: needed for the energy of the wall on {', '.join(needed_by)}"
                ) from None
            raise

    selection = np.arange(len(keys))  # rows of the (deduplicated) COLVAR still kept
    joined: dict[Path, np.ndarray] = {}  # file -> its row for each kept COLVAR row
    for path, table in other_tables.items():
        t = _file_time_ps(table, run)
        tkeys, torder, _ = _keys(t)
        common, i_colvar, i_table = np.intersect1d(
            keys[selection], tkeys, assume_unique=True, return_indices=True
        )
        lacking = len(selection) - len(common)
        if lacking:
            report.missing[path.name] = lacking
        for done in joined:
            joined[done] = joined[done][i_colvar]
        selection = selection[i_colvar]
        joined[path] = torder[i_table]
        if len(selection) == 0:
            raise EstError(f"{path}: no times in common with {colvar.path.name}")

    rows = order[selection]
    for name, path in wanted.items():
        if path is None:
            take(colvar, name, name, rows)
        else:
            table = other_tables[path]
            take(table, name, _column_in(table, name), joined[path])
    time_ps = time_ps[selection]
    bias_values = colvar.column(bias_column)[rows] if bias and bias_column else None

    # Walls act on the unscaled CVs, so they are evaluated before scaling.
    wall_past: dict[str, np.ndarray] = {}
    wall_energy: dict[str, np.ndarray | None] = {}
    for wall in walls:
        if wall.cv in columns:
            name = f"{wall.cv} {wall.side}"
            wall_past[name] = wall.past(columns[wall.cv])
            wall_energy[name] = wall.energy(columns[wall.cv]) if wall.kappa is not None else None
    if include_walls and bias_values is not None and walls:
        missing_kappa = [n for n, e in wall_energy.items() if e is None]
        if missing_kappa:
            raise EstError(f"{run.name}: no KAPPA for the wall(s) {', '.join(missing_kappa)}")
        bias_values = bias_values + sum(e for e in wall_energy.values() if e is not None)
        report.walls_in_bias = True

    # Scale CVs as set in estk.yaml (e.g. radians -> degrees).
    for name in columns:
        scale = config.cv[name].scale if name in config.cv else 1.0
        if scale != 1.0:
            columns[name] = columns[name] * scale

    keep = np.ones(len(time_ps), dtype=bool)
    if time is not None:
        start, end = time
        in_window = np.ones_like(keep)
        if start is not None:
            in_window &= time_ps >= start * 1000.0
        if end is not None:
            in_window &= time_ps <= end * 1000.0
        report.outside_time = int(np.count_nonzero(keep & ~in_window))
        keep &= in_window
    for name, values in columns.items():
        bad = keep & np.isnan(values)
        if bad.any():
            report.nan[name] = int(np.count_nonzero(bad))
            keep &= ~bad
    for name, (lo, hi) in (filters or {}).items():
        inside = np.ones_like(keep)
        if lo is not None:
            inside &= columns[name] >= lo
        if hi is not None:
            inside &= columns[name] <= hi
        report.filtered += int(np.count_nonzero(keep & ~inside))
        keep &= inside

    kbt = meta.kbt if meta.temperature is not None else None
    for name, past in wall_past.items():
        n_past = int(np.count_nonzero(past & keep))
        energy = wall_energy[name]
        max_kbt = None
        if energy is not None and kbt is not None and n_past:
            max_kbt = float(np.max(energy[keep])) / kbt
        report.walls[name] = (n_past, max_kbt)

    for name in hidden:
        if name not in cvs and name not in (extra or {}):
            columns.pop(name)
    report.rows = int(np.count_nonzero(keep))
    if report.rows == 0:
        raise EstError(f"{run.name}: no rows left after selection ({report.summary()})")

    table = CVTable(
        run_name=run.name,
        time=time_ps[keep],
        columns={name: values[keep] for name, values in columns.items()},
        bias=bias_values[keep] if bias_values is not None else None,
        kbt=kbt,
        energy_units=meta.energy_units,
        report=report,
    )
    log.debug("%s: %s", run.name, report.summary())
    return table


def _keys(time_ps: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer time keys, deduplicated keeping the last occurrence, sorted.

    Returns ``(keys, rows, n_duplicates)`` where ``rows`` are the original row indices.
    """
    raw = np.rint(time_ps / TIME_TOLERANCE_PS).astype(np.int64)
    reversed_keys = raw[::-1]
    keys, first_in_reversed = np.unique(reversed_keys, return_index=True)
    rows = len(raw) - 1 - first_in_reversed
    return keys, rows, len(raw) - len(keys)


def _file_time_ps(table: Table, run: Run) -> np.ndarray:
    first = table.fields[0].lower() if table.fields else ""
    if first == "step":
        dt = run.metadata.timestep
        if dt is None:
            raise EstError(
                f"{table.path}: indexed by step but the timestep is unknown; "
                "set simulation.timestep in estk.yaml"
            )
        return table.data[:, 0] * dt
    if first != "time":
        log.warning("%s: first column '%s' is used as time", table.path, table.fields[0])
    return table.data[:, 0] * run.metadata.time_factor


def _cv_source(name: str, config: ProjectConfig) -> str:
    cv = config.cv.get(name)
    if cv is not None and cv.source:
        return cv.source
    return f"CV-{name.split('.')[0]}"


def _column_in(table: Table, name: str) -> str:
    """The column of ``table`` holding ``name``: same name, or the only data column."""
    if table.has(name):
        return name
    data_columns = table.fields[1:]
    if len(data_columns) == 1:
        return data_columns[0]
    raise EstError(
        f"{table.path}: no column '{name}' and more than one data column "
        f"({', '.join(data_columns)}); name the column '{name}'"
    )
