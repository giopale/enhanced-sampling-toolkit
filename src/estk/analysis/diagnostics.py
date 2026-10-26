"""Checks of the simulations: OPES quantities and the model deviation of ML potentials."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from estk.errors import EstError
from estk.io.colvar import _file_time_ps, _keys, read_table
from estk.logging import get_logger

if TYPE_CHECKING:
    from estk.project import Run

log = get_logger(__name__)

# OPES output components, in the order they are shown.
OPES_COMPONENTS = ("rct", "zed", "neff", "nker")


@dataclass
class TimeSeries:
    """Columns of one file of one run against time (ns), restarts removed."""

    run: str
    time: np.ndarray  # ns
    values: dict[str, np.ndarray]
    source: Path

    def __len__(self) -> int:
        return len(self.time)


def read_timeseries(
    run: Run, columns: Sequence[str] | None = None, file: str | None = None
) -> TimeSeries:
    """Read ``columns`` (default: all but the first) of ``file`` (default: COLVAR) in ``run``.

    Files indexed by ``step`` (e.g. model_devi.out) are converted to time with the run's
    timestep. Repeated times (restarts) keep the last occurrence.
    """
    path = run.colvar_path if file is None else run.path / file
    table = read_table(path)
    names = list(columns) if columns is not None else table.fields[1:]
    missing = [c for c in names if not table.has(c)]
    if missing:
        raise EstError(
            f"{path}: no column(s) {', '.join(missing)} (columns: {', '.join(table.fields)})"
        )
    time_ps = _file_time_ps(table, run)
    _, rows, duplicates = _keys(time_ps)
    if duplicates:
        log.debug("%s: %d repeated times removed", path, duplicates)
    return TimeSeries(
        run=run.name,
        time=time_ps[rows] / 1000.0,
        values={c: table.column(c)[rows] for c in names},
        source=path,
    )


def opes_columns(run: Run) -> list[str]:
    """The OPES output columns of the run's bias that COLVAR contains (e.g. opes.rct)."""
    label = run.metadata.bias_label
    if label is None:
        raise EstError(f"{run.name}: no bias found in {run.plumed_path.name}")
    from estk.io.colvar import read_header

    fields = read_header(run.colvar_path)
    found = [f"{label}.{c}" for c in OPES_COMPONENTS if f"{label}.{c}" in fields]
    if not found:
        raise EstError(
            f"{run.colvar_path}: no {label}.rct/zed/neff/nker columns; add {label}.* to PRINT ARG"
        )
    return found


@dataclass
class OpesSummary:
    run: str
    time: float  # ns, end of the run
    final: dict[str, float]
    rct_change: float | None  # change of rct over the last quarter of the run (energy units)


def opes_summary(series: TimeSeries) -> OpesSummary:
    """Final values, and how much ``rct`` (c(t)) still changed over the last quarter of the run.

    A converged OPES run has an almost constant c(t).
    """
    final = {name: float(values[-1]) for name, values in series.values.items()}
    rct = next((v for k, v in series.values.items() if k.endswith(".rct")), None)
    change = None
    if rct is not None and len(rct) > 4:
        t = series.time
        quarter = t >= t[0] + 0.75 * (t[-1] - t[0])
        change = float(rct[-1] - rct[quarter][0])
    return OpesSummary(series.run, float(series.time[-1]), final, change)


@dataclass
class DeviationStats:
    """How a model deviation is distributed over the trust levels ``low`` < ``high``."""

    run: str
    frames: int
    low: float
    high: float
    below: float  # fraction of frames < low (accurate)
    between: float  # low <= value < high (candidates for retraining)
    above: float  # >= high (unreliable)
    mean: float
    maximum: float
    extra: dict[str, float] = field(default_factory=dict)


def deviation_stats(series: TimeSeries, column: str, low: float, high: float) -> DeviationStats:
    if not low < high:
        raise EstError(f"the trust levels must be increasing, got {low} and {high}")
    values = series.values[column]
    n = len(values)
    if n == 0:
        raise EstError(f"{series.source}: no data")
    return DeviationStats(
        run=series.run,
        frames=n,
        low=low,
        high=high,
        below=float(np.count_nonzero(values < low) / n),
        between=float(np.count_nonzero((values >= low) & (values < high)) / n),
        above=float(np.count_nonzero(values >= high) / n),
        mean=float(np.mean(values)),
        maximum=float(np.max(values)),
    )
