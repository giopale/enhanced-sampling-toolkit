"""Equilibrium (reweighted) averages of quantities over whole runs.

``<q> = sum_i w_i q_i / sum_i w_i`` with ``w_i = exp(V_i / kT)`` over every frame of a run (in
the time window), then the runs combined like surfaces: ``equal`` (each run counts the same)
or ``probability`` (by each run's total weight). Frames where a quantity is NaN are left out
of that quantity's average only. The error is the standard error between runs, or between
time blocks when there is one run.

For a site's coordination number (the number of protons on it), the average is the site's
equilibrium occupation, e.g. its protonated fraction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.errors import EstError, MissingFileError
from estk.io.textfile import PREFIX, decode, metadata_lines, read_header
from estk.reweight.core import logsumexp

if TYPE_CHECKING:
    from estk.project import Project


@dataclass
class AverageResult:
    """Combined averages with their errors, and each run's own averages."""

    quantities: list[str]
    values: dict[str, float]
    errors: dict[str, float]  # standard error between runs (or blocks); NaN without either
    runs: dict[str, dict[str, float]]  # run -> quantity -> average
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        """One line per quantity: name, value, error, then one column per run."""
        path = Path(path)
        runs = list(self.runs)
        fields = ["quantity", "value", "error", *[f"run.{r}" for r in runs]]
        lines = [f"#! FIELDS {' '.join(fields)}", *metadata_lines({"kind": "averages",
                                                                   **self.metadata})]  # fmt: skip
        for q in self.quantities:
            values = [self.values[q], self.errors[q], *[self.runs[r][q] for r in runs]]
            lines.append(" ".join([q, *(f"{v:.10g}" for v in values)]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> AverageResult:
        path = Path(path)
        if not path.is_file():
            raise MissingFileError(f"averages file not found: {path}")
        fields, sets = read_header(path)
        meta = {k[len(PREFIX) :]: decode(v) for k, v in sets.items() if k.startswith(PREFIX)}
        if meta.pop("kind", None) != "averages":
            raise EstError(f"{path}: not an averages file (estk average)")
        runs = [f[4:] for f in fields if f.startswith("run.")]
        quantities, values, errors = [], {}, {}
        per_run: dict[str, dict[str, float]] = {r: {} for r in runs}
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            name, *raw = line.split()
            numbers = [float(v) for v in raw]
            quantities.append(name)
            values[name], errors[name] = numbers[0], numbers[1]
            for r, v in zip(runs, numbers[2:], strict=True):
                per_run[r][name] = v
        return cls(quantities, values, errors, per_run, meta)


def _mean(w: np.ndarray, v: np.ndarray) -> float:
    ok = np.isfinite(v)
    total = float(w[ok].sum())
    return float(np.sum(w[ok] * v[ok]) / total) if total > 0 else math.nan


def _standard_error(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def ensemble_averages(
    project: Project,
    quantities: Sequence[str],
    *,
    sources: Mapping[str, str | Path] | None = None,
    time: tuple[float | None, float | None] | None = None,
    blocks: int = 1,
    combine: str = "equal",
) -> AverageResult:
    """Reweighted average of each quantity over all frames of every run, runs combined.

    ``quantities`` are loaded like CVs (COLVAR, ``CV-<name>``, ``cv.<name>.source``) or from
    ``sources`` (``{quantity: file}``). ``blocks``: time blocks per run, for the error bar when
    there is only one run.
    """
    quantities = list(dict.fromkeys(quantities))
    if not quantities:
        raise EstError("no quantities to average")
    if combine not in ("equal", "probability"):
        raise EstError(f"combine must be equal or probability, got '{combine}'")
    if blocks < 1:
        raise EstError(f"blocks must be at least 1, got {blocks}")
    extra = {q: f for q, f in (sources or {}).items() if q in quantities}
    loaded = [q for q in quantities if q not in extra]
    per_run: dict[str, dict[str, float]] = {}
    per_block: list[dict[str, float]] = []
    log_totals, frames = [], 0
    for run in project.runs:
        table = run.load_cvs(loaded, time=time, extra=extra or None, drop_nan=False)
        if table.bias is None or table.kbt is None:
            raise EstError(f"{run.name}: no bias or temperature; cannot weight the frames")
        logw = table.bias / table.kbt
        w = np.exp(logw - logw.max())
        log_totals.append(float(logsumexp(logw)))
        frames += len(table)
        per_run[run.name] = {q: _mean(w, np.asarray(table[q], float)) for q in quantities}
        if blocks > 1 and not per_block:  # the blocks of the first run
            order = np.argsort(table.time, kind="stable")
            for rows in np.array_split(order, blocks):
                per_block.append({q: _mean(w[rows], np.asarray(table[q], float)[rows])
                                  for q in quantities})  # fmt: skip
    if not per_run:
        raise EstError("no runs found")
    runs = list(per_run)
    if combine == "probability":
        totals = np.array(log_totals)
        weights = np.exp(totals - logsumexp(totals))
    else:
        weights = np.full(len(runs), 1.0 / len(runs))
    values, errors = {}, {}
    for q in quantities:
        column = np.array([per_run[r][q] for r in runs])
        ok = np.isfinite(column)
        values[q] = float(np.sum(weights[ok] * column[ok]) / weights[ok].sum()) \
            if ok.any() else math.nan  # fmt: skip
        if len(runs) > 1:
            errors[q] = _standard_error(column)
        else:
            errors[q] = _standard_error([b[q] for b in per_block]) if per_block else math.nan
    metadata = {
        "combined": combine,
        "run_weights": {r: float(w) for r, w in zip(runs, weights, strict=True)},
        "time_window_ns": list(time) if time else None,
        "blocks": blocks,
        "frames": frames,
        "error": "between runs" if len(runs) > 1 else ("between blocks" if per_block else None),
    }
    return AverageResult(quantities, values, errors, per_run, metadata)
