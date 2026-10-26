"""Reweighting biased runs to free-energy surfaces and property averages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.errors import EstError
from estk.grid import Grid, Limits, resolve_limits
from estk.io.colvar import CVTable
from estk.logging import get_logger
from estk.reweight.bandwidth import effective_sample_size, silverman
from estk.reweight.combine import combine as combine_surfaces
from estk.reweight.kde import kde_on_grid
from estk.surface import Surface

if TYPE_CHECKING:
    from estk.project import Project

log = get_logger(__name__)

DEFAULT_POINTS = 100
COMBINED_STEM = "combined"  # <folder>/combined.dat


def log_weights(bias: np.ndarray, kbt: float) -> np.ndarray:
    """Log of the reweighting factor of each frame, ``bias / kT`` (OPES/METAD convention)."""
    if kbt <= 0:
        raise EstError("kT must be positive")
    return np.asarray(bias, dtype=float) / kbt


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    peak = float(np.max(values))
    return peak + math.log(float(np.sum(np.exp(values - peak))))


def choose_bandwidths(
    table: CVTable,
    cvs: Sequence[str],
    weights: np.ndarray,
    *,
    explicit: Mapping[str, float] | None = None,
    configured: Mapping[str, float | None] | None = None,
    bias_factor: float | None = None,
) -> tuple[np.ndarray, dict[str, str]]:
    """Bandwidth of each CV: ``explicit`` > ``configured`` (estk.yaml) > Silverman's rule."""
    values, sources = [], {}
    for name in cvs:
        if explicit and explicit.get(name) is not None:
            values.append(float(explicit[name]))
            sources[name] = "command line"
        elif configured and configured.get(name) is not None:
            values.append(float(configured[name]))  # type: ignore[arg-type]
            sources[name] = "estk.yaml"
        else:
            values.append(silverman(table[name], weights, ndim=len(cvs), bias_factor=bias_factor))
            sources[name] = "silverman"
    return np.array(values), sources


def reweight_table(
    table: CVTable,
    grid: Grid,
    *,
    quantity: str | None = None,
    bandwidth: Mapping[str, float] | None = None,
    configured_bandwidth: Mapping[str, float | None] | None = None,
    bias_factor: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Surface:
    """Reweight one run's frames onto ``grid``.

    With ``quantity=None`` the result is the free-energy surface; otherwise the kernel-weighted
    average of that column of ``table``. Frame weights are ``exp(bias / kT)`` normalised to
    sum 1; ``metadata["log_weight_total"]`` is ``log(sum exp(bias / kT))`` before normalisation.
    """
    if table.bias is None:
        raise EstError(f"{table.run_name}: the table has no bias column")
    if table.kbt is None:
        raise EstError(
            f"{table.run_name}: temperature unknown; set simulation.temperature in estk.yaml"
        )
    cvs = list(grid.names)
    raw = log_weights(table.bias, table.kbt)
    log_total = logsumexp(raw)
    logw = raw - log_total
    weights = np.exp(logw)

    bandwidths, bw_sources = choose_bandwidths(
        table,
        cvs,
        weights,
        explicit=bandwidth,
        configured=configured_bandwidth,
        bias_factor=bias_factor,
    )
    quantities = {quantity: table[quantity]} if quantity else None
    log_p, averages = kde_on_grid(table.array(cvs), logw, grid, bandwidths, quantities=quantities)

    meta: dict[str, Any] = {
        "run": table.run_name,
        "frames": len(table),
        "effective_frames": round(effective_sample_size(weights), 1),
        "kbt": table.kbt,
        "bias_factor": bias_factor,
        "bandwidth": {n: float(b) for n, b in zip(cvs, bandwidths, strict=True)},
        "bandwidth_source": bw_sources,
        "log_weight_total": log_total,
        "log_weight_max": float(raw.max()),
        "walls_in_bias": table.report.walls_in_bias,
        "time_range_ns": [
            round(float(table.time[0]) / 1000, 9),
            round(float(table.time[-1]) / 1000, 9),
        ]
        if len(table)
        else None,
    }
    meta.update(metadata or {})

    free_energy = -table.kbt * log_p
    finite = np.isfinite(free_energy)
    if not finite.any():
        raise EstError(f"{table.run_name}: no frames inside the grid")
    offset = float(free_energy[finite].min())
    meta["fes_offset"] = offset
    meta["energy_units"] = table.energy_units
    if quantity is None:
        return Surface(
            grid=grid, values=free_energy - offset, units=table.energy_units, metadata=meta
        )
    return Surface(
        grid=grid,
        values=averages[quantity],
        quantity=quantity,
        units="",
        metadata=meta,
        fes=free_energy - offset,
    )


@dataclass
class ReweightResult:
    """Per-run surfaces on a common grid."""

    grid: Grid
    surfaces: dict[str, Surface]
    limit_sources: dict[str, tuple[str, str]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    tables: dict[str, CVTable] = field(default_factory=dict)
    combined: Surface | None = None

    def save(self, folder: Path | str, *, compress: bool = False) -> list[Path]:
        """Write ``<folder>/<run>.dat`` for each run and ``<folder>/combined.dat`` (``.dat.gz``,
        gzip-compressed, with ``compress``)."""
        folder = Path(folder)
        suffix = ".dat.gz" if compress else ".dat"
        paths = [s.save(folder / f"{name}{suffix}") for name, s in self.surfaces.items()]
        if self.combined is not None:
            paths.append(self.combined.save(folder / f"{COMBINED_STEM}{suffix}"))
        return paths


def reweight(
    project: Project,
    cvs: Sequence[str],
    *,
    quantity: str | None = None,
    source: str | Path | None = None,
    time: tuple[float | None, float | None] | None = None,
    points: int | Sequence[int] = DEFAULT_POINTS,
    limits: Mapping[str, Limits] | None = None,
    bandwidth: Mapping[str, float] | None = None,
    include_walls: bool = False,
    jobs: int = 1,
    skip_failed: bool = False,
    combine: str | None = "probability",
) -> ReweightResult:
    """Reweight every run of ``project`` onto one grid, and combine the runs.

    Parameters
    ----------
    cvs:
        The CVs of the grid (any number; 2 is typical).
    quantity, source:
        Average this column instead of computing the free energy. ``source`` is the file it is
        in when it is not a CV (e.g. ``model_devi.out``).
    time:
        Time window ``(start, end)`` in ns.
    points:
        Grid points per CV (one number for all CVs, or one per CV).
    limits:
        ``{cv: (min, max)}`` overriding estk.yaml, the PLUMED walls and the data range.
    bandwidth:
        ``{cv: h}`` overriding estk.yaml and Silverman's rule.
    include_walls:
        Add the PLUMED wall energies to the bias.
    jobs:
        Runs computed in parallel.
    skip_failed:
        Leave out runs that fail (they are listed in ``failed``) instead of stopping.
    combine:
        How to combine the runs into ``result.combined`` (see ``estk.reweight.combine``):
        ``"probability"``, ``"equal"``, ``"legacy"``, or None for no combination.
    """
    cvs = list(cvs)
    if not cvs:
        raise EstError("no CVs given")
    if len(set(cvs)) != len(cvs):
        raise EstError(f"CVs listed twice: {cvs}")
    point_list = [points] * len(cvs) if isinstance(points, int) else list(points)
    if len(point_list) != len(cvs):
        raise EstError(f"{len(point_list)} values for points but {len(cvs)} CVs")

    extra = {quantity: source} if quantity and source else None
    to_load = cvs + ([quantity] if quantity and not source and quantity not in cvs else [])

    tables: dict[str, CVTable] = {}
    failed: dict[str, str] = {}
    for run in project.runs:
        try:
            tables[run.name] = run.load_cvs(
                to_load, time=time, extra=extra, include_walls=include_walls
            )
        except EstError as exc:
            if not skip_failed:
                raise
            failed[run.name] = str(exc)
            log.warning("skipping %s: %s", run.name, exc)
    if not tables:
        raise EstError("no run could be loaded")

    runs = {run.name: run for run in project.runs if run.name in tables}
    data_range = {
        name: (
            float(min(t[name].min() for t in tables.values())),
            float(max(t[name].max() for t in tables.values())),
        )
        for name in cvs
    }
    grid_limits, limit_sources = resolve_limits(
        cvs,
        explicit=limits,
        config=project.config,
        walls=[r.metadata.walls for r in runs.values()],
        data=data_range,
    )
    grid = Grid(
        names=tuple(cvs),
        lower=tuple(grid_limits[n][0] for n in cvs),
        upper=tuple(grid_limits[n][1] for n in cvs),
        points=tuple(int(p) for p in point_list),
    )
    configured = {n: project.config.cv[n].bandwidth for n in cvs if n in project.config.cv}

    jobs_args = {}
    for name, table in tables.items():
        meta = runs[name].metadata
        jobs_args[name] = dict(
            table=table,
            grid=grid,
            quantity=quantity,
            bandwidth=bandwidth,
            configured_bandwidth=configured,
            bias_factor=meta.bias_factor,
            metadata={
                "temperature": meta.temperature,
                "time_window_ns": list(time) if time else None,
            },
        )

    surfaces: dict[str, Surface] = {}
    if jobs > 1 and len(jobs_args) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {n: pool.submit(_reweight_job, a) for n, a in jobs_args.items()}
            outcomes = {n: _outcome(f.result) for n, f in futures.items()}
    else:
        outcomes = {n: _outcome(lambda a=a: _reweight_job(a)) for n, a in jobs_args.items()}
    for name, (surface, error) in outcomes.items():
        if error is None:
            surfaces[name] = surface  # type: ignore[assignment]
        elif skip_failed:
            failed[name] = str(error)
            log.warning("skipping %s: %s", name, error)
        else:
            raise error
    if not surfaces:
        raise EstError("reweighting failed for every run")
    combined = combine_surfaces(surfaces, mode=combine) if combine else None
    return ReweightResult(
        grid=grid,
        surfaces=surfaces,
        limit_sources=limit_sources,
        failed=failed,
        tables=tables,
        combined=combined,
    )


def _reweight_job(args: dict[str, Any]) -> Surface:
    return reweight_table(**args)


def _outcome(call: Any) -> tuple[Surface | None, EstError | None]:
    try:
        return call(), None
    except EstError as exc:
        return None, exc
