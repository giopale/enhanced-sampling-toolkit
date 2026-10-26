"""How the barrier converges with simulation time or with the kernel bandwidth.

Each step reweights the runs (with a growing time window, or with another bandwidth for one CV)
on the same grid, combines them, and measures the barrier along a minimum energy path:

- ``relax=False`` (default, as the original scripts): the energies are read along one fixed
  path, computed on the reference surface (all the data / the default bandwidth) or given;
- ``relax=True``: the path is relaxed again on each surface, from the same two ends.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.errors import EstError, MissingFileError
from estk.logging import get_logger
from estk.paths import MinimumEnergyPath, evaluate, minimum_energy_path

if TYPE_CHECKING:
    from estk.project import Project

log = get_logger(__name__)

PARAMETERS = ("time", "bandwidth")


@dataclass
class ConvergenceRow:
    value: float  # end of the time window (ns) or bandwidth
    frames: int
    barrier: float
    reverse_barrier: float
    reaction_energy: float
    run_barriers: dict[str, float] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        """Standard deviation of the runs' barriers (NaN with fewer than two runs)."""
        values = np.array(list(self.run_barriers.values()), dtype=float)
        return float(np.std(values)) if len(values) > 1 else float("nan")


@dataclass
class ConvergenceResult:
    parameter: str  # "time" or "bandwidth"
    label: str  # x axis, e.g. "simulation time (ns)" or "bandwidth of d0"
    rows: list[ConvergenceRow]
    units: str = "kj/mol"
    path: MinimumEnergyPath | None = None  # the reference path
    metadata: dict[str, Any] = field(default_factory=dict)

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(r, name) for r in self.rows], dtype=float)

    def save(self, path: Path | str) -> Path:
        """Write a table: value, frames, barrier, reverse_barrier, reaction_energy, spread, runs."""
        path = Path(path)
        runs = sorted({k for r in self.rows for k in r.run_barriers})
        fields = ["value", "frames", "barrier", "reverse_barrier", "reaction_energy", "spread"]
        fields += [f"barrier.{r}" for r in runs]
        meta = {"parameter": self.parameter, "label": self.label, "units": self.units}
        meta.update(self.metadata)
        lines = [f"#! FIELDS {' '.join(fields)}"]
        lines += [f"#! SET estk.{k} {json.dumps(v)}" for k, v in meta.items()]
        for r in self.rows:
            values = [r.value, r.frames, r.barrier, r.reverse_barrier, r.reaction_energy, r.spread]
            values += [r.run_barriers.get(k, float("nan")) for k in runs]
            lines.append(" ".join(f"{v:.10g}" for v in values))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> ConvergenceResult:
        path = Path(path)
        if not path.is_file():
            raise MissingFileError(f"convergence file not found: {path}")
        fields: list[str] = []
        meta: dict[str, Any] = {}
        for line in path.read_text().splitlines():
            if line.startswith("#! FIELDS"):
                fields = line.split()[2:]
            elif line.startswith("#! SET estk."):
                _, _, key, raw = line.split(maxsplit=3)
                meta[key[5:]] = json.loads(raw)
        data = np.loadtxt(path, comments="#", ndmin=2)
        runs = [f[8:] for f in fields if f.startswith("barrier.")]
        rows = []
        for line in data:
            values = dict(zip(fields, line, strict=True))
            rows.append(
                ConvergenceRow(
                    value=values["value"],
                    frames=int(values["frames"]),
                    barrier=values["barrier"],
                    reverse_barrier=values["reverse_barrier"],
                    reaction_energy=values["reaction_energy"],
                    run_barriers={r: values[f"barrier.{r}"] for r in runs},
                )
            )
        return cls(
            parameter=meta.pop("parameter"),
            label=meta.pop("label"),
            rows=rows,
            units=meta.pop("units", "kj/mol"),
            metadata=meta,
        )


def barrier_convergence(
    project: Project,
    cvs: Sequence[str],
    parameter: str,
    values: Sequence[float],
    *,
    cv: str | None = None,
    start: float | None = None,
    time: tuple[float | None, float | None] | None = None,
    path: MinimumEnergyPath | None = None,
    relax: bool = False,
    reweight_options: Mapping[str, Any] | None = None,
    path_options: Mapping[str, Any] | None = None,
    save_steps: Path | str | None = None,
) -> ConvergenceResult:
    """Barrier as a function of the end of the time window or of one CV's bandwidth.

    Parameters
    ----------
    parameter:
        ``"time"``: ``values`` are ends of the time window in ns (from ``start``, default 0);
        ``"bandwidth"``: ``values`` are bandwidths of ``cv`` (time window ``time``).
    path:
        Reference path. Default: the minimum energy path (``path_options``) on the reference
        surface: all of the time window (time) or the usual bandwidths (bandwidth).
    relax:
        Relax the path again on each surface (ends fixed at the reference path's ends) instead
        of reading the energies along the reference path.
    reweight_options:
        Passed to ``estk.reweight.reweight`` (e.g. ``points``, ``limits``, ``bandwidth``,
        ``include_walls``, ``combine``, ``jobs``). The grid of the reference surface is used for
        every step.
    save_steps:
        Folder to write each step's combined surface and path to.
    """
    if parameter not in PARAMETERS:
        raise EstError(f"unknown parameter '{parameter}' (time or bandwidth)")
    if not values:
        raise EstError("no values to scan")
    options = dict(reweight_options or {})
    combine = options.pop("combine", "probability") or "probability"
    for key in ("time", "quantity", "source"):
        if key in options:
            raise EstError(f"'{key}' can't be set in reweight_options here")
    cvs = list(cvs)
    if parameter == "bandwidth":
        if cv is None or cv not in cvs:
            raise EstError(f"choose which CV's bandwidth to vary (one of {', '.join(cvs)})")
        label = f"bandwidth of {cv}"
    else:
        label = "simulation time (ns)"
    values = sorted(float(v) for v in values)

    # Reference surface: fixes the grid and, unless given, the path.
    if parameter == "time":
        reference_time: tuple[float | None, float | None] | None = (start, max(values))
    else:
        reference_time = time
    reference = project.reweight(cvs, time=reference_time, combine=combine, **options)
    grid = reference.grid
    fixed_grid = {
        "points": grid.points,
        "limits": {
            n: (lo, hi) for n, lo, hi in zip(grid.names, grid.lower, grid.upper, strict=True)
        },
    }
    step_options = {**options, **fixed_grid}
    if path is None:
        reaction = project.config.reaction
        hints = {"start_near": reaction.start or None, "end_near": reaction.end or None}
        path = minimum_energy_path(reference.combined, **{**hints, **dict(path_options or {})})
    if tuple(path.names) != tuple(cvs):
        raise EstError(f"the path is in {path.names}, the surfaces in {tuple(cvs)}")

    rows: list[ConvergenceRow] = []
    previous_frames = None
    for value in values:
        if parameter == "time":
            if start is not None and value <= start:
                raise EstError(f"time {value} ns is not after the start {start} ns")
            result = project.reweight(cvs, time=(start, value), combine=combine, **step_options)
        else:
            bandwidth = {**dict(options.get("bandwidth") or {}), cv: value}
            result = project.reweight(
                cvs, time=time, combine=combine, **{**step_options, "bandwidth": bandwidth}
            )
        combined = result.combined
        if relax:
            step_path = minimum_energy_path(
                combined,
                start=path.points[0],
                end=path.points[-1],
                **{k: v for k, v in dict(path_options or {}).items() if k not in ("start", "end")},
                profiles=result.surfaces,
            )
        else:
            step_path = MinimumEnergyPath(
                names=path.names,
                points=path.points,
                energies=evaluate(combined, path.points),
                units=combined.units,
                profiles={n: evaluate(s, path.points) for n, s in result.surfaces.items()},
                metadata={"kbt": combined.metadata.get("kbt")},
            )
        frames = int(combined.metadata.get("frames", 0))
        if parameter == "time" and previous_frames is not None and frames == previous_frames:
            log.warning("no new frames between steps at %g ns: the runs may be shorter", value)
        previous_frames = frames
        rows.append(
            ConvergenceRow(
                value=value,
                frames=frames,
                barrier=step_path.barrier,
                reverse_barrier=step_path.reverse_barrier,
                reaction_energy=step_path.reaction_energy,
                run_barriers=step_path.profile_barriers(),
            )
        )
        log.info("%s = %g: barrier %.3f %s", parameter, value, step_path.barrier, combined.units)
        if save_steps is not None:
            folder = Path(save_steps)
            combined.save(folder / f"{parameter}-{value:g}.dat")
            step_path.save(folder / f"{parameter}-{value:g}.mep.dat")

    units = reference.combined.units
    meta: dict[str, Any] = {
        "cvs": cvs,
        "relax": relax,
        "combine": combine,
        "kbt": reference.combined.metadata.get("kbt"),
    }
    if parameter == "time":
        meta["start_ns"] = start
    else:
        meta["cv"] = cv
        meta["time_window_ns"] = list(time) if time else None
    return ConvergenceResult(parameter, label, rows, units=units, path=path, metadata=meta)
