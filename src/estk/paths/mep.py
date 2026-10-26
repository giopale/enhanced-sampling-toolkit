"""Minimum energy paths and barriers on free-energy surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from estk.errors import EstError
from estk.io.textfile import PREFIX, decode, metadata_lines, open_text, read_header
from estk.logging import get_logger
from estk.paths.minima import Minimum, find_minima
from estk.paths.string import evolve_string, initial_string
from estk.surface import Surface

log = get_logger(__name__)

DEFAULT_IMAGES = 15
DEFAULT_MIN_DEPTH_KT = 2.0


@dataclass
class MinimumEnergyPath:
    """Images of a path on a surface and the free energy along it.

    ``energies`` are the values of the surface the path was computed on (minimum of the surface
    = 0). ``profiles`` holds the energies of other surfaces along the same path (e.g. each run),
    ``error`` the error of the surface along the path, if it has one.
    """

    names: tuple[str, ...]
    points: np.ndarray  # (images, ndim)
    energies: np.ndarray  # (images,)
    units: str = "kj/mol"
    error: np.ndarray | None = None
    profiles: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def s(self) -> np.ndarray:
        """Arc length along the path, from 0 at the start to 1 at the end."""
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(lengths)])
        return s / s[-1] if s[-1] > 0 else s

    @property
    def saddle_index(self) -> int:
        return int(np.nanargmax(self.energies))

    @property
    def barrier(self) -> float:
        """Highest energy along the path minus the energy at the start."""
        return float(np.nanmax(self.energies) - self.energies[0])

    @property
    def reverse_barrier(self) -> float:
        """Highest energy along the path minus the energy at the end."""
        return float(np.nanmax(self.energies) - self.energies[-1])

    @property
    def reaction_energy(self) -> float:
        """Energy at the end minus the energy at the start."""
        return float(self.energies[-1] - self.energies[0])

    def profile_barriers(self) -> dict[str, float]:
        """The barrier (highest minus start) of each profile in ``profiles``."""
        return {k: float(np.nanmax(v) - v[0]) for k, v in self.profiles.items()}

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        fields = ["s", *self.names, "fes"]
        columns = [self.s, *self.points.T, self.energies]
        if self.error is not None:
            fields.append("fes.err")
            columns.append(self.error)
        for name, values in self.profiles.items():
            fields.append(f"fes.{name}")
            columns.append(values)
        meta = {
            "units": self.units,
            "barrier": self.barrier,
            "reverse_barrier": self.reverse_barrier,
            "reaction_energy": self.reaction_energy,
            **self.metadata,
        }
        header = [f"#! FIELDS {' '.join(fields)}", *metadata_lines(meta)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open_text(path, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, np.column_stack(columns), fmt="%.10g")
        return path

    @classmethod
    def load(cls, path: Path | str) -> MinimumEnergyPath:
        path = Path(path)
        fields, sets = read_header(path)
        if "fes" not in fields or fields[0] != "s":
            raise EstError(f"{path}: not a path file (FIELDS {' '.join(fields)})")
        meta = {k[len(PREFIX) :]: decode(v) for k, v in sets.items() if k.startswith(PREFIX)}
        data = np.loadtxt(path, comments="#", ndmin=2)
        i_fes = fields.index("fes")
        names = tuple(fields[1:i_fes])
        error = data[:, fields.index("fes.err")] if "fes.err" in fields else None
        profiles = {
            f[4:]: data[:, j]
            for j, f in enumerate(fields)
            if f.startswith("fes.") and f != "fes.err"
        }
        units = meta.pop("units", "kj/mol")
        for key in ("barrier", "reverse_barrier", "reaction_energy"):
            meta.pop(key, None)
        return cls(
            names=names,
            points=data[:, 1:i_fes],
            energies=data[:, i_fes],
            units=units,
            error=error,
            profiles=profiles,
            metadata=meta,
        )


def evaluate(surface: Surface, points: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    """Linear interpolation of the surface (or of ``values`` on its grid) at ``points``."""
    values = surface.values if values is None else values
    interpolator = RegularGridInterpolator(
        surface.grid.axes(), values, method="linear", bounds_error=False, fill_value=np.nan
    )
    with np.errstate(invalid="ignore"):
        return interpolator(np.asarray(points, dtype=float))


def minimum_energy_path(
    surface: Surface,
    *,
    start: Sequence[float] | None = None,
    end: Sequence[float] | None = None,
    via: Sequence[Sequence[float]] = (),
    method: str = "basins",
    minima: int = 2,
    split_axis: int | str = 0,
    min_depth_kt: float = DEFAULT_MIN_DEPTH_KT,
    reverse: bool = False,
    images: int = DEFAULT_IMAGES,
    dt: float = 0.01,
    max_steps: int = 2000,
    fixed_ends: bool = True,
    cap: float | None = None,
    profiles: Mapping[str, Surface] | None = None,
) -> MinimumEnergyPath:
    """Minimum energy path between two minima of a free-energy surface.

    The ends are ``start`` and ``end`` if given; otherwise minima found with ``method``:

    - ``"basins"``: the ``minima`` lowest minima whose basin is at least ``min_depth_kt`` kT deep;
      start and end are the two lowest, in order along the first CV (``via`` = a third one);
    - ``"split"``: as the original scripts, the lowest point of each of ``minima`` equal parts
      of the grid along ``split_axis`` (path from the last part to the first when splitting the
      second CV, from the first to the last otherwise; a middle part's minimum is used as
      ``via``).

    ``reverse`` swaps start and end. The string has ``images`` images and is relaxed with steps
    of ``dt`` (CV units) for up to ``max_steps``; with ``fixed_ends`` the ends stay on the
    minima. Free energies above ``cap`` (default: the highest finite value) are set to ``cap``
    for the relaxation. ``profiles`` are other surfaces (e.g. each run) to evaluate along the
    path.
    """
    if not surface.is_fes:
        raise EstError(f"a minimum energy path needs a free-energy surface, not {surface.quantity}")
    names = surface.grid.names
    axis = names.index(split_axis) if isinstance(split_axis, str) else int(split_axis)
    kbt = surface.metadata.get("kbt")
    min_depth = min_depth_kt * kbt if kbt else 0.0

    found: list[Minimum] = []
    if start is None or end is None:
        found = find_minima(surface, minima, method=method, min_depth=min_depth, split_axis=axis)
        if len(found) < 2:
            raise EstError(
                f"found {len(found)} minimum(s) deeper than {min_depth_kt:g} kT; lower "
                "--min-depth or give --start and --end"
            )
        if method == "split":
            ordered = found[::-1] if axis == 1 else found
            auto_start, auto_end = ordered[0].point, ordered[-1].point
            auto_via = [m.point for m in ordered[1:-1]]
        else:
            ends = sorted(found[:2], key=lambda m: m.point)
            auto_start, auto_end = ends[0].point, ends[1].point
            auto_via = [m.point for m in found[2:3]]
        start = auto_start if start is None else start
        end = auto_end if end is None else end
        via = via or auto_via
    if reverse:
        start, end, via = end, start, list(via)[::-1]
    for label, point in (("start", start), ("end", end), *(("via", v) for v in via)):
        if len(point) != len(names):
            raise EstError(f"{label} point needs {len(names)} values ({', '.join(names)})")

    values = np.array(surface.values, dtype=float)
    finite = np.isfinite(values)
    ceiling = float(values[finite].max()) if cap is None else float(cap)
    capped = np.where(finite, np.minimum(values, ceiling), ceiling)

    knots = [list(start), *[list(v) for v in via], list(end)]
    if not via:
        # As the original scripts: the initial guess goes through the midpoint.
        knots = [list(start), list((np.array(start) + np.array(end)) / 2), list(end)]
    result = evolve_string(
        capped,
        surface.grid.axes(),
        initial_string(knots, images),
        dt=dt,
        max_steps=max_steps,
        fixed_ends=fixed_ends,
    )
    if not result.converged:
        log.warning("the string did not converge in %d steps", max_steps)

    points = result.points
    energies = evaluate(surface, points)
    error = evaluate(surface, points, surface.error) if surface.error is not None else None
    profile_values = {name: evaluate(s, points) for name, s in (profiles or {}).items()}

    meta: dict[str, Any] = {
        "method": "given" if not found else method,
        "minima": [{"point": list(m.point), "value": m.value, "depth": m.depth} for m in found],
        "images": images,
        "dt": dt,
        "fixed_ends": fixed_ends,
        "cap": ceiling,
        "converged": result.converged,
        "steps": result.steps,
    }
    for key in ("kbt", "temperature", "combined", "run", "runs"):
        if key in surface.metadata:
            meta[key] = surface.metadata[key]
    return MinimumEnergyPath(
        names=tuple(names),
        points=points,
        energies=energies,
        units=surface.units,
        error=error,
        profiles=profile_values,
        metadata=meta,
    )
