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
    start_near: Mapping[str, float] | None = None,
    end_near: Mapping[str, float] | None = None,
    images: int = DEFAULT_IMAGES,
    dt: float = 0.01,
    max_steps: int = 2000,
    fixed_ends: bool = True,
    cap: float | None = None,
    profiles: Mapping[str, Surface] | None = None,
) -> MinimumEnergyPath:
    """Minimum energy path between two minima of a free-energy surface.

    The ends are ``start`` and ``end`` if given; otherwise minima found with ``method``:

    - ``"basins"``: minima whose basin is at least ``min_depth_kt`` kT deep; start and end are
      the two lowest, in order along the first CV (``via``: the next ``minima - 2``);
    - ``"split"``: as the original scripts, the lowest point of each of ``minima`` equal parts
      of the grid along ``split_axis`` (path from the last part to the first when splitting the
      second CV, from the first to the last otherwise; a middle part's minimum is used as
      ``via``).

    ``start_near`` (``reaction.start`` in estk.yaml, e.g. ``{"cn": 2.0}``) picks as start the
    minimum nearest that point among all those found (distance over the CVs it names, in units
    of each CV's grid range); the end is then the minimum nearest ``end_near``
    (``reaction.end``) or else the lowest other one. ``reverse`` then swaps start and end.
    ``metadata["ends"]`` says how the ends were chosen.

    The string has ``images`` images and is relaxed with steps of ``dt`` (CV units) for up to
    ``max_steps``; with ``fixed_ends`` the ends stay on the minima. Free energies above ``cap``
    (default: the highest finite value) are set to ``cap`` for the relaxation. ``profiles`` are
    other surfaces (e.g. each run) to evaluate along the path.
    """
    if not surface.is_fes:
        raise EstError(f"a minimum energy path needs a free-energy surface, not {surface.quantity}")
    names = surface.grid.names
    axis = names.index(split_axis) if isinstance(split_axis, str) else int(split_axis)
    kbt = surface.metadata.get("kbt")
    min_depth = min_depth_kt * kbt if kbt else 0.0

    found: list[Minimum] = []
    ends = "given"
    if start is None or end is None:
        hints = {k: _usable(surface, v, k) for k, v in (("start", start_near), ("end", end_near))}
        chosen = any(hints.values()) and method == "basins"
        found = find_minima(surface, None if chosen else minima, method=method,
                            min_depth=min_depth, split_axis=axis)  # fmt: skip
        if len(found) < 2:
            raise EstError(
                f"found {len(found)} minimum(s) deeper than {min_depth_kt:g} kT; lower "
                "--min-depth or give --start and --end"
            )
        if method == "split":
            found = found[::-1] if axis == 1 else found  # in path order
            first, last = found[0], found[-1]
            middle = found[1:-1]
            ends = "split order"
        else:
            first, last = sorted(found[:2], key=lambda m: m.point)
            middle = [m for m in found if m not in (first, last)][: max(0, minima - 2)]
            ends = f"two lowest, smaller {names[0]} first"
        if any(hints.values()):
            first, last, middle, ends = _choose_ends(surface, found, hints, minima, method)
        auto_start, auto_end, auto_via = first.point, last.point, [m.point for m in middle]
        start = auto_start if start is None else start
        end = auto_end if end is None else end
        via = via or auto_via
    if reverse:
        start, end, via = end, start, list(via)[::-1]
        ends += ", reversed"
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
        "ends": ends,
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


def _usable(surface: Surface, point: Mapping[str, float] | None, which: str) -> dict[int, float]:
    """``{axis: value}`` for the CVs of ``point`` that are on the surface."""
    if not point:
        return {}
    used = {k: float(point[n]) for k, n in enumerate(surface.grid.names) if n in point}
    if not used:
        log.warning("reaction.%s names none of the CVs %s; ignored", which,
                    ", ".join(surface.grid.names))  # fmt: skip
    return used


def _choose_ends(
    surface: Surface,
    found: list[Minimum],
    hints: dict[str, dict[int, float]],
    minima: int,
    method: str,
) -> tuple[Minimum, Minimum, list[Minimum], str]:
    """Start and end among ``found`` (lowest first, or in split order) from the hints."""
    grid = surface.grid

    def distance(m: Minimum, target: dict[int, float]) -> float:
        return float(np.sqrt(sum(((m.point[k] - v) / (grid.upper[k] - grid.lower[k] or 1.0)) ** 2
                                 for k, v in target.items())))  # fmt: skip

    def text(target: dict[int, float], which: str) -> str:
        values = ", ".join(f"{grid.names[k]}={v:g}" for k, v in target.items())
        return f"{which} nearest {values} (reaction.{which})"

    # without a hint: the lowest remaining minimum (basins), or the split order
    rest = list(found)
    if hints["start"]:
        start = min(rest, key=lambda m: distance(m, hints["start"]))
        how_start = text(hints["start"], "start")
    else:
        start = rest[0]
        how_start = "start the lowest other minimum" if method == "basins" else "start split order"
    rest.remove(start)
    if hints["end"]:
        end = min(rest, key=lambda m: distance(m, hints["end"]))
        how_end = text(hints["end"], "end")
    else:
        end = rest[0] if method == "basins" else rest[-1]
        how_end = "end the lowest other minimum" if method == "basins" else "end split order"
    rest.remove(end)
    via = sorted(rest, key=lambda m: m.value)[: max(0, minima - 2)] if method == "basins" else rest
    return start, end, via, f"{how_start}; {how_end}"
