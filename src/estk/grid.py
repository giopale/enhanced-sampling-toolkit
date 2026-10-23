"""Regular grids in CV space and how their limits are chosen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from estk.config import ProjectConfig
from estk.errors import EstError, MissingLimitsError
from estk.logging import get_logger

log = get_logger(__name__)

Limits = tuple[float | None, float | None]


@dataclass(frozen=True)
class Grid:
    """A regular grid including both end points: ``points[d]`` values from ``lower[d]`` to
    ``upper[d]`` along each CV ``names[d]``. Arrays on the grid have shape ``points`` and use
    ``ij`` indexing (axis d is CV d)."""

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    points: tuple[int, ...]

    def __post_init__(self) -> None:
        n = len(self.names)
        if not (len(self.lower) == len(self.upper) == len(self.points) == n) or n == 0:
            raise EstError("grid: names, lower, upper and points must have the same length")
        for name, lo, hi, p in zip(self.names, self.lower, self.upper, self.points, strict=True):
            if not lo < hi:
                raise EstError(f"grid: lower limit of {name} ({lo}) must be below the upper ({hi})")
            if p < 2:
                raise EstError(f"grid: {name} needs at least 2 points, got {p}")

    @property
    def ndim(self) -> int:
        return len(self.names)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.points)

    def axes(self) -> list[np.ndarray]:
        return [
            np.linspace(lo, hi, p)
            for lo, hi, p in zip(self.lower, self.upper, self.points, strict=True)
        ]

    def mesh(self) -> list[np.ndarray]:
        return np.meshgrid(*self.axes(), indexing="ij")

    def coordinates(self) -> np.ndarray:
        """All grid points as an array of shape (prod(points), ndim), in C order."""
        return np.column_stack([m.reshape(-1) for m in self.mesh()])


def resolve_limits(
    names: Sequence[str],
    *,
    explicit: Mapping[str, Limits] | None = None,
    config: ProjectConfig | None = None,
    walls: Sequence[Mapping[str, Limits]] = (),
    data: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[str, str]]]:
    """Choose the grid limits of each CV, separately for the lower and upper end.

    Order of precedence: ``explicit`` (e.g. CLI) > ``config`` (estk.yaml) > PLUMED ``walls``
    (one mapping per run; if runs disagree the outermost position is used) > ``data`` range.

    Returns ``(limits, sources)`` where ``sources[name]`` names where each end came from.
    """
    limits: dict[str, tuple[float, float]] = {}
    sources: dict[str, tuple[str, str]] = {}
    for name in names:
        candidates: list[tuple[str, Limits]] = []
        if explicit and name in explicit:
            candidates.append(("command line", explicit[name]))
        if config is not None and name in config.cv:
            candidates.append(("estk.yaml", config.cv[name].limits))
        candidates.append(("PLUMED walls", _wall_limits(name, walls)))
        if data and name in data:
            candidates.append(("data range", data[name]))

        ends: list[float] = []
        where: list[str] = []
        for side in (0, 1):
            for source, values in candidates:
                if values[side] is not None:
                    ends.append(float(values[side]))  # type: ignore[arg-type]
                    where.append(source)
                    break
            else:
                raise MissingLimitsError(
                    f"no {'lower' if side == 0 else 'upper'} limit for {name}: set "
                    f"cv.{name}.limits in estk.yaml or pass --limits"
                )
        limits[name] = (ends[0], ends[1])
        sources[name] = (where[0], where[1])
    return limits, sources


def _wall_limits(name: str, walls: Sequence[Mapping[str, Limits]]) -> Limits:
    lows = [w[name][0] for w in walls if name in w and w[name][0] is not None]
    highs = [w[name][1] for w in walls if name in w and w[name][1] is not None]
    if len(set(lows)) > 1 or len(set(highs)) > 1:
        log.warning("the PLUMED walls on %s differ between runs; using the outermost", name)
    return (min(lows) if lows else None, max(highs) if highs else None)  # type: ignore[type-var]
