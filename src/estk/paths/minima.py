"""Finding the minima of a free-energy surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from estk.errors import EstError
from estk.surface import Surface


@dataclass(frozen=True)
class Minimum:
    """A minimum of a surface.

    ``depth`` is its persistence: how far the free energy must rise from the minimum before its
    basin joins a lower one (``inf`` for the global minimum). ``saddle`` is that joining level:
    the lowest pass to a lower basin, measured on the grid.
    """

    index: tuple[int, ...]
    point: tuple[float, ...]
    value: float
    depth: float = float("inf")
    saddle: float = float("inf")


def find_minima(
    surface: Surface,
    n: int | None = 2,
    *,
    method: str = "basins",
    min_depth: float = 0.0,
    split_axis: int = 0,
) -> list[Minimum]:
    """Minima of a free-energy surface.

    ``method="basins"``: local minima whose basin is at least ``min_depth`` deep (persistence on
    the grid, nearest neighbours along each axis), lowest first; the ``n`` lowest are returned
    (all if ``n`` is None).

    ``method="split"``: as the original scripts: cut the grid into ``n`` equal parts along
    ``split_axis`` and take the lowest point of each part, in order along that axis.
    """
    values = np.asarray(surface.values, dtype=float)
    if method == "split":
        if n is None or n < 1:
            raise EstError("split needs the number of minima")
        return _split_minima(surface, values, n, split_axis)
    if method != "basins":
        raise EstError(f"unknown method '{method}' to find minima (basins or split)")
    minima = [m for m in _persistent_minima(surface, values) if m.depth >= min_depth]
    minima.sort(key=lambda m: m.value)
    return minima if n is None else minima[:n]


def _point(surface: Surface, index: tuple[int, ...]) -> tuple[float, ...]:
    axes = surface.grid.axes()
    return tuple(float(axes[d][i]) for d, i in enumerate(index))


def _split_minima(surface: Surface, values: np.ndarray, n: int, axis: int) -> list[Minimum]:
    size = values.shape[axis]
    base = size // n
    cuts = [t * base for t in range(n)] + [size]
    minima = []
    for t in range(n):
        part = [slice(None)] * values.ndim
        part[axis] = slice(cuts[t], cuts[t + 1])
        block = values[tuple(part)]
        local = np.unravel_index(int(np.argmin(block)), block.shape)
        index = tuple(int(i) for i in local)
        index = tuple(i + (cuts[t] if d == axis else 0) for d, i in enumerate(index))
        minima.append(Minimum(index, _point(surface, index), float(values[index])))
    return minima


def _persistent_minima(surface: Surface, values: np.ndarray) -> list[Minimum]:
    """0-dimensional persistence of the sublevel sets (union-find over grid points)."""
    shape = values.shape
    flat = values.reshape(-1)
    order = [int(i) for i in np.argsort(flat, kind="stable") if np.isfinite(flat[i])]
    parent = np.full(flat.size, -1)
    strides = np.array([int(np.prod(shape[d + 1 :])) for d in range(len(shape))])

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    birth: dict[int, int] = {}  # component root -> index of its minimum
    deaths: dict[int, float] = {}  # index of a minimum -> level at which its basin merged
    for cell in order:
        coords = np.unravel_index(cell, shape)
        roots = set()
        for d in range(len(shape)):
            for step in (-1, 1):
                c = coords[d] + step
                if 0 <= c < shape[d]:
                    neighbour = cell + step * int(strides[d])
                    if parent[neighbour] >= 0:
                        roots.add(root(neighbour))
        parent[cell] = cell
        if not roots:
            birth[cell] = cell
            continue
        # The basin with the lowest minimum survives; the others end here.
        keep = min(roots, key=lambda r: flat[birth[r]])
        for r in roots:
            if r != keep:
                deaths[birth[r]] = float(flat[cell])
                parent[r] = keep
        parent[cell] = keep

    minima = []
    for m in birth.values():
        index = tuple(int(i) for i in np.unravel_index(m, shape))
        level = deaths.get(m, float("inf"))
        depth = level - float(flat[m]) if m in deaths else float("inf")
        minima.append(Minimum(index, _point(surface, index), float(flat[m]), depth, level))
    return minima
