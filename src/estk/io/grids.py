"""Reading and writing surfaces as PLUMED-style grid text files.

Format written by estk::

    #! FIELDS d0 cn fes [fes.err]          (free energy)
    #! FIELDS d0 cn q [q.err] fes           (property average, with the free energy)
    #! SET min_d0 1.5
    #! SET max_d0 2.5
    #! SET num_grid_points_d0 100
    ...
    #! SET estk.units "kj/mol"
    #! SET estk.<key> <JSON value>
    x y value [error]

The first CV varies fastest, with a blank line after each sweep of it (as in PLUMED grids).
``estk.version`` and ``estk.command`` record what wrote the file. Names ending in ``.gz`` are
gzip-compressed. The reader also accepts the files of the original scripts (``# FIELDS``/``# SET``
header, any row order, ``num_grid_points_*`` or ``nbins_*``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from estk.errors import EstError
from estk.grid import Grid
from estk.io.textfile import PREFIX, decode, metadata_lines, open_text, read_header
from estk.surface import Surface


def write_surface(surface: Surface, path: Path) -> Path:
    grid = surface.grid
    fields = [*grid.names, surface.quantity]
    if surface.error is not None:
        fields.append(f"{surface.quantity}.err")
    if surface.fes is not None and not surface.is_fes:
        fields.append("fes")
    header = [f"#! FIELDS {' '.join(fields)}"]
    for name, lo, hi, n in zip(grid.names, grid.lower, grid.upper, grid.points, strict=True):
        header += [
            f"#! SET min_{name} {lo!r}",
            f"#! SET max_{name} {hi!r}",
            f"#! SET num_grid_points_{name} {n}",
            f"#! SET periodic_{name} false",
        ]
    header += metadata_lines({"units": surface.units, **surface.metadata})

    # Fortran order = first CV fastest.
    columns = [m.reshape(-1, order="F") for m in grid.mesh()]
    columns.append(surface.values.reshape(-1, order="F"))
    if surface.error is not None:
        columns.append(np.asarray(surface.error).reshape(-1, order="F"))
    if surface.fes is not None and not surface.is_fes:
        columns.append(np.asarray(surface.fes).reshape(-1, order="F"))
    data = np.column_stack(columns)

    path.parent.mkdir(parents=True, exist_ok=True)
    sweep = grid.points[0]
    with open_text(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for start in range(0, len(data), sweep):
            np.savetxt(f, data[start : start + sweep], fmt="%.10g")
            if grid.ndim > 1:
                f.write("\n")
    return path


def read_surface(path: Path) -> Surface:
    fields, sets = read_header(path)

    names = [f for f in fields if f"num_grid_points_{f}" in sets or f"nbins_{f}" in sets] or fields[
        :2
    ]
    ndim = len(names)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < ndim + 1:
        raise EstError(f"{path}: expected at least {ndim + 1} columns")

    declared_axes = _declared_axes(names, sets)
    if declared_axes is not None:  # estk's files: the grid is in the header
        axes = declared_axes
        index = tuple(_snap(path, name, data[:, d], axes[d]) for d, name in enumerate(names))
    else:  # other files: the grid from the values in the columns
        axes = [np.unique(data[:, d]) for d in range(ndim)]
        for name, a in zip(names, axes, strict=True):
            declared = sets.get(f"num_grid_points_{name}") or sets.get(f"nbins_{name}")
            if declared is not None and int(float(declared)) != len(a):
                raise EstError(f"{path}: {name} has {len(a)} distinct values, header says "
                               f"{declared}")  # fmt: skip
        index = tuple(np.searchsorted(axes[d], data[:, d]) for d in range(ndim))
    points = tuple(len(a) for a in axes)
    flat = np.ravel_multi_index(index, points)
    if len(np.unique(flat)) != len(flat):
        row = int(np.flatnonzero(np.bincount(flat) > 1)[0])
        raise EstError(f"{path}: grid point {np.unravel_index(row, points)} appears more than "
                       "once; the file may be damaged: write it again")  # fmt: skip
    if len(data) != int(np.prod(points)):
        raise EstError(
            f"{path}: {len(data)} grid lines for a grid of {' x '.join(map(str, points))}"
        )
    grid = Grid(
        names=tuple(names),
        lower=tuple(float(a[0]) for a in axes),
        upper=tuple(float(a[-1]) for a in axes),
        points=points,
    )
    quantity = fields[ndim] if len(fields) > ndim else "value"
    fill = np.inf if quantity == "fes" else np.nan
    values = np.full(points, fill)
    values[index] = data[:, ndim]
    # Further columns: the free energy of a property surface ("fes"), otherwise the error
    # (estk writes "<quantity>.err", the old -AVG files "stddev(fes)").
    error = fes = None
    for j in range(ndim + 1, min(data.shape[1], len(fields))):
        column = np.full(points, np.inf if fields[j] == "fes" else np.nan)
        column[index] = data[:, j]
        if fields[j] == "fes" and quantity != "fes":
            fes = column
        elif error is None:
            error = column

    metadata: dict[str, Any] = {}
    units = "kj/mol"
    for key, raw in sets.items():
        if key.startswith(PREFIX):
            value = decode(raw)
            if key == f"{PREFIX}units":
                units = value
            else:
                metadata[key[len(PREFIX) :]] = value
        elif not key.startswith(("min_", "max_", "num_grid_points_", "nbins_", "periodic_")):
            metadata[key] = _number_or_text(raw)
    return Surface(
        grid=grid,
        values=values,
        quantity=quantity,
        units=units,
        error=error,
        metadata=metadata,
        fes=fes,
    )


def _number_or_text(raw: str) -> Any:
    try:
        return float(raw)
    except ValueError:
        return raw


def _declared_axes(names: list[str], sets: dict[str, str]) -> list[np.ndarray] | None:
    """The axes from ``min_<cv>``, ``max_<cv>`` and ``num_grid_points_<cv>`` (estk's header:
    both ends are grid points), or None if the header does not give all of them."""
    axes = []
    for name in names:
        keys = (f"min_{name}", f"max_{name}", f"num_grid_points_{name}")
        if not all(k in sets for k in keys):
            return None
        try:
            lo, hi, n = float(sets[keys[0]]), float(sets[keys[1]]), int(float(sets[keys[2]]))
        except ValueError:
            return None
        axes.append(np.linspace(lo, hi, n))
    return axes


def _snap(path: Path, name: str, values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Index of each value on ``axis``; an error naming the first value that is not on it."""
    step = (axis[-1] - axis[0]) / (len(axis) - 1)
    index = np.rint((values - axis[0]) / step).astype(int)
    inside = (index >= 0) & (index < len(axis))
    off = ~inside
    off[inside] = np.abs(values[inside] - axis[index[inside]]) > 1e-3 * step
    if off.any():
        row = int(np.flatnonzero(off)[0])
        raise EstError(f"{path}: {name} = {values[row]!r} on grid line {row + 1} is not a point "
                       f"of the grid in the header ({axis[0]:g} to {axis[-1]:g}, {len(axis)} "
                       "points); the file may be damaged: write it again")  # fmt: skip
    return index
