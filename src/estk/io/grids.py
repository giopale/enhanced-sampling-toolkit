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
The reader also accepts the files of the original scripts (``# FIELDS``/``# SET`` header, any
row order, ``num_grid_points_*`` or ``nbins_*``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from estk.errors import EstError, MissingFileError
from estk.grid import Grid
from estk.surface import Surface

_PREFIX = "estk."


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
    header.append(f"#! SET {_PREFIX}units {json.dumps(surface.units)}")
    for key, value in surface.metadata.items():
        header.append(f"#! SET {_PREFIX}{key} {json.dumps(_jsonable(value))}")

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
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for start in range(0, len(data), sweep):
            np.savetxt(f, data[start : start + sweep], fmt="%.10g")
            if grid.ndim > 1:
                f.write("\n")
    return path


def read_surface(path: Path) -> Surface:
    if not path.is_file():
        raise MissingFileError(f"surface file not found: {path}")
    fields: list[str] = []
    sets: dict[str, str] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            words = line.lstrip("#!").split(maxsplit=2)
            if len(words) >= 2 and words[0] == "FIELDS":
                fields = line.lstrip("#!").split()[1:]
            elif len(words) == 3 and words[0] == "SET":
                sets[words[1]] = words[2].strip()
    if not fields:
        raise EstError(f"{path}: no FIELDS header")

    names = [f for f in fields if f"num_grid_points_{f}" in sets or f"nbins_{f}" in sets] or fields[
        :2
    ]
    ndim = len(names)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] < ndim + 1:
        raise EstError(f"{path}: expected at least {ndim + 1} columns")

    axes = [np.unique(data[:, d]) for d in range(ndim)]
    points = tuple(len(a) for a in axes)
    for name, n in zip(names, points, strict=True):
        declared = sets.get(f"num_grid_points_{name}") or sets.get(f"nbins_{name}")
        if declared is not None and int(float(declared)) != n:
            raise EstError(f"{path}: {name} has {n} distinct values, header says {declared}")
    grid = Grid(
        names=tuple(names),
        lower=tuple(float(a[0]) for a in axes),
        upper=tuple(float(a[-1]) for a in axes),
        points=points,
    )
    index = tuple(np.searchsorted(axes[d], data[:, d]) for d in range(ndim))
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
        if key.startswith(_PREFIX):
            value = json.loads(raw)
            if key == f"{_PREFIX}units":
                units = value
            else:
                metadata[key[len(_PREFIX) :]] = value
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _number_or_text(raw: str) -> Any:
    try:
        return float(raw)
    except ValueError:
        return raw
