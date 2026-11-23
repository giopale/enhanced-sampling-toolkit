"""Map of the basins of a free-energy surface, labelled with their populations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from estk.errors import EstError
from estk.plotting.core import INK
from estk.plotting.surface import plot_surface
from estk.surface import Surface


def plot_basins(
    result: object,
    surface: Surface,
    ax: Axes | None = None,
    *,
    path: object | None = None,
    units: str | None = None,
    labels: Mapping[str, str] | None = None,
    cmap: str | Colormap | None = None,
    boundary_color: str = "white",
) -> Axes:
    """The free-energy surface with each basin's boundary and its name and population.

    ``result`` is a ``BasinResult`` and ``surface`` the surface it was computed on (same grid).
    2D: the boundaries are drawn as lines; 1D: as vertical lines. Each basin is labelled at its
    lowest point.
    """
    grid = result.grid  # type: ignore[attr-defined]
    if surface.grid.shape != grid.shape or surface.grid.names != grid.names:
        raise EstError("the surface is not on the grid of the basins")
    ax = plot_surface(surface, ax, path=path, units=units, labels=labels, cmap=cmap,
                      contour_labels="none")  # fmt: skip
    basin_labels = result.labels  # type: ignore[attr-defined]
    basins = result.basins  # type: ignore[attr-defined]
    if grid.ndim == 2:
        X, Y = grid.mesh()
        for k in range(len(basins)):
            inside = (basin_labels == k).astype(float)
            if 0 < inside.sum() < inside.size:
                ax.contour(X, Y, inside, levels=[0.5], colors=boundary_color, linewidths=1.0)
    elif grid.ndim == 1:
        x = grid.axes()[0]
        changes = np.nonzero(np.diff(basin_labels) != 0)[0]
        for i in changes:
            ax.axvline(0.5 * (x[i] + x[i + 1]), color=INK, lw=0.6, ls="--")
    outline = [patheffects.withStroke(linewidth=2.0, foreground="white")]
    for basin in basins:
        if not np.all(np.isfinite(basin.point)):
            continue
        text = f"{basin.name}\n{100 * basin.population:.3g}%"
        if grid.ndim == 2:
            xy = basin.point
        else:
            y = ax.get_ylim()
            xy = (basin.point[0], y[0] + 0.85 * (y[1] - y[0]))
        if grid.ndim == 2:
            ax.plot(*xy, "o", color=INK, ms=2.5)
        label = ax.annotate(text, xy, xytext=(4, 4), textcoords="offset points",
                            fontsize="small", color=INK, ha="left", va="bottom")  # fmt: skip
        label.set_path_effects(outline)
    return ax
