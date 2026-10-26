"""Plots of free-energy surfaces and property averages."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from estk.errors import EstError
from estk.plotting.core import (
    FREE_ENERGY_CMAP,
    INK,
    PATH_COLOR,
    PROPERTY_CMAP,
    SERIES,
    axis_label,
    energy_scale,
)
from estk.surface import Surface

DEFAULT_MAX_KT = 20.0  # default colour range of free-energy maps, in kT


def plot_surface(
    surface: Surface,
    ax: Axes | None = None,
    *,
    path: object | None = None,
    units: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_step: float | None = None,
    levels: Sequence[float] | None = None,
    contour_labels: str = "some",
    labels: Mapping[str, str] | None = None,
    colorbar: bool = True,
    title: str | None = None,
) -> Axes:
    """Draw one surface on ``ax`` (a new figure if None) and return the axes.

    2D: filled map (free energy: light = low, dark = high; above ``vmax`` in the darkest colour;
    no data left blank) in discrete colour bands between black contour lines every
    ``contour_step`` (default: a round step giving about 8 lines; 0 for a smooth scale without
    lines), or at the given ``levels`` (which then also set the colour range), and, if given, a
    ``MinimumEnergyPath`` on top. 1D: the profile, with the error
    band if the surface has one.

    ``contour_labels``: write the value on ``"all"`` contour lines, on every other one
    (``"some"``, default) or on ``"none"``.

    ``units`` converts free energies for display (``"kJ/mol"``, ``"eV"``, ``"kT"``, ...).
    ``vmax`` defaults to 20 kT (free energies) or the data range (properties).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    kbt = surface.metadata.get("kbt")
    factor, unit_label = (
        (1.0, "") if not surface.is_fes else energy_scale(surface.units, units, kbt)
    )
    values = surface.values * factor
    error = surface.error * factor if surface.error is not None else None
    finite = np.isfinite(values)
    if not finite.any():
        raise EstError("the surface has no finite values to plot")
    value_label = _value_label(surface, unit_label)

    if surface.grid.ndim == 1:
        _plot_profile_1d(ax, surface, values, error, labels, value_label)
    elif surface.grid.ndim == 2:
        if levels is not None:
            levels = sorted(float(v) for v in levels)
            if len(levels) < 2:
                raise EstError("give at least two contour levels")
            lo, hi = levels[0], levels[-1]
        else:
            lo, hi = _color_range(surface, values, finite, vmin, vmax, kbt, factor)
        spacing = levels if levels is not None else contour_step
        _plot_map_2d(ax, surface, values, lo, hi, spacing, contour_labels, labels)
        if colorbar:
            _colorbar(ax.figure, ax.collections[0], ax, value_label)
        if path is not None:
            draw_path(ax, path, surface.grid.names)
    else:
        raise EstError(f"can't plot a {surface.grid.ndim}D surface; reweight on 1 or 2 CVs")
    if title:
        ax.set_title(title)
    return ax


def plot_surfaces(
    surfaces: Mapping[str, Surface],
    *,
    ncols: int = 2,
    path: object | None = None,
    units: str | None = None,
    vmax: float | None = None,
    contour_step: float | None = None,
    levels: Sequence[float] | None = None,
    contour_labels: str = "some",
    labels: Mapping[str, str] | None = None,
) -> Figure:
    """Small multiples of several surfaces on the same grid, with one shared colour scale.

    ``surfaces`` maps a title (e.g. the run name) to its surface. ``path`` is drawn on every
    panel.
    """
    import matplotlib.pyplot as plt

    items = list(surfaces.items())
    if not items:
        raise EstError("no surfaces to plot")
    first = items[0][1]
    ncols = max(1, min(ncols, len(items)))
    nrows = math.ceil(len(items) / ncols)
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width * ncols / 1.6, height * nrows / 1.2), squeeze=False,
        sharex=True, sharey=True, layout="constrained",
    )  # fmt: skip
    if first.grid.ndim == 2 and first.is_fes and vmax is None:
        kbt = first.metadata.get("kbt")
        factor, _ = energy_scale(first.units, units, kbt)
        vmax = DEFAULT_MAX_KT * kbt * factor if kbt else None
    for k, (name, surface) in enumerate(items):
        ax = axes[k // ncols][k % ncols]
        plot_surface(
            surface, ax, path=path, units=units, vmax=vmax, contour_step=contour_step,
            levels=levels, contour_labels=contour_labels, labels=labels, colorbar=False,
            title=name,
        )  # fmt: skip
        if k % ncols:
            ax.set_ylabel("")
        if k // ncols < nrows - 1:
            ax.set_xlabel("")
    for k in range(len(items), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if first.grid.ndim == 2:
        _, unit_label = (
            (1.0, "")
            if not first.is_fes
            else energy_scale(first.units, units, first.metadata.get("kbt"))
        )
        _colorbar(fig, axes[0][0].collections[0], axes, _value_label(first, unit_label))
    return fig


def _colorbar(fig: Figure, mappable: object, ax: object, label: str) -> None:
    bar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.05)
    bar.set_label(label)
    levels = np.asarray(getattr(mappable, "levels", []))
    if 1 < len(levels) <= 11:  # discrete bands: a tick on every contour level
        ticks = levels
    else:
        lo, hi = mappable.get_clim()  # type: ignore[attr-defined]
        ticks = MaxNLocator(5).tick_values(lo, hi)
        span = hi - lo
        ticks = ticks[(ticks >= lo - 1e-9 * span) & (ticks <= hi + 1e-9 * span)]
    bar.set_ticks(ticks)
    bar.ax.minorticks_off()
    bar.ax.tick_params(direction="out", right=True, left=False)
    bar.outline.set_linewidth(0.5)


def draw_path(ax: Axes, path: object, names: Sequence[str]) -> None:
    """Draw a ``MinimumEnergyPath`` on a 2D map: line, start (open), end (filled), saddle (x)."""
    path_names = tuple(getattr(path, "names", ()))
    if path_names != tuple(names):
        raise EstError(f"the path is in {path_names}, the surface in {tuple(names)}")
    points = path.points  # type: ignore[attr-defined]
    ax.plot(points[:, 0], points[:, 1], color=PATH_COLOR, lw=1.0, zorder=5, solid_capstyle="round")
    halo = {"markeredgecolor": "white", "markeredgewidth": 0.6, "zorder": 6}
    ax.plot(*points[0], "o", color=PATH_COLOR, markerfacecolor="white", ms=3.5, zorder=6)
    ax.plot(*points[-1], "o", color=PATH_COLOR, ms=3.5, **halo)
    saddle = points[path.saddle_index]  # type: ignore[attr-defined]
    ax.plot(*saddle, "X", color=INK, ms=4, **halo)


def _value_label(surface: Surface, unit_label: str) -> str:
    if surface.is_fes:
        return f"F ({unit_label})" if unit_label else "F"
    return surface.quantity


def _color_range(
    surface: Surface,
    values: np.ndarray,
    finite: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    kbt: float | None,
    factor: float,
) -> tuple[float, float]:
    data_lo, data_hi = float(values[finite].min()), float(values[finite].max())
    if surface.is_fes:
        lo = 0.0 if vmin is None else vmin
        hi = vmax if vmax is not None else (
            min(data_hi, DEFAULT_MAX_KT * kbt * factor) if kbt else data_hi
        )  # fmt: skip
    else:
        lo = data_lo if vmin is None else vmin
        hi = data_hi if vmax is None else vmax
    if not hi > lo:
        hi = lo + 1.0
    return lo, hi


def _plot_map_2d(
    ax: Axes,
    surface: Surface,
    values: np.ndarray,
    lo: float,
    hi: float,
    contour_step: float | Sequence[float] | None,
    contour_labels: str,
    labels: Mapping[str, str] | None,
) -> None:
    X, Y = surface.grid.mesh()
    masked = np.ma.masked_invalid(values)
    cmap = FREE_ENERGY_CMAP if surface.is_fes else PROPERTY_CMAP
    # One colour band between consecutive contour lines.
    if contour_step is not None and not np.isscalar(contour_step):
        lines = np.asarray(contour_step, dtype=float)
    elif contour_step is None:
        lines = MaxNLocator(8).tick_values(lo, hi)
        lines = lines[(lines >= lo - 1e-12) & (lines <= hi + 1e-12)]
    elif contour_step > 0:
        lines = np.arange(lo, hi + contour_step / 2, contour_step)
    else:
        lines = np.array([])
    extend = "max" if surface.is_fes else "both"
    if len(lines) > 1:
        ax.contourf(X, Y, masked, levels=lines, cmap=cmap, extend=extend)
        contours = ax.contour(X, Y, masked, levels=lines, colors=INK, linewidths=0.4)
        _label_contours(ax, contours, lines, contour_labels)
    else:  # no contour lines: a smooth colour scale
        ax.contourf(X, Y, masked, levels=np.linspace(lo, hi, 65), cmap=cmap, extend=extend)
    names = surface.grid.names
    ax.set_xlabel(axis_label(names[0], labels))
    ax.set_ylabel(axis_label(names[1], labels))
    ax.set_xlim(surface.grid.lower[0], surface.grid.upper[0])
    ax.set_ylim(surface.grid.lower[1], surface.grid.upper[1])


def _plot_profile_1d(
    ax: Axes,
    surface: Surface,
    values: np.ndarray,
    error: np.ndarray | None,
    labels: Mapping[str, str] | None,
    value_label: str,
) -> None:
    (x,) = surface.grid.axes()
    color = INK if surface.metadata.get("combined") else SERIES[0]
    if error is not None:
        ax.fill_between(x, values - error, values + error, color=color, alpha=0.15, lw=0)
    ax.plot(x, values, color=color)
    ax.set_xlabel(axis_label(surface.grid.names[0], labels))
    ax.set_ylabel(value_label)
    ax.set_xlim(x[0], x[-1])


def _label_contours(ax: Axes, contours: object, lines: np.ndarray, which: str) -> None:
    """Write the level on the contour lines: all, every other one (some) or none."""
    if which not in ("all", "some", "none"):
        raise EstError(f"contour labels: choose all, some or none, not '{which}'")
    if which == "none":
        return
    chosen = lines[1:] if which == "all" else lines[2::2]  # the lowest level is just a point
    if len(chosen) == 0:
        return
    decimals = max(0, -int(np.floor(np.log10(np.min(np.diff(lines)))))) if len(lines) > 1 else 2
    texts = ax.clabel(
        contours, levels=chosen, inline=True, inline_spacing=2, fontsize="xx-small",
        fmt=lambda v: f"{v:.{decimals}f}",
    )  # fmt: skip
    # A thin white outline keeps the numbers readable on dark bands and over a path.
    halo = [patheffects.withStroke(linewidth=1.2, foreground="white")]
    for text in texts:
        text.set_path_effects(halo)
        text.set_zorder(7)
