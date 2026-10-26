"""Plots of free-energy profiles along a minimum energy path."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from estk.plotting.core import INK, INK_SECONDARY, SERIES, energy_scale


def plot_path_profile(
    path: object,
    ax: Axes | None = None,
    *,
    units: str | None = None,
    runs: bool = True,
    annotate: bool = True,
) -> Axes:
    """Free energy along a ``MinimumEnergyPath`` against the path coordinate (0 to 1).

    The path's own profile is drawn in ink, with its error band if it has one; with ``runs``,
    each run's profile along the same path is drawn in a series colour and labelled. With
    ``annotate``, the barrier, reverse barrier and reaction free energy are written in the
    corner.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    kbt = path.metadata.get("kbt")  # type: ignore[attr-defined]
    factor, unit_label = energy_scale(path.units, units, kbt)  # type: ignore[attr-defined]
    s = path.s  # type: ignore[attr-defined]
    energies = path.energies * factor  # type: ignore[attr-defined]

    profiles = path.profiles if runs else {}  # type: ignore[attr-defined]
    for k, (name, values) in enumerate(profiles.items()):
        color = SERIES[k] if k < len(SERIES) else INK_SECONDARY
        ax.plot(s, values * factor, color=color, lw=0.6, label=name)
    error = path.error  # type: ignore[attr-defined]
    if error is not None:
        ax.fill_between(
            s, energies - error * factor, energies + error * factor, color=INK, alpha=0.12, lw=0,
            label="spread between runs",
        )  # fmt: skip
    combined = path.metadata.get("combined")  # type: ignore[attr-defined]
    label = "all runs" if combined else "path"
    ax.plot(s, energies, color=INK, lw=1.0, marker="o", ms=2.0, label=label, zorder=5)
    top = int(np.nanargmax(energies))
    ax.plot(s[top], energies[top], "X", color=INK, ms=4, zorder=6)

    ax.set_xlabel("path coordinate")
    ax.set_ylabel(f"F ({unit_label})")
    ax.set_xlim(0, 1)
    if profiles:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.01),
            ncols=min(4, len(labels)), borderaxespad=0.0, columnspacing=1.0, handlelength=1.2,
        )  # fmt: skip
        if len(profiles) <= 4:
            ends = {name: float(v[-1]) * factor for name, v in profiles.items()}
            _direct_labels(ax, ends)
    if annotate:
        b = path.barrier * factor  # type: ignore[attr-defined]
        r = path.reverse_barrier * factor  # type: ignore[attr-defined]
        d = path.reaction_energy * factor  # type: ignore[attr-defined]
        text = (
            f"$\\Delta F^\\ddagger$ = {b:.3g} {unit_label}\n"
            f"reverse = {r:.3g} {unit_label}\n"
            f"$\\Delta F$ = {d:.3g} {unit_label}"
        )
        ax.text(0.03, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                fontsize="small", color=INK)  # fmt: skip
    return ax


def _direct_labels(ax: Axes, ends: dict[str, float]) -> None:
    """Label line ends just right of the axes, spread apart so that they don't overlap."""
    fig = ax.figure
    fig.canvas.draw()
    to_pixels = ax.transData.transform
    gap = 1.15 * plt_fontsize_points(ax) * fig.dpi / 72
    placed: list[tuple[str, float]] = []
    for name, y in sorted(ends.items(), key=lambda item: item[1]):
        pixel = float(to_pixels((1.0, y))[1])
        if placed and pixel - placed[-1][1] < gap:
            pixel = placed[-1][1] + gap
        placed.append((name, pixel))
    inverse = ax.transData.inverted()
    for name, pixel in placed:
        x_data, y_data = inverse.transform((to_pixels((1.0, 0.0))[0], pixel))
        ax.annotate(
            name, (1.0, y_data), xytext=(3, 0), textcoords="offset points", va="center",
            fontsize="x-small", color=INK_SECONDARY, annotation_clip=False,
            xycoords=("axes fraction", "data"),
        )  # fmt: skip
        del x_data


def plt_fontsize_points(ax: Axes) -> float:
    from matplotlib.font_manager import FontProperties

    return FontProperties(size="x-small").get_size_in_points()
