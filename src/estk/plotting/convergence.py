"""Plot of a barrier convergence scan."""

from __future__ import annotations

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes

from estk.plotting.core import INK_MUTED, INK_SECONDARY, SERIES, energy_scale


def plot_convergence(result: object, ax: Axes | None = None, *, units: str | None = None) -> Axes:
    """Barrier, reverse barrier and reaction free energy against time or bandwidth.

    The band around the barrier is the spread (standard deviation) of the runs' barriers.
    Bandwidth scans spanning a factor 8 or more use a logarithmic x axis.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    kbt = result.metadata.get("kbt")  # type: ignore[attr-defined]
    factor, unit_label = energy_scale(result.units, units, kbt)  # type: ignore[attr-defined]
    x = result.column("value")  # type: ignore[attr-defined]
    series = [
        ("barrier", r"$\Delta F^\ddagger$", SERIES[0]),
        ("reverse_barrier", "reverse", SERIES[1]),
        ("reaction_energy", r"$\Delta F$", SERIES[2]),
    ]
    spread = result.column("spread") * factor  # type: ignore[attr-defined]
    barrier = result.column("barrier") * factor  # type: ignore[attr-defined]
    if np.isfinite(spread).any():
        ax.fill_between(x, barrier - spread, barrier + spread, color=SERIES[0], alpha=0.15, lw=0,
                        label="spread between runs")  # fmt: skip
    for name, label, color in series:
        y = result.column(name) * factor  # type: ignore[attr-defined]
        ax.plot(x, y, color=color, marker="o", ms=2.0, label=label)
        label_text = ax.annotate(
            f"{y[-1]:.3g}", (x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
            va="center", fontsize="x-small", color=INK_SECONDARY, annotation_clip=False,
        )  # fmt: skip
        label_text.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="white")])
    ax.axhline(0.0, color=INK_MUTED, lw=0.4, zorder=0)
    log_x = result.parameter == "bandwidth" and x.min() > 0 and x.max() / x.min() >= 8  # type: ignore[attr-defined]
    if log_x:
        ax.set_xscale("log")
    # room for the end values inside the axes
    left, right = ax.get_xlim()
    ax.set_xlim(left, right * (right / left) ** 0.12 if log_x else right + 0.12 * (right - left))
    ax.set_xlabel(result.label)  # type: ignore[attr-defined]
    ax.set_ylabel(f"energy ({unit_label})")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncols=len(labels),
              borderaxespad=0.0, columnspacing=1.0, handlelength=1.2)  # fmt: skip
    return ax
