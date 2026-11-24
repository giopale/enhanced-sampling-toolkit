"""Bar chart of equilibrium averages, e.g. the occupation of each surface site."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator

from estk.errors import EstError
from estk.plotting.core import INK, INK_MUTED, INK_SECONDARY, SERIES, axis_label

GROUP_GAP = 0.6  # extra space between groups, in bar slots
SUM = "__sum__"  # the "group" of sums (totals): last, after a gap
SUM_COLOR = "#a3a19b"  # sums are neutral: not one of the groups


def plot_averages(
    result: object,
    ax: Axes | None = None,
    *,
    quantities: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    labels: Mapping[str, str] | None = None,
    ylabel: str = "average",
    runs: bool = True,
    sums: bool = True,
) -> Axes:
    """One bar per quantity: the combined average, with its error bar (standard error between
    runs or blocks) and, with ``runs`` and several runs, each run's own average as a dot.

    ``groups`` (``{type: [quantity, ...]}``, e.g. site types) colours the bars by group, in the
    fixed series order, and leaves a gap between groups; the legend names the groups.
    ``labels`` renames quantities on the x axis (default: ``cv.<name>.label`` style names).
    With ``sums``, the sums and means of the result (``ensemble_averages(sums=..., means=...)``,
    e.g. the fraction of all sites together) come last, after a gap, in grey.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    available = list(result.quantities)  # type: ignore[attr-defined]
    meta = result.metadata or {}  # type: ignore[attr-defined]
    summed = [*(meta.get("sums") or {}), *(meta.get("means") or {})]
    if not sums:
        available = [q for q in available if q not in summed]
    if groups:
        order = [q for members in groups.values() for q in members]
        missing = [q for q in order if q not in available]
        if missing:
            raise EstError(f"not in the averages: {', '.join(missing)}")
        if len(set(order)) != len(order):
            raise EstError("a quantity is in two groups")
        if len(groups) > len(SERIES):
            raise EstError(f"at most {len(SERIES)} groups (one colour each)")
        group_of = {q: g for g, members in groups.items() for q in members}
        if sums:
            order += [q for q in summed if q not in order]
            group_of.update({q: SUM for q in summed})
    else:
        if quantities:
            order = list(quantities) + ([q for q in summed if q not in quantities] if sums else [])
        else:
            order = [q for q in available if q not in summed] + (summed if sums else [])
        missing = [q for q in order if q not in available]
        if missing:
            raise EstError(f"not in the averages: {', '.join(missing)}")
        group_of = {q: SUM if q in summed else None for q in order}
    if not order:
        raise EstError("no quantities to plot")

    # x positions: one slot per bar, a gap where the group changes
    slots, position = [], 0.0
    for i, q in enumerate(order):
        if i and group_of[q] != group_of[order[i - 1]]:
            position += GROUP_GAP
        slots.append(position)
        position += 1.0
    x = np.array(slots)
    names = list(groups) if groups else []
    colors = [
        SUM_COLOR if group_of[q] == SUM else SERIES[names.index(group_of[q])] if groups
        else SERIES[0]
        for q in order
    ]  # fmt: skip
    values = np.array([result.values[q] for q in order])  # type: ignore[attr-defined]
    errors = np.array([result.errors[q] for q in order])  # type: ignore[attr-defined]

    ax.bar(x, values, width=0.72, color=colors, edgecolor="white", linewidth=1.0, zorder=2)
    finite = np.isfinite(errors)
    if finite.any():
        ax.errorbar(x[finite], values[finite], yerr=errors[finite], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)  # fmt: skip
    per_run = result.runs if runs else {}  # type: ignore[attr-defined]
    if len(per_run) > 1:
        offsets = np.linspace(-0.18, 0.18, len(per_run))
        for offset, run_values in zip(offsets, per_run.values(), strict=True):
            ys = np.array([run_values[q] for q in order])
            ax.plot(x + offset, ys, "o", ms=2.5, mfc="white", mec=INK_SECONDARY, mew=0.6,
                    zorder=3, label="_nolegend_")  # fmt: skip
    ax.axhline(0.0, color=INK_MUTED, lw=0.6, zorder=1)
    ax.set_xticks(x, [axis_label(q, labels) for q in order])
    ax.tick_params(axis="x", which="both", length=0, top=False)  # categories: no ticks
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)
    ax.set_ylabel(ylabel)
    if np.nanmin(values) >= 0:
        ax.set_ylim(bottom=0.0)
    if len(names) > 1:
        from matplotlib.patches import Patch

        handles = [Patch(facecolor=SERIES[k], label=g) for k, g in enumerate(names)]
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
                  ncols=min(4, len(handles)), borderaxespad=0.0, columnspacing=1.0,
                  handlelength=1.0)  # fmt: skip
    if len(per_run) > 1:
        ax.text(1.0, 1.01, "dots: each run", transform=ax.transAxes, ha="right", va="bottom",
                fontsize="x-small", color=INK_SECONDARY)  # fmt: skip
    return ax
