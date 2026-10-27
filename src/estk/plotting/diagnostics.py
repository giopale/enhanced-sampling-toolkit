"""Plots of time series: OPES quantities and model deviation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure

from estk.errors import EstError
from estk.plotting.core import INK_MUTED, INK_SECONDARY, SERIES

MAX_POINTS = 4000  # per line; longer series are thinned for drawing (not for the statistics)
MEDIAN_BINS = 100  # time bins of the median line of the model deviation
SKIP = 0.05  # start of each run left out of the y range of linear panels when it is a transient


def _thin(time: np.ndarray, values: np.ndarray, parts: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Every n-th point, so that ``parts`` lines of this length have MAX_POINTS in all."""
    step = max(1, math.ceil(len(time) * parts / MAX_POINTS))
    return time[::step], values[::step]


def _pieces(ts: object, column: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(time, values)`` of each walker of a ``TimeSeries`` (one piece without walkers)."""
    segments = ts.segments()  # type: ignore[attr-defined]
    return [
        _thin(ts.time[rows], ts.values[column][rows], len(segments))  # type: ignore[attr-defined]
        for _, rows in segments
    ]


def _bulk_limits(series: Sequence[object], column: str) -> tuple[float, float] | None:
    """Y limits that leave out the start of each run (first ``SKIP`` of its time) when that
    start-up transient (e.g. of OPES) would otherwise squash the rest; None to keep autoscaling."""
    full, bulk = [], []
    for ts in series:
        values = ts.values.get(column)  # type: ignore[attr-defined]
        if values is None or len(values) < 20:
            continue
        t = ts.time  # type: ignore[attr-defined]
        full.append(values)
        bulk.append(values[t >= t.min() + SKIP * (t.max() - t.min())])
    if not bulk:
        return None
    full_values, bulk_values = np.concatenate(full), np.concatenate(bulk)
    lo, hi = np.nanmin(bulk_values), np.nanmax(bulk_values)
    if not hi > lo or np.nanmax(full_values) - np.nanmin(full_values) < 2 * (hi - lo):
        return None
    pad = 0.05 * (hi - lo)
    return float(lo - pad), float(hi + pad)


def _binned_median(
    time: np.ndarray, values: np.ndarray, bins: int = MEDIAN_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """Median of ``values`` in equal bins of time (all walkers together)."""
    edges = np.linspace(time.min(), time.max(), bins + 1)
    index = np.clip(np.searchsorted(edges, time, side="right") - 1, 0, bins - 1)
    centres = (edges[:-1] + edges[1:]) / 2
    medians = np.array([np.median(values[index == b]) if np.any(index == b) else np.nan
                        for b in range(bins)])  # fmt: skip
    return centres, medians


def _color(k: int) -> str:
    return SERIES[k] if k < len(SERIES) else INK_SECONDARY


def plot_timeseries(
    series: Mapping[str, object],
    columns: Sequence[str],
    *,
    labels: Mapping[str, str] | None = None,
    log_time: bool = False,
    log_columns: Sequence[str] = (),
) -> Figure:
    """One panel per column (shared time axis), one line per run in a fixed colour.

    ``series`` maps run names to ``TimeSeries``.
    """
    import matplotlib.pyplot as plt

    if not series:
        raise EstError("no runs to plot")
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(
        len(columns), 1, sharex=True, squeeze=False, layout="constrained",
        figsize=(width, max(height, 0.9 * height * len(columns))),
    )  # fmt: skip
    for row, column in enumerate(columns):
        ax = axes[row][0]
        for k, (name, ts) in enumerate(series.items()):
            if column not in ts.values:  # type: ignore[attr-defined]
                continue
            for i, (t, v) in enumerate(_pieces(ts, column)):  # one line per walker
                ax.plot(t, v, color=_color(k), lw=0.5, label=name if i == 0 else "_nolegend_")
        ax.set_ylabel((labels or {}).get(column, column))
        if column in log_columns:
            ax.set_yscale("log")
        else:
            limits = _bulk_limits(list(series.values()), column)
            if limits is not None:
                ax.set_ylim(*limits)
    axes[-1][0].set_xlabel("time (ns)")
    if log_time:
        axes[-1][0].set_xscale("log")
    handles, names = axes[0][0].get_legend_handles_labels()
    if len(names) > 1:
        axes[0][0].legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                          ncols=min(len(names), 6), borderaxespad=0.0,
                          handlelength=1.2)  # fmt: skip
    return fig


def plot_model_deviation(
    series: Mapping[str, object],
    column: str,
    *,
    low: float | None = None,
    high: float | None = None,
    bins: int = 60,
) -> Figure:
    """Model deviation against time (log scale) and its distribution, with the trust levels."""
    import matplotlib.pyplot as plt

    if not series:
        raise EstError("no runs to plot")
    fig, (ax, hist) = plt.subplots(
        1, 2, sharey=True, layout="constrained", gridspec_kw={"width_ratios": [4, 1], "wspace": 0}
    )
    every = np.concatenate([ts.values[column] for ts in series.values()])  # type: ignore[attr-defined]
    positive = every[every > 0]
    if positive.size == 0:
        raise EstError(f"{column}: no positive values to plot on a log scale")
    edges = np.geomspace(positive.min(), positive.max(), bins + 1)
    for k, (name, ts) in enumerate(series.items()):
        values = ts.values[column]  # type: ignore[attr-defined]
        # raw values faintly (each walker), the median over time bins (all walkers) on top,
        # so that several runs stay readable
        for t, v in _pieces(ts, column):
            ax.plot(t, v, color=_color(k), lw=0.3, alpha=0.25, zorder=1)
        ax.plot(*_binned_median(ts.time, values), color=_color(k), lw=1.0, label=name,
                zorder=3)  # type: ignore[attr-defined]  # fmt: skip
        counts, _ = np.histogram(values, bins=edges)
        hist.stairs(counts / max(counts.sum(), 1), edges, orientation="horizontal",
                    color=_color(k), lw=0.6)  # fmt: skip
    ax.set_yscale("log")
    for level in (low, high):
        if level is not None:
            for a in (ax, hist):
                a.axhline(level, color=INK_MUTED, lw=0.6, ls="--", zorder=0)
            hist.annotate(f"{level:g}", (1.0, level), xycoords=("axes fraction", "data"),
                          xytext=(2, 0), textcoords="offset points", va="center",
                          fontsize="x-small", color=INK_SECONDARY,
                          annotation_clip=False)  # fmt: skip
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(column)
    hist.set_xlabel("fraction")
    hist.tick_params(axis="y", left=False, labelleft=False)
    handles, names = ax.get_legend_handles_labels()
    if len(names) > 1:
        ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.02),
                  ncols=min(len(names), 6), borderaxespad=0.0, handlelength=1.2)  # fmt: skip
    return fig
