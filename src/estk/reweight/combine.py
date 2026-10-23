"""Combining the surfaces of several runs into one.

Modes
-----
``probability`` (default)
    Sum the runs' densities, each with its total weight ``sum_i exp(V_i / kT)``. This is the
    same as reweighting all frames of all runs together. Kernels are normalised per run, so runs
    with different bandwidths are combined correctly.
``equal``
    Average the runs' normalised densities: every run counts the same.
``legacy``
    What the original scripts did, to compare with old ``-AVG.dat`` files: average the per-run
    free energies (each shifted to minimum 0) with run weights ``sum_i exp((V_i - V_max) / kT)``,
    where ``V_max`` is each run's own largest bias. Those weights are not on a common scale
    between runs. The error column is the unweighted standard deviation between runs.

For ``probability`` and ``equal``, the error column is the spread of the runs around the
combined value, weighted like the combination (free energies: the spread of the per-run
free energies; properties: of the per-run averages, weighted by each run's density there).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from estk.errors import EstError
from estk.surface import Surface

MODES = ("probability", "equal", "legacy")


def combine(
    surfaces: Mapping[str, Surface] | Sequence[Surface], mode: str = "probability"
) -> Surface:
    """Combine per-run surfaces on the same grid into one surface; see the module docstring."""
    if mode not in MODES:
        raise EstError(f"unknown combination mode '{mode}' (choose from {', '.join(MODES)})")
    if isinstance(surfaces, Mapping):
        named = dict(surfaces)
    else:
        named = {str(s.metadata.get("run", i)): s for i, s in enumerate(surfaces)}
    if not named:
        raise EstError("no surfaces to combine")
    runs = list(named)
    items = list(named.values())
    first = items[0]
    _check_compatible(named)

    kbt = float(first.metadata["kbt"])
    log_totals = np.array([_meta(s, "log_weight_total") for s in items])

    if mode == "legacy":
        return _legacy(runs, items, log_totals)

    # Log of each run's normalised density on the grid (kernels normalised to integrate to 1).
    log_density = np.array([_log_density(s, kbt) for s in items])  # (runs, *grid)
    run_log_weight = log_totals if mode == "probability" else np.zeros(len(items))
    run_log_weight = run_log_weight - _logsumexp(run_log_weight)  # shares sum to 1
    shares = np.exp(run_log_weight)

    expand = (slice(None),) + (None,) * first.grid.ndim
    terms = run_log_weight[expand] + log_density  # log of each run's contribution
    log_total = _logsumexp_axis0(terms)

    meta: dict[str, Any] = {
        "combined": mode,
        "runs": runs,
        "run_weights": {r: float(w) for r, w in zip(runs, shares, strict=True)},
        "kbt": kbt,
        "frames": int(sum(s.metadata.get("frames", 0) for s in items)),
    }
    for key in ("temperature", "energy_units", "time_window_ns", "walls_in_bias"):
        if key in first.metadata:
            meta[key] = first.metadata[key]

    if first.is_fes:
        free_energy = -kbt * log_total
        offset = _finite_min(free_energy)
        per_run = -kbt * log_density
        error = _weighted_spread(
            per_run, free_energy, np.broadcast_to(shares[expand], per_run.shape)
        )
        meta["fes_offset"] = offset
        return Surface(
            grid=first.grid,
            values=free_energy - offset,
            units=first.units,
            error=error,
            metadata=meta,
        )

    # Property: average of the run averages, weighted by each run's contribution to the density.
    values = np.array([s.values for s in items])
    local = np.exp(terms - log_total[None])  # each run's share of the density at each point
    local = np.where(np.isfinite(values) & np.isfinite(local), local, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = local.sum(axis=0)
        average = np.where(norm > 0, (local * np.nan_to_num(values)).sum(axis=0) / norm, np.nan)
    error = _weighted_spread(values, average, local)
    free_energy = -kbt * log_total
    offset = _finite_min(free_energy)
    meta["fes_offset"] = offset
    return Surface(
        grid=first.grid,
        values=average,
        quantity=first.quantity,
        units=first.units,
        error=error,
        metadata=meta,
        fes=free_energy - offset,
    )


def _legacy(runs: list[str], items: list[Surface], log_totals: np.ndarray) -> Surface:
    first = items[0]
    old_log_totals = log_totals - np.array([_meta(s, "log_weight_max") for s in items])
    weights = np.exp(old_log_totals) / np.sum(np.exp(old_log_totals))
    stack = np.array([s.values for s in items])  # free energies: each shifted to minimum 0
    with np.errstate(invalid="ignore"):
        average = np.average(stack, axis=0, weights=weights)
        spread = np.std(stack, axis=0)
    meta: dict[str, Any] = {
        "combined": "legacy",
        "runs": runs,
        "run_weights": {r: float(w) for r, w in zip(runs, weights, strict=True)},
        "kbt": float(first.metadata["kbt"]),
        "frames": int(sum(s.metadata.get("frames", 0) for s in items)),
        "fes_offset": 0.0,
    }
    for key in ("temperature", "energy_units", "time_window_ns"):
        if key in first.metadata:
            meta[key] = first.metadata[key]
    return Surface(
        grid=first.grid,
        values=average,
        quantity=first.quantity,
        units=first.units,
        error=spread,
        metadata=meta,
    )


def _log_density(surface: Surface, kbt: float) -> np.ndarray:
    """log of the run's density, normalised: weights sum to 1 and kernels integrate to 1."""
    fes = surface.values if surface.is_fes else surface.fes
    if fes is None:
        raise EstError(
            f"run {surface.metadata.get('run', '?')}: the {surface.quantity} surface has no "
            "free-energy column, needed to weight the runs"
        )
    offset = _meta(surface, "fes_offset")
    bandwidths = surface.metadata.get("bandwidth") or {}
    if set(bandwidths) != set(surface.grid.names):
        raise EstError(f"run {surface.metadata.get('run', '?')}: bandwidths missing in metadata")
    log_norm = 0.5 * surface.grid.ndim * math.log(2 * math.pi) + sum(
        math.log(bandwidths[n]) for n in surface.grid.names
    )
    return -(fes + offset) / kbt - log_norm


def _check_compatible(named: Mapping[str, Surface]) -> None:
    items = list(named.items())
    name0, first = items[0]
    kbt0 = first.metadata.get("kbt")
    if kbt0 is None:
        raise EstError(f"run {name0}: kT missing in metadata (not a surface made by estk?)")
    for name, surface in items[1:]:
        if surface.grid != first.grid:
            raise EstError(f"runs {name0} and {name} are on different grids")
        if surface.quantity != first.quantity:
            raise EstError(
                f"runs {name0} ({first.quantity}) and {name} ({surface.quantity}) differ"
            )
        kbt = surface.metadata.get("kbt")
        if kbt is None or not math.isclose(kbt, kbt0, rel_tol=1e-9):
            raise EstError(f"runs {name0} and {name} have different temperatures (kT)")
        if surface.units != first.units:
            raise EstError(f"runs {name0} and {name} have different units")


def _meta(surface: Surface, key: str) -> float:
    try:
        return float(surface.metadata[key])
    except KeyError:
        raise EstError(
            f"run {surface.metadata.get('run', '?')}: '{key}' missing in metadata "
            "(reweight again with this version of estk)"
        ) from None


def _logsumexp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.sum(np.exp(values - peak))))


def _logsumexp_axis0(values: np.ndarray) -> np.ndarray:
    peak = np.max(values, axis=0)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        return safe + np.log(np.sum(np.exp(values - safe[None]), axis=0))


def _weighted_spread(per_run: np.ndarray, center: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """sqrt(sum_r w_r (x_r - center)^2 / sum_r w_r) over runs with finite values."""
    usable = np.isfinite(per_run) & np.isfinite(center)[None] & (weights > 0)
    w = np.where(usable, weights, 0.0)
    diff = np.where(usable, per_run - np.where(np.isfinite(center), center, 0.0)[None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt((w * diff**2).sum(axis=0) / w.sum(axis=0))
    spread[usable.sum(axis=0) < 2] = np.nan
    return spread


def _finite_min(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        raise EstError("the combined surface has no finite values")
    return float(values[finite].min())
