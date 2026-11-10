"""Kernel bandwidths."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from estk.errors import EstError
from estk.logging import get_logger

log = get_logger(__name__)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size, (sum w)^2 / sum w^2."""
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights) ** 2 / np.sum(weights**2))


def silverman(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    ndim: int = 1,
    bias_factor: float | None = None,
    weighted_std: bool = False,
) -> float:
    """Silverman's rule for one CV of a ``ndim``-dimensional KDE.

    ``h = sigma / sqrt(bias_factor) * (n_eff * (ndim + 2) / 4) ** (-1 / (ndim + 4))``

    ``n_eff`` is the effective sample size of ``weights`` (the number of frames without them).
    ``sigma`` is the standard deviation of ``values``: unweighted by default, as in the original
    scripts (the biased sampling is broader than the unbiased distribution, and the bias factor
    corrects for it), or weighted with ``weighted_std=True``.
    """
    values = np.asarray(values, dtype=float)
    if weights is None:
        n_eff = float(values.size)
        sigma = float(np.std(values))
    else:
        n_eff = effective_sample_size(weights)
        if weighted_std:
            mean = np.average(values, weights=weights)
            sigma = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        else:
            sigma = float(np.std(values))
    gamma = 1.0 if bias_factor is None or not np.isfinite(bias_factor) else bias_factor
    return sigma / np.sqrt(gamma) * (n_eff * (ndim + 2) / 4.0) ** (-1.0 / (ndim + 4.0))


# --- smoothed cross-validation -----------------------------------------------------------------

SCV_METHODS = ("binned", "exact")
_BINS = {1: 2000, 2: 400, 3: 100}  # grid points per CV for the binned evaluation
_H_BOUNDS = (1e-3, 3.0)  # in units of each CV's standard deviation


@dataclass
class ScvResult:
    """Bandwidths minimising the smoothed cross-validation criterion, per CV."""

    bandwidth: np.ndarray  # (d,), CV units
    pilot: np.ndarray  # (d,), CV units
    objective: float
    converged: bool
    effective_frames: float
    weighted: bool
    method: str


def scv(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    weighted: bool = True,
    pilot: Sequence[float] | None = None,
    method: str = "binned",
    bins: int | None = None,
) -> ScvResult:
    """Diagonal bandwidth matrix by smoothed cross-validation (SCV) with Gaussian kernels.

    ``values`` has one row per frame and one column per CV. The criterion is that of Duong &
    Hazelton (2005), with the frame weights ``w`` (normalised to sum 1) in place of ``1/n``::

        SCV(H) = sum(w^2) |H|^{-1/2} (4 pi)^{-d/2}
                 + sum_ij w_i w_j [phi_{2H+2G} - 2 phi_{H+2G} + phi_{2G}](X_i - X_j)

    so the bandwidth suits the reweighted (unbiased) density that the weighted KDE estimates;
    no bias-factor correction is needed. ``weighted=False`` treats the frames equally (the
    bandwidth of the biased density, as the original scripts did).

    Which to use for free energies: SCV minimises the integrated squared error of the
    *density*, to which the barrier regions (almost no probability) contribute nothing. With
    the few effective frames of a biased run, the weighted criterion therefore picks wide
    kernels that fill the barriers in. The biased frames cover the barrier regions, so the
    unweighted criterion keeps their detail; ``estk`` uses it for ``--bandwidth scv``.

    ``G`` is the pilot: by default the normal-scale value
    ``g_k = sigma_k (2 / (d + 4))^{1/(d+6)} n_eff^{-1/(d+6)}``; ``pilot`` sets it in CV units.

    ``method="binned"`` (default) bins the weighted frames on a grid (linear binning) and sums
    over grid offsets, using the autocorrelation of the bin weights computed once by FFT: each
    evaluation then costs the same for any number of frames. ``method="exact"`` sums over all
    pairs of frames (O(n^2) per evaluation; for checks on small samples).
    """
    if method not in SCV_METHODS:
        raise EstError(f"unknown SCV method '{method}' (choose from {', '.join(SCV_METHODS)})")
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n, d = x.shape
    if n < 10:
        raise EstError(f"SCV needs more frames ({n} given)")
    if weights is None or not weighted:
        w = np.full(n, 1.0 / n)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise EstError(f"{w.size} weights for {n} frames")
        w = w / w.sum()
    n_eff = 1.0 / float(np.sum(w**2))

    # Standardise each CV (weighted spread) so that one set of bounds and steps fits all.
    mean = w @ x
    sigma = np.sqrt(w @ (x - mean) ** 2)
    if np.any(sigma <= 1e-12 * np.maximum(1.0, np.abs(mean))):
        raise EstError("a CV has no spread; SCV needs varying values")
    z = (x - mean) / sigma

    if pilot is None:
        g = np.full(d, (2.0 / (d + 4)) ** (1.0 / (d + 6)) * n_eff ** (-1.0 / (d + 6)))
    else:
        g = np.asarray(pilot, dtype=float) / sigma
        if g.shape != (d,) or np.any(g <= 0):
            raise EstError(f"pilot needs {d} positive values")
    g2 = g**2

    if method == "binned":
        pair_sum = _binned_pair_sum(z, w, bins or _BINS.get(d, 40))
    else:
        pair_sum = _exact_pair_sum(z, w)
    constant = pair_sum(2 * g2)  # phi_{2G} term: does not depend on H
    norm = (4 * np.pi) ** (-d / 2) * float(np.sum(w**2))

    def objective(u: np.ndarray) -> float:
        h2 = np.exp(2 * u)
        return (
            norm * float(np.prod(h2)) ** -0.5
            + pair_sum(2 * h2 + 2 * g2)
            - 2 * pair_sum(h2 + 2 * g2)
            + constant
        )

    # start from the normal-scale (Silverman) bandwidth
    start = np.full(d, np.log((4.0 / (d + 2)) ** (1 / (d + 4)) * n_eff ** (-1 / (d + 4))))
    bounds = [tuple(np.log(_H_BOUNDS))] * d
    scale = objective(start)  # > 0 (an estimate of the integrated squared error)
    result = minimize(
        lambda u: objective(u) / scale,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8},
    )
    h = np.exp(result.x)
    at_bound = np.isclose(result.x, np.log(_H_BOUNDS[0])) | np.isclose(
        result.x, np.log(_H_BOUNDS[1])
    )
    if np.any(at_bound):
        log.warning("SCV bandwidth at the search bound (%s sigma)", np.round(h, 4).tolist())
    return ScvResult(
        bandwidth=h * sigma,
        pilot=g * sigma,
        objective=float(result.fun) * scale,
        converged=bool(result.success) and not np.any(at_bound),
        effective_frames=n_eff,
        weighted=weighted and weights is not None,
        method=method,
    )


def _gaussian(x: np.ndarray, var: float) -> np.ndarray:
    return np.exp(-0.5 * x * x / var) / np.sqrt(2 * np.pi * var)


def _binned_pair_sum(z: np.ndarray, w: np.ndarray, bins: int) -> Callable[[np.ndarray], float]:
    """``var -> sum_ij w_i w_j phi_diag(var)(z_i - z_j)`` from linearly binned weights."""
    n, d = z.shape
    lo, hi = z.min(axis=0), z.max(axis=0)
    step = (hi - lo) / (bins - 1)
    step[step == 0] = 1.0
    counts = np.zeros((bins,) * d)
    # linear binning: each frame's weight shared between the 2^d surrounding grid points
    position = (z - lo) / step
    base = np.clip(np.floor(position).astype(int), 0, bins - 2)
    frac = position - base
    for corner in range(2**d):
        offset = np.array([(corner >> k) & 1 for k in range(d)])
        share = np.prod(np.where(offset, frac, 1 - frac), axis=1)
        np.add.at(counts, tuple((base + offset).T), w * share)
    # autocorrelation of the bin weights for every offset, zero-padded (no wrap-around)
    size = (2 * bins,) * d
    axes = tuple(range(d))
    spectrum = np.fft.rfftn(counts, s=size, axes=axes)
    autocorrelation = np.fft.irfftn(spectrum * np.conj(spectrum), s=size, axes=axes)
    offsets = [np.fft.fftfreq(2 * bins, 1.0 / (2 * bins)) * step[k] for k in range(d)]

    def pair_sum(var: np.ndarray) -> float:
        total = autocorrelation
        for k in range(d):  # the Gaussian is a product over CVs: contract one axis at a time
            total = np.tensordot(total, _gaussian(offsets[k], float(var[k])), axes=([0], [0]))
        return float(total)

    return pair_sum


def _exact_pair_sum(z: np.ndarray, w: np.ndarray) -> Callable[[np.ndarray], float]:
    n, d = z.shape
    chunk = max(1, 2_000_000 // n)

    def pair_sum(var: np.ndarray) -> float:
        log_norm = -0.5 * (d * np.log(2 * np.pi) + np.log(var).sum())
        total = 0.0
        for start in range(0, n, chunk):
            diff2 = (z[start : start + chunk, None, :] - z[None, :, :]) ** 2
            kernel = np.exp(log_norm - 0.5 * (diff2 / var).sum(axis=2))
            total += float(w[start : start + chunk] @ kernel @ w)
        return total

    return pair_sum
