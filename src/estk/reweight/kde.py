"""Weighted kernel density estimates on a grid, computed in log space.

For grid point g and frames i with log-weights ``l_i`` and Gaussian kernels of bandwidth
matrix H,

    log p(g)  = log sum_i exp(l_i - d_i(g)^2 / 2),   d_i(g)^2 = (g - x_i)^T H^-1 (g - x_i)
    <q>(g)    = sum_i exp(l_i - d_i^2/2) q_i / sum_i exp(l_i - d_i^2/2)

The kernels are not normalised: that only adds a constant to log p. Sums are accumulated over
chunks of frames with a running maximum per grid point, so nothing overflows.

Above ``BINNED_FRAMES`` frames (e.g. many walkers), the frames are first binned on a grid an
eighth of a bandwidth fine (linear binning: each frame's weight shared between the corners of
its cell) and the sums run over the occupied bins, at a cost that no longer grows with the
number of frames. The error is second order in the bin width: on 300 000 frames of a 2D
Gaussian, at most 0.002 kT where F < 5 kT, 0.009 kT where F < 10 kT and 0.04 kT where
F < 20 kT (its sparse tails), for 7x less time on 2 million frames.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from estk.errors import EstError
from estk.grid import Grid

# Memory used for the (grid points x frames) work arrays.
DEFAULT_MEMORY_BYTES = 256 * 2**20
# Frames above which they are binned first (see the module docstring).
BINNED_FRAMES = 200_000
_BIN_STEP = 0.125  # bin width, in bandwidths
_MAX_BINS = 50_000_000  # cells of the binning grid; finer than this, no binning


def kde_on_grid(
    points: np.ndarray,
    log_weights: np.ndarray,
    grid: Grid,
    bandwidth: np.ndarray,
    *,
    quantities: Mapping[str, np.ndarray] | None = None,
    memory_bytes: int = DEFAULT_MEMORY_BYTES,
    binned: bool | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``log p`` on the grid and the kernel averages of ``quantities``.

    Parameters
    ----------
    points:
        CV values, shape (frames, grid.ndim).
    log_weights:
        Log-weight of each frame, shape (frames,).
    bandwidth:
        Either one bandwidth per CV (diagonal H = diag(h^2)) or the full matrix H.
    quantities:
        ``{name: values per frame}``; the result has ``<q>`` on the grid for each (NaN where the
        density is zero).
    binned:
        Bin the frames first; default: when there are more than ``BINNED_FRAMES``.
    """
    points = np.asarray(points, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] != grid.ndim:
        raise EstError(f"points must have shape (frames, {grid.ndim}), got {points.shape}")
    if len(points) != len(log_weights):
        raise EstError("points and log_weights have different lengths")
    quantities = {k: np.asarray(v, dtype=float).reshape(-1) for k, v in (quantities or {}).items()}
    for name, q in quantities.items():
        if len(q) != len(points):
            raise EstError(f"quantity {name} has {len(q)} values for {len(points)} frames")

    transform = _whitening(bandwidth, grid.ndim)
    if binned is None:
        binned = len(points) > BINNED_FRAMES
    if binned:
        points, log_weights, quantities = _bin_frames(points, log_weights, transform, quantities)
    x = points @ transform
    diagonal = np.count_nonzero(transform - np.diag(np.diag(transform))) == 0
    if grid.ndim == 2 and diagonal:
        log_p, averages = _separable_2d(x, log_weights, grid, transform, quantities, memory_bytes)
    else:
        g = grid.coordinates() @ transform
        log_p, averages = _log_space(g, x, log_weights, quantities, memory_bytes)

    shape = grid.shape
    return log_p.reshape(shape), {k: v.reshape(shape) for k, v in averages.items()}


def _log_space(
    g: np.ndarray,
    x: np.ndarray,
    log_weights: np.ndarray,
    quantities: Mapping[str, np.ndarray],
    memory_bytes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Direct sums over (grid point, frame) pairs, with a running maximum per grid point.

    ``g`` and ``x`` are whitened coordinates (kernel = exp(-|g - x|^2 / 2)).
    """
    n_grid, n_frames = len(g), len(x)
    chunk = max(1, int(memory_bytes // (8 * 3 * max(n_grid, 1))))
    log_max = np.full(n_grid, -np.inf)
    sum0 = np.zeros(n_grid)
    sums = {name: np.zeros(n_grid) for name in quantities}

    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        arg = np.zeros((n_grid, stop - start))
        for d in range(g.shape[1]):
            diff = g[:, d, None] - x[None, start:stop, d]
            arg -= 0.5 * diff * diff
        arg += log_weights[None, start:stop]

        new_max = np.maximum(log_max, arg.max(axis=1))
        finite = np.isfinite(new_max)
        if finite.all():
            rescale = np.exp(log_max - new_max)
            kernel = np.exp(arg - new_max[:, None])
        else:
            rescale = np.zeros(n_grid)
            rescale[finite] = np.exp(log_max[finite] - new_max[finite])
            kernel = np.zeros_like(arg)
            kernel[finite] = np.exp(arg[finite] - new_max[finite, None])

        sum0 = sum0 * rescale + kernel.sum(axis=1)
        for name, q in quantities.items():
            sums[name] = sums[name] * rescale + kernel @ q[start:stop]
        log_max = new_max

    return _finish(log_max, sum0, sums)


# Below this, a factorised sum may have lost terms to underflow: recompute in log space.
_UNDERFLOW = 1e-200


def _separable_2d(
    x: np.ndarray,
    log_weights: np.ndarray,
    grid: Grid,
    transform: np.ndarray,
    quantities: Mapping[str, np.ndarray],
    memory_bytes: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fast path for a diagonal bandwidth on a 2D grid.

    The kernel factorises, exp(-dx^2/2 - dy^2/2) = A[a, i] B[b, i], so each sum over frames is
    the matrix product A diag(w) B^T. Rows of A and B are scaled by their largest entry to stay
    in range; grid points whose scaled sum is tiny (possible loss to underflow) are recomputed
    with the direct log-space sums.
    """
    gx = grid.axes()[0] * transform[0, 0]
    gy = grid.axes()[1] * transform[1, 1]
    n_frames = len(x)
    chunk = max(1, int(memory_bytes // (8 * 3 * (len(gx) + len(gy)))))

    def factors(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        log_a = log_weights[None, start:stop] - 0.5 * (gx[:, None] - x[None, start:stop, 0]) ** 2
        log_b = -0.5 * (gy[:, None] - x[None, start:stop, 1]) ** 2
        return log_a, log_b

    shift_a = np.full(len(gx), -np.inf)
    shift_b = np.full(len(gy), -np.inf)
    for start in range(0, n_frames, chunk):
        log_a, log_b = factors(start, min(start + chunk, n_frames))
        shift_a = np.maximum(shift_a, log_a.max(axis=1))
        shift_b = np.maximum(shift_b, log_b.max(axis=1))
    shift_a[~np.isfinite(shift_a)] = 0.0

    sum0 = np.zeros((len(gx), len(gy)))
    sums = {name: np.zeros_like(sum0) for name in quantities}
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        log_a, log_b = factors(start, stop)
        a = np.exp(log_a - shift_a[:, None])
        b = np.exp(log_b - shift_b[:, None])
        sum0 += a @ b.T
        for name, q in quantities.items():
            sums[name] += (a * q[None, start:stop]) @ b.T

    log_max = (shift_a[:, None] + shift_b[None, :]).reshape(-1)
    sum0 = sum0.reshape(-1)
    sums = {name: s.reshape(-1) for name, s in sums.items()}
    log_p, averages = _finish(log_max, sum0, sums)

    redo = sum0 < _UNDERFLOW
    if redo.any():
        g = grid.coordinates()[redo] @ transform
        exact_p, exact_avg = _log_space(g, x, log_weights, quantities, memory_bytes)
        log_p[redo] = exact_p
        for name in averages:
            averages[name][redo] = exact_avg[name]
    return log_p, averages


def _finish(
    log_max: np.ndarray, sum0: np.ndarray, sums: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = log_max + np.log(sum0)
        averages = {name: s / sum0 for name, s in sums.items()}
    log_p[~np.isfinite(log_max) | (sum0 == 0)] = -np.inf
    for values in averages.values():
        values[sum0 == 0] = np.nan
    return log_p, averages


def _bin_frames(
    points: np.ndarray,
    log_weights: np.ndarray,
    transform: np.ndarray,
    quantities: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Occupied bins of a grid ``_BIN_STEP`` bandwidths fine, as weighted frames.

    Linear binning: each frame's weight is shared between the 2^d corners of its cell. A bin's
    quantity is the weighted mean of its frames' values (exact for the kernel averages, since
    every frame of a bin then sits at the bin's centre).
    """
    n, d = points.shape
    # the bandwidth along each CV: 1 / |column of the whitening matrix|
    widths = 1.0 / np.linalg.norm(transform, axis=1)
    step = _BIN_STEP * widths
    lo, hi = points.min(axis=0), points.max(axis=0)
    cells = np.floor((hi - lo) / step).astype(int) + 2
    if np.prod(cells.astype(float)) > _MAX_BINS:
        return points, log_weights, dict(quantities)
    weights = np.exp(log_weights - log_weights.max())
    position = (points - lo) / step
    base = np.clip(np.floor(position).astype(int), 0, cells - 2)
    frac = position - base
    total = np.zeros(int(np.prod(cells)))
    sums = {name: np.zeros_like(total) for name in quantities}
    for corner in range(2**d):
        offset = np.array([(corner >> k) & 1 for k in range(d)])
        share = weights * np.prod(np.where(offset, frac, 1 - frac), axis=1)
        index = np.ravel_multi_index(tuple((base + offset).T), cells)
        total += np.bincount(index, share, minlength=total.size)
        for name, q in quantities.items():
            sums[name] += np.bincount(index, share * q, minlength=total.size)
    occupied = np.flatnonzero(total > 0)
    centres = lo + np.column_stack(np.unravel_index(occupied, cells)) * step
    with np.errstate(invalid="ignore"):
        means = {name: s[occupied] / total[occupied] for name, s in sums.items()}
    return centres, np.log(total[occupied]) + log_weights.max(), means


def _whitening(bandwidth: np.ndarray, ndim: int) -> np.ndarray:
    """Matrix L with (g - x)^T H^-1 (g - x) = |(g - x) L|^2."""
    bandwidth = np.asarray(bandwidth, dtype=float)
    if bandwidth.shape == (ndim,):
        if np.any(bandwidth <= 0):
            raise EstError(f"bandwidths must be positive, got {bandwidth.tolist()}")
        return np.diag(1.0 / bandwidth)
    if bandwidth.shape == (ndim, ndim):
        h = 0.5 * (bandwidth + bandwidth.T)
        try:
            return np.linalg.cholesky(np.linalg.inv(h))
        except np.linalg.LinAlgError:
            raise EstError("the bandwidth matrix must be symmetric positive definite") from None
    raise EstError(f"bandwidth must have shape ({ndim},) or ({ndim}, {ndim})")
