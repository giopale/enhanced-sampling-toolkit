
from string import Template
from typing import Tuple, Union
import sys
import numpy as np
import numpy.typing as npt
import concurrent.futures
from tqdm import tqdm
from logger_setup import logger




def compute_frame_weights(
    bias: npt.NDArray[np.floating],
    kbT: float,
    *,
    bias_in_kbt: bool = False,
    sign: float = +1.0,
    normalize: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Compute per-frame weights w_i for reweighting.

    Parameters
    ----------
    bias : (n,) array
        Bias values per frame.
    kbT : float
        Thermal energy.
    bias_in_kbt : bool
        If True, bias is already dimensionless (i.e. bias/kbT).
    sign : float
        Use +1.0 for exp(+beta*bias), -1.0 for exp(-beta*bias), depending on convention.
    normalize : bool
        If True, normalize weights to sum to 1 (optional; FES shape unchanged).

    Returns
    -------
    w : (n,) array
        Positive weights.
    """
    bias = np.asarray(bias, dtype=float).reshape(-1)
    if bias_in_kbt:
        logw = sign * bias
    else:
        if kbT <= 0:
            raise ValueError("kbT must be > 0.")
        beta = 1.0 / kbT
        logw = sign * beta * bias

    # subtract max for numerical stability (safe because only ratios matter)
    logw = logw - np.max(logw)
    w = np.exp(logw)

    if normalize:
        s = np.sum(w)
        if s == 0.0:
            raise ValueError("All weights underflowed to zero; check bias scale/sign.")
        w = w / s
    return w


def make_grid_1d(
    xmin: float,
    xmax: float,
    n_grid: int,
) -> npt.NDArray[np.floating]:

    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")

    return np.linspace(xmin, xmax, n_grid, dtype=float)


def make_grid_2d(
    xmin: float, xmax: float, n_grid_x: int,
    ymin: float, ymax: float, n_grid_y: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:

    if n_grid_x < 2 or n_grid_y < 2:
        raise ValueError("n_grid_x/y must be >= 2")

    x = np.linspace(xmin, xmax, n_grid_x, dtype=float)
    y = np.linspace(ymin, ymax, n_grid_y, dtype=float)

    return np.meshgrid(x, y, indexing="ij")


def _as_spd_2x2(H: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    H = np.asarray(H, dtype=float)
    if H.shape != (2, 2):
        raise ValueError("H must be 2x2.")
    H = 0.5 * (H + H.T)
    np.linalg.cholesky(H)
    return H


def fes_kde_2d(
    cv_x: npt.NDArray[np.floating],
    cv_y: npt.NDArray[np.floating],
    grid_X: npt.NDArray[np.floating],
    grid_Y: npt.NDArray[np.floating],
    H: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    kbT: float,
    *,
    use_H_inv: bool = False,
    frame_chunk: int = 15000,
    dtype_work=np.float32,
    add_eps: float = 0.0,
    shift_min_to_zero: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Chunk over frames (data), accumulate p on the full grid.
    Best when nframes is huge and grid is moderate.
    """
    cv_x = np.asarray(cv_x, dtype=float).reshape(-1)
    cv_y = np.asarray(cv_y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if not (cv_x.size == cv_y.size == weights.size):
        raise ValueError("cv_x, cv_y, weights must have the same length.")
    if kbT <= 0:
        raise ValueError("kbT must be > 0.")

    grid_X = np.asarray(grid_X, dtype=float)
    grid_Y = np.asarray(grid_Y, dtype=float)
    if grid_X.shape != grid_Y.shape:
        raise ValueError("grid_X and grid_Y must have the same shape.")

    nx, ny = grid_X.shape
    gx = grid_X.reshape(-1).astype(dtype_work, copy=False)  # (G,)
    gy = grid_Y.reshape(-1).astype(dtype_work, copy=False)  # (G,)
    G = gx.size
    N = cv_x.size

    H = _as_spd_2x2(H)
    Hinv = H if use_H_inv else np.linalg.inv(H)
    a = np.array(Hinv[0, 0], dtype=dtype_work)
    b = np.array(Hinv[0, 1], dtype=dtype_work)
    c = np.array(Hinv[1, 1], dtype=dtype_work)

    p = np.zeros(G, dtype=np.float64)

    for start in range(0, N, frame_chunk):
        end = min(start + frame_chunk, N)

        x = cv_x[start:end].astype(dtype_work, copy=False)   # (B,)
        y = cv_y[start:end].astype(dtype_work, copy=False)   # (B,)
        w = weights[start:end].astype(dtype_work, copy=False)

        # Build (G,B) differences via broadcasting
        dX = gx[:, None] - x[None, :]
        dY = gy[:, None] - y[None, :]

        dist2 = a * dX * dX + 2.0 * b * dX * dY + c * dY * dY
        K = np.exp(-0.5 * dist2, dtype=dtype_work)           # (G,B)

        p += (K @ w).astype(np.float64)

    if add_eps > 0.0:
        p = p + add_eps

    F = -kbT * np.log(p).reshape(nx, ny)

    if shift_min_to_zero:
        finite = np.isfinite(F)
        if np.any(finite):
            F = F - np.min(F[finite])

    return F



def _logsumexp_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute log(exp(a)+exp(b)) elementwise, stable, handling -inf."""
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def fes_kde_2d_logsum(
    cv_x: npt.NDArray[np.floating],
    cv_y: npt.NDArray[np.floating],
    grid_X: npt.NDArray[np.floating],
    grid_Y: npt.NDArray[np.floating],
    H: npt.NDArray[np.floating],
    logw: npt.NDArray[np.floating],     # log weights per frame (dimensionless)
    kbT: float,
    *,
    use_H_inv: bool = False,
    frame_chunk: int = 5000,
    add_log_eps: float = -np.inf,       # optional: log(eps) floor on p
    shift_min_to_zero: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Stable computation:
        log p(g) = logsumexp_i( logw[i] - 1/2 * dist2(g,i) )
        F(g) = -kbT * log p(g)

    Works with full 2x2 bandwidth matrix H.
    """
    cv_x = np.asarray(cv_x, dtype=float).reshape(-1)
    cv_y = np.asarray(cv_y, dtype=float).reshape(-1)
    logw = np.asarray(logw, dtype=float).reshape(-1)

    if not (cv_x.size == cv_y.size == logw.size):
        raise ValueError("cv_x, cv_y, logw must have the same length.")
    if kbT <= 0:
        raise ValueError("kbT must be > 0.")

    grid_X = np.asarray(grid_X, dtype=float)
    grid_Y = np.asarray(grid_Y, dtype=float)
    if grid_X.shape != grid_Y.shape:
        raise ValueError("grid_X and grid_Y must have the same shape.")

    # H handling
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    np.linalg.cholesky(H)
    Hinv = H if use_H_inv else np.linalg.inv(H)

    a = Hinv[0, 0]
    b = Hinv[0, 1]
    c = Hinv[1, 1]

    nx, ny = grid_X.shape
    gx = grid_X.reshape(-1)  # (G,)
    gy = grid_Y.reshape(-1)  # (G,)
    G = gx.size
    N = cv_x.size

    # Accumulator in log-space: start at -inf = log(0)
    logp = np.full(G, -np.inf, dtype=float)

    for start in tqdm(range(0, N, frame_chunk),total=int(np.ceil(N/frame_chunk)),ncols=80):
        end = min(start + frame_chunk, N)

        x = cv_x[start:end]
        y = cv_y[start:end]
        lw = logw[start:end]  # (B,)

        dX = gx[:, None] - x[None, :]
        dY = gy[:, None] - y[None, :]

        dist2 = a * dX * dX + 2.0 * b * dX * dY + c * dY * dY  # (G,B)
        arg = lw[None, :] - 0.5 * dist2                         # (G,B)

        # logsumexp over frames in this chunk
        m = np.max(arg, axis=1)
        chunk_logp = m + np.log(np.sum(np.exp(arg - m[:, None]), axis=1))

        # combine with running logp
        logp = _logsumexp_pair(logp, chunk_logp)

    # Optional floor on p to avoid +inf if you really want a finite cap:
    if np.isfinite(add_log_eps):
        logp = np.maximum(logp, add_log_eps)

    F = (-kbT * logp).reshape(nx, ny)

    if shift_min_to_zero:
        finite = np.isfinite(F)
        if np.any(finite):
            F = F - np.min(F[finite])

    return F



def point_kernel_weights(
    cv_x: np.ndarray,
    cv_y: np.ndarray,
    logw: np.ndarray,
    H: np.ndarray,
    xg: float,
    yg: float,
    *,
    use_H_inv: bool = False,
    frame_chunk: int = 200_000,
    return_log: bool = False,
):
    """
    Compute per-frame contributions at a single grid point (xg, yg).

    Returns:
      - weights: normalized weights over frames at this point (sum=1)
        (or log_weights if return_log=True)
      - logp: log p(xg,yg) = logsumexp_i( logw[i] - 1/2 dist2_i )
    """
    cv_x = np.asarray(cv_x, dtype=float).reshape(-1)
    cv_y = np.asarray(cv_y, dtype=float).reshape(-1)
    logw = np.asarray(logw, dtype=float).reshape(-1)
    if not (cv_x.size == cv_y.size == logw.size):
        raise ValueError("cv_x, cv_y, logw must have same length.")

    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    np.linalg.cholesky(H)
    Hinv = H if use_H_inv else np.linalg.inv(H)
    a = Hinv[0, 0]
    b = Hinv[0, 1]
    c = Hinv[1, 1]

    N = cv_x.size
    # First pass: compute logp = logsumexp(u_i) in chunks (stable)
    logp = -np.inf
    for start in range(0, N, frame_chunk):
        end = min(start + frame_chunk, N)
        dx = xg - cv_x[start:end]
        dy = yg - cv_y[start:end]
        dist2 = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        u = logw[start:end] - 0.5 * dist2

        # chunk logsumexp
        m = np.max(u)
        chunk = m + np.log(np.sum(np.exp(u - m)))
        # combine running logp with chunk
        if logp == -np.inf:
            logp = chunk
        else:
            M = max(logp, chunk)
            logp = M + np.log(np.exp(logp - M) + np.exp(chunk - M))

    # Second pass: compute normalized weights w_i = exp(u_i - logp)
    # (You need a second pass unless you store all u_i from pass 1.)
    out = np.empty(N, dtype=float)
    for start in range(0, N, frame_chunk):
        end = min(start + frame_chunk, N)
        dx = xg - cv_x[start:end]
        dy = yg - cv_y[start:end]
        dist2 = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
        u = logw[start:end] - 0.5 * dist2
        out[start:end] = u - logp  # log normalized weights

    if return_log:
        return out, logp  # log(weights), logp
    return np.exp(out), logp


def logsumexp_rows_safe(A: np.ndarray) -> np.ndarray:
    """
    Row-wise logsumexp, safe when a row is all -inf.
    A: (G,B) -> out: (G,)
    """
    m = np.max(A, axis=1)
    out = np.full_like(m, -np.inf, dtype=float)

    mask = np.isfinite(m)
    if np.any(mask):
        tmp = A[mask] - m[mask, None]
        out[mask] = m[mask] + np.log(np.sum(np.exp(tmp), axis=1))

    return out

#def _logsumexp_pair(loga: np.ndarray, logb: np.ndarray) -> np.ndarray:
#    """
#    Elementwise logsumexp of two vectors:
#      out = log(exp(loga) + exp(logb))
#    Handles -inf correctly.
#    """
#    m = np.maximum(loga, logb)
#    return m + np.log(np.exp(loga - m) + np.exp(logb - m))

def _logsumexp_pair(loga: np.ndarray, logb: np.ndarray) -> np.ndarray:
    # elementwise log(exp(loga) + exp(logb)), safe for -inf
    return np.logaddexp(loga, logb)

def kde_avg_2d_logsum_signed(
    cv_x: npt.NDArray[np.floating],
    cv_y: npt.NDArray[np.floating],
    q: npt.NDArray[np.floating],        # signed quantity per frame
    grid_X: npt.NDArray[np.floating],
    grid_Y: npt.NDArray[np.floating],
    H: npt.NDArray[np.floating],
    logw: npt.NDArray[np.floating],
    *,
    use_H_inv: bool = False,
    frame_chunk: int = 5000,
    q_floor: float = 0.0,               # add small floor if you want; 0.0 keeps exact zeros
    denom_logfloor: float = -np.inf,    # optional: floor on log(D) to avoid NaNs in empty regions
) -> npt.NDArray[np.floating]:
    """
    Returns <q>(g) on the (grid_X, grid_Y) grid using kernel-reweighted conditional averaging.

    <q>(g) = (sum_i w_i K(g,i) q_i) / (sum_i w_i K(g,i))
          = (Splus - Sminus) / D
    where Splus = sum w K max(q,0) and Sminus = sum w K max(-q,0).

    Stable in log-space for each nonnegative sum; final subtraction happens in linear space.
    """

    cv_x = np.asarray(cv_x, dtype=float).reshape(-1)
    cv_y = np.asarray(cv_y, dtype=float).reshape(-1)
    q    = np.asarray(q,    dtype=float).reshape(-1)
    logw = np.asarray(logw, dtype=float).reshape(-1)

    if not (cv_x.size == cv_y.size == q.size == logw.size):
        raise ValueError("cv_x, cv_y, q, logw must have the same length.")

    grid_X = np.asarray(grid_X, dtype=float)
    grid_Y = np.asarray(grid_Y, dtype=float)
    if grid_X.shape != grid_Y.shape:
        raise ValueError("grid_X and grid_Y must have the same shape.")

    # H handling (SPD)
    H = np.asarray(H, dtype=float)
    H = 0.5 * (H + H.T)
    np.linalg.cholesky(H)  # raises if not SPD
    Hinv = H if use_H_inv else np.linalg.inv(H)

    a = Hinv[0, 0]
    b = Hinv[0, 1]
    c = Hinv[1, 1]

    nx, ny = grid_X.shape
    gx = grid_X.reshape(-1)
    gy = grid_Y.reshape(-1)
    G = gx.size
    N = cv_x.size

    # Precompute log(q_plus) and log(q_minus) safely
    q_plus = np.maximum(q, 0.0)
    q_minus = np.maximum(-q, 0.0)

    if q_floor > 0.0:
        q_plus = np.maximum(q_plus, q_floor)
        q_minus = np.maximum(q_minus, q_floor)

    logq_plus = np.full_like(q_plus, -np.inf, dtype=float)
    maskp = q_plus > 0
    logq_plus[maskp] = np.log(q_plus[maskp])

    logq_minus = np.full_like(q_minus, -np.inf, dtype=float)
    maskm = q_minus > 0
    logq_minus[maskm] = np.log(q_minus[maskm])

    logD = np.full(G, -np.inf, dtype=float)   # log denom
    logSp = np.full(G, -np.inf, dtype=float)  # log Splus
    logSm = np.full(G, -np.inf, dtype=float)  # log Sminus

    #for start in range(0, N, frame_chunk):
    for start in tqdm(range(0, N, frame_chunk),total=int(np.ceil(N/frame_chunk)),ncols=80):
        end = min(start + frame_chunk, N)
    
        x = cv_x[start:end]
        y = cv_y[start:end]
        lw = logw[start:end]
        lqp = logq_plus[start:end]
        lqm = logq_minus[start:end]
    
        dX = gx[:, None] - x[None, :]
        dY = gy[:, None] - y[None, :]
    
        dist2 = a * dX * dX + 2.0 * b * dX * dY + c * dY * dY  # (G,B)
        base = lw[None, :] - 0.5 * dist2                       # (G,B)
    
        # denom chunk (G,)
        chunk_logD = logsumexp_rows_safe(base)
    
        # positive numerator chunk (G,)
        base_p = base + lqp[None, :]
        chunk_logSp = logsumexp_rows_safe(base_p)
    
        # negative numerator chunk (G,)
        base_m = base + lqm[None, :]
        chunk_logSm = logsumexp_rows_safe(base_m)
    
        # combine with running accumulators (all are (G,))
        logD  = _logsumexp_pair(logD,  chunk_logD)
        logSp = _logsumexp_pair(logSp, chunk_logSp)
        logSm = _logsumexp_pair(logSm, chunk_logSm)

    if np.isfinite(denom_logfloor):
        logD = np.maximum(logD, denom_logfloor)

    # Convert to linear only at the end
    D  = np.exp(logD)
    Sp = np.exp(logSp)
    Sm = np.exp(logSm)

    numer = Sp - Sm
    avg = numer / D

    # Regions with essentially no support -> NaN
    avg[~np.isfinite(avg)] = np.nan

    return avg.reshape(nx, ny)
