"""Kernel bandwidths."""

from __future__ import annotations

import numpy as np


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
