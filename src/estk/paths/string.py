"""Zero-temperature string method on a gridded surface (any number of dimensions).

Weinan E, Weiqing Ren, Eric Vanden-Eijnden, "Simplified and improved string method for computing
the minimum energy paths in barrier-crossing events", J. Chem. Phys. 126, 164103 (2007).

The update follows the original scripts (``to_refactor/stringmethod_custom.py``, after
A. Pallath's implementation): gradients from finite differences on the grid, interpolated with
quadratic splines; each Euler step moves the image with the largest gradient by ``dt`` (in CV
units); then the images are redistributed at equal arc length.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from estk.errors import EstError


@dataclass
class StringResult:
    points: np.ndarray  # (images, ndim)
    converged: bool
    steps: int
    history: list[float] = field(default_factory=list)  # rms change per step


def initial_string(knots: Sequence[Sequence[float]], images: int) -> np.ndarray:
    """Images equally spaced along the polyline through ``knots`` (start, via..., end)."""
    knots = np.asarray(knots, dtype=float)
    if len(knots) < 2:
        raise EstError("a path needs a start and an end")
    lengths = np.linalg.norm(np.diff(knots, axis=0), axis=1)
    if lengths.sum() == 0:
        raise EstError("the start and end of the path are the same point")
    s = np.concatenate([[0.0], np.cumsum(lengths)]) / lengths.sum()
    target = np.linspace(0.0, 1.0, images)
    return np.column_stack([np.interp(target, s, knots[:, d]) for d in range(knots.shape[1])])


def evolve_string(
    values: np.ndarray,
    axes: Sequence[np.ndarray],
    string: np.ndarray,
    *,
    dt: float = 0.01,
    max_steps: int = 2000,
    tol: float | None = None,
    fixed_ends: bool = True,
) -> StringResult:
    """Relax ``string`` (images x ndim) towards the minimum energy path on ``values``.

    ``values`` must be finite (replace ``inf`` before calling). Stops when the rms change of the
    images in one step is below ``tol`` (default ``max(images**-4, 1e-10)``, as in the original
    scripts) or after ``max_steps``.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise EstError("the surface has non-finite values; cap them first")
    ndim = values.ndim
    string = np.array(string, dtype=float)
    images = len(string)
    if images < 3:
        raise EstError("the string needs at least 3 images")
    tol = max(images**-4.0, 1e-10) if tol is None else tol

    spacing = np.array([a[1] - a[0] for a in axes])
    origin = np.array([a[0] for a in axes])
    gradients = np.gradient(values, *axes)
    if ndim == 1:
        gradients = [gradients]

    def gradient_at(points: np.ndarray) -> np.ndarray:
        coords = ((points - origin) / spacing).T
        return np.column_stack(
            [map_coordinates(g, coords, order=2, mode="nearest") for g in gradients]
        )

    history: list[float] = []
    converged = False
    steps = 0
    while steps < max_steps:
        steps += 1
        previous = string.copy()
        grad = gradient_at(string)
        largest = float(np.max(np.linalg.norm(grad, axis=1)))
        if largest == 0.0:
            converged = True
            break
        move = dt * grad / largest
        if fixed_ends:
            move[0] = move[-1] = 0.0
        string = string - move

        # Redistribute the images at equal arc length.
        lengths = np.linalg.norm(np.diff(string, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(lengths)])
        if s[-1] == 0:
            break
        s /= s[-1]
        target = np.linspace(0.0, 1.0, images)
        string = np.column_stack([np.interp(target, s, string[:, d]) for d in range(ndim)])

        change = float(np.sqrt(np.mean((string - previous) ** 2)))
        history.append(change)
        if change < tol:
            converged = True
            break
    return StringResult(points=string, converged=converged, steps=steps, history=history)
