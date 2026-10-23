"""Values on a CV grid: a free-energy surface or the average of a quantity."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from estk.grid import Grid


@dataclass
class Surface:
    """Values on a grid.

    For a free-energy surface (``quantity == "fes"``) the values are free energies in ``units``,
    shifted so that the minimum is 0; ``metadata["fes_offset"]`` is the shift that was removed
    (so ``values + fes_offset = -kT log p`` with weights normalised to sum 1). Points without
    data are ``inf``. For any other quantity the values are kernel-weighted averages (NaN
    without data).

    ``metadata`` records how the surface was made (run, temperature, bandwidths, time window,
    ...); it is written to and read from the file header.
    """

    grid: Grid
    values: np.ndarray
    quantity: str = "fes"
    units: str = "kj/mol"
    error: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.shape != self.grid.shape:
            raise ValueError(f"values have shape {self.values.shape}, grid {self.grid.shape}")
        if self.error is not None and np.shape(self.error) != self.grid.shape:
            raise ValueError("error must have the same shape as values")

    @property
    def is_fes(self) -> bool:
        return self.quantity == "fes"

    def save(self, path: Path | str) -> Path:
        """Write the surface as a PLUMED-style grid text file."""
        from estk.io.grids import write_surface

        return write_surface(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> Surface:
        """Read a surface written by estk, or a grid file of the original scripts."""
        from estk.io.grids import read_surface

        return read_surface(Path(path))
