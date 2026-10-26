"""estk: analysis tools for MD + enhanced sampling simulations."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from estk.config import ProjectConfig, load_config
from estk.errors import ConfigError, EstError, MissingFileError, MissingLimitsError
from estk.io.colvar import CVTable
from estk.project import Project, Run, RunMetadata

if TYPE_CHECKING:
    from estk.paths import MinimumEnergyPath
    from estk.surface import Surface

try:
    __version__ = version("enhanced-sampling-toolkit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"


def load(path: Path | str) -> Surface | MinimumEnergyPath:
    """Read a file written by estk: a surface (``estk reweight``) or a path (``estk mep``).

    Grid files of the original scripts are read as surfaces.
    """
    from estk.io.textfile import read_header
    from estk.paths import MinimumEnergyPath
    from estk.surface import Surface

    path = Path(path)
    fields, _ = read_header(path)
    if fields[0] == "s" and "fes" in fields:
        return MinimumEnergyPath.load(path)
    return Surface.load(path)


__all__ = [
    "CVTable",
    "ConfigError",
    "EstError",
    "MissingFileError",
    "MissingLimitsError",
    "Project",
    "ProjectConfig",
    "Run",
    "RunMetadata",
    "__version__",
    "load",
    "load_config",
]
