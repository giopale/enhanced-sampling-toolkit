"""estk: analysis tools for MD + enhanced sampling simulations."""

from importlib.metadata import PackageNotFoundError, version

from estk.config import ProjectConfig, load_config
from estk.errors import ConfigError, EstError, MissingFileError, MissingLimitsError
from estk.io.colvar import CVTable
from estk.project import Project, Run, RunMetadata

try:
    __version__ = version("enhanced-sampling-toolkit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

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
    "load_config",
]
