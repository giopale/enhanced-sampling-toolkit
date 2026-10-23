"""estk: analysis tools for MD + enhanced sampling simulations."""

from importlib.metadata import PackageNotFoundError, version

from estk.config import ProjectConfig, load_config
from estk.errors import ConfigError, EstError, MissingFileError, MissingLimitsError

try:
    __version__ = version("enhanced-sampling-toolkit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "ConfigError",
    "EstError",
    "MissingFileError",
    "MissingLimitsError",
    "ProjectConfig",
    "__version__",
    "load_config",
]
