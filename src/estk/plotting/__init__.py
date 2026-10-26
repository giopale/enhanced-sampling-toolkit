"""Figures: free-energy maps, profiles and paths.

Every function draws on a matplotlib ``Axes`` (or makes a figure) and returns it, so figures
can be composed and restyled. ``plot_style("paper")`` applies the bundled single-column journal
style; ``"slides"`` is larger; any ``.mplstyle`` file works too.
"""

from estk.plotting.convergence import plot_convergence
from estk.plotting.core import (
    FREE_ENERGY_CMAP,
    PROPERTY_CMAP,
    SERIES,
    labels_from_config,
    plot_style,
    style_path,
)
from estk.plotting.path import plot_path_profile
from estk.plotting.surface import draw_path, plot_surface, plot_surfaces

__all__ = [
    "FREE_ENERGY_CMAP",
    "PROPERTY_CMAP",
    "SERIES",
    "draw_path",
    "labels_from_config",
    "plot_convergence",
    "plot_path_profile",
    "plot_style",
    "plot_surface",
    "plot_surfaces",
    "style_path",
]
