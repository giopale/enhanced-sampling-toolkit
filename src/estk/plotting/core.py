"""Shared plotting helpers: styles, colours, units and axis labels."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from pathlib import Path

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from estk.errors import EstError
from estk.units import energy_factor

STYLES_DIR = Path(__file__).parent / "styles"

# Ink for text, axes and secondary marks (text never takes a series colour).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# Categorical series colours, assigned in this fixed order (validated for colour-vision
# deficiencies as adjacent pairs). Three slots are below 3:1 contrast on white, so series
# drawn with them are always labelled (legend + direct labels).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# Path overlays on free-energy maps (stands out from the blue sequential scale).
PATH_COLOR = SERIES[1]

# One-hue sequential scale for free energies. The colour encodes the depth of the basins: the
# lowest free energy is darkest, the top of the range lightest, and everything above the range
# (and points without data) is left white so that it recedes into the page.
_BLUE_STEPS = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)  # fmt: skip
FREE_ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "estk_free_energy", [*_BLUE_STEPS[::-1], "#e6f0fc"]
).with_extremes(over="white", bad="white")

# Property averages: a second one-hue scale (orange), as the second sequential context.
_ORANGE_STEPS = ("#fde4d6", "#f9c3a6", "#f4a079", "#ef7d4e", "#eb6834", "#c9501f", "#9e3b12")
PROPERTY_CMAP = LinearSegmentedColormap.from_list(
    "estk_property", list(_ORANGE_STEPS)
).with_extremes(bad="white")

DISPLAY_UNITS = {"kj/mol": "kJ/mol", "kcal/mol": "kcal/mol", "ev": "eV", "kt": "kT"}


def style_path(style: str | Path) -> str:
    """A bundled style name (paper, slides) or a path to an .mplstyle file."""
    bundled = STYLES_DIR / f"{style}.mplstyle"
    if bundled.is_file():
        return str(bundled)
    path = Path(style).expanduser()
    if path.is_file():
        return str(path)
    names = ", ".join(p.stem for p in sorted(STYLES_DIR.glob("*.mplstyle")))
    raise EstError(f"unknown plot style '{style}' (bundled: {names}; or a .mplstyle file)")


@contextlib.contextmanager
def plot_style(style: str | Path | None) -> Iterator[None]:
    """Use a style for the figures made inside the ``with`` block."""
    import matplotlib.pyplot as plt

    if style is None:
        yield
        return
    with plt.style.context(style_path(style)):
        yield


def use_agg() -> None:
    """Draw without a display (command line, batch jobs)."""
    matplotlib.use("Agg")


def energy_scale(units: str, to: str | None, kbt: float | None) -> tuple[float, str]:
    """Factor converting energies in ``units`` to ``to`` (default: keep), and its label."""
    if to is None or to.lower() == units.lower():
        return 1.0, DISPLAY_UNITS.get(units.lower(), units)
    target = to.lower()
    if target == "kt":
        if not kbt:
            raise EstError("kT unknown for this surface; can't show energies in kT")
        return 1.0 / kbt, "kT"
    return energy_factor(units) / energy_factor(target), DISPLAY_UNITS.get(target, to)


def axis_label(name: str, labels: Mapping[str, str] | None = None) -> str:
    return (labels or {}).get(name, name)


def labels_from_config(config: object | None) -> dict[str, str]:
    """Axis labels from estk.yaml: ``cv.<name>.label`` and ``units``."""
    labels: dict[str, str] = {}
    for name, cv in getattr(config, "cv", {}).items():
        text = cv.label or name
        if cv.units:
            text = f"{text} ({cv.units})"
        labels[name] = text
    return labels
