import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.cli.main import app  # noqa: E402
from estk.config import ProjectConfig  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.grid import Grid  # noqa: E402
from estk.paths import minimum_energy_path  # noqa: E402
from estk.plotting import (  # noqa: E402
    labels_from_config,
    plot_path_profile,
    plot_style,
    plot_surface,
    plot_surfaces,
    style_path,
)
from estk.plotting.core import energy_scale  # noqa: E402
from estk.surface import Surface  # noqa: E402

KBT = 2.91


@pytest.fixture
def surface():
    grid = Grid(("d0", "cn"), (1.5, 0.75), (2.5, 2.25), (41, 31))
    X, Y = grid.mesh()
    f = 5 * KBT * (((X - 2.0) / 0.25) ** 2 - 1) ** 2 + 2 * KBT * ((Y - 1.5) / 0.3) ** 2
    f[0, 0] = np.inf
    error = np.full(grid.shape, 0.5)
    return Surface(
        grid,
        f - f[np.isfinite(f)].min(),
        error=error,
        metadata={"kbt": KBT, "combined": "probability", "runs": ["a", "b"]},
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_surface_map_with_path(surface):
    path = minimum_energy_path(surface, max_steps=200)
    with plot_style("paper"):
        ax = plot_surface(surface, path=path, units="eV", labels={"d0": "d(Ti–O) (Å)"})
    assert ax.get_xlabel() == "d(Ti–O) (Å)"
    assert ax.get_ylabel() == "cn"
    colorbar = ax.figure.axes[-1]
    assert colorbar.get_ylabel() == "F (eV)"
    ticks = colorbar.get_yticks()
    assert ticks.min() >= 0 and ticks.max() <= 20 * KBT / 96.485 + 1e-9  # inside the range
    assert len(ax.lines) == 4  # path, start, end, saddle


def test_small_multiples_share_a_scale(surface):
    other = Surface(surface.grid, surface.values * 0.8, metadata={"kbt": KBT})
    fig = plot_surfaces({"a": surface, "b": other, "c": other}, ncols=2)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == 4  # 3 panels + one shared colour bar
    assert {ax.get_title() for ax in visible[:3]} == {"a", "b", "c"}


def test_1d_and_property_surfaces(surface):
    grid = Grid(("d0",), (1.5,), (2.5,), (50,))
    one_d = Surface(grid, np.linspace(0, 10, 50), error=np.ones(50), metadata={"kbt": KBT})
    ax = plot_surface(one_d, units="kT")
    assert ax.get_ylabel() == "F (kT)"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), np.linspace(0, 10, 50) / KBT)
    prop = Surface(surface.grid, np.random.default_rng(0).random(surface.grid.shape), quantity="q")
    ax = plot_surface(prop)
    assert ax.figure.axes[-1].get_ylabel() == "q"


def test_path_profile_with_runs(surface):
    runs = {
        "a": Surface(surface.grid, surface.values * 1.1, metadata={"kbt": KBT}),
        "b": Surface(surface.grid, surface.values * 0.9, metadata={"kbt": KBT}),
    }
    path = minimum_energy_path(surface, max_steps=200, profiles=runs)
    ax = plot_path_profile(path, units="kJ/mol")
    labels = ax.get_legend_handles_labels()[1]
    assert labels == ["a", "b", "spread between runs", "all runs"]
    texts = [t.get_text() for t in ax.texts]
    assert "a" in texts and "b" in texts  # direct labels
    assert any("reverse" in t for t in texts)


def test_units_and_styles():
    assert energy_scale("kj/mol", "eV", None)[0] == pytest.approx(1 / 96.485, rel=1e-4)
    assert energy_scale("kj/mol", "kT", 2.5) == (0.4, "kT")
    assert energy_scale("kj/mol", None, None) == (1.0, "kJ/mol")
    with pytest.raises(EstError, match="kT unknown"):
        energy_scale("kj/mol", "kT", None)
    assert style_path("paper").endswith("paper.mplstyle")
    with pytest.raises(EstError, match="unknown plot style"):
        style_path("poster")
    config = ProjectConfig.model_validate({"cv": {"d0": {"label": "d(Ti–O)", "units": "Å"}}})
    assert labels_from_config(config) == {"d0": "d(Ti–O) (Å)"}


def test_plot_commands(surface, workdir):
    surface.save(workdir / "combined.dat")
    Surface(surface.grid, surface.values * 1.1, metadata={"kbt": KBT}).save(workdir / "a.dat")
    Surface(surface.grid, surface.values * 0.9, metadata={"kbt": KBT}).save(workdir / "b.dat")
    runner = CliRunner()
    result = runner.invoke(app, ["mep", "combined.dat"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "plot",
            "surface",
            "combined.dat",
            "--path",
            "combined.mep.dat",
            "--units",
            "eV",
            "-o",
            "fes.png",
            "-o",
            "fes.pdf",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (workdir / "fes.png").stat().st_size > 0 and (workdir / "fes.pdf").stat().st_size > 0
    result = runner.invoke(app, ["plot", "surface", "a.dat", "b.dat", "combined.dat"])
    assert result.exit_code == 0, result.output
    assert (workdir / "a.png").exists()
    result = runner.invoke(app, ["plot", "path", "combined.mep.dat", "--style", "slides"])
    assert result.exit_code == 0, result.output
    assert (workdir / "combined.mep.png").exists()
    result = runner.invoke(app, ["plot", "path", "combined.mep.dat", "--style", "poster"])
    assert result.exit_code == 1


def test_colour_bands_match_contours(surface):
    ax = plot_surface(surface, vmax=10.0, contour_step=2.0)
    filled, lines = ax.collections[0], ax.collections[1]
    np.testing.assert_allclose(filled.levels, [0, 2, 4, 6, 8, 10])
    np.testing.assert_allclose(lines.levels, filled.levels)
    assert matplotlib.colors.to_hex(lines.get_edgecolor()[0]) == "#0b0b0b"
    np.testing.assert_allclose(ax.figure.axes[-1].get_yticks(), filled.levels)
    smooth = plot_surface(surface, vmax=10.0, contour_step=0)
    assert len(smooth.collections[0].levels) == 65


def test_contour_labels(surface):
    some = plot_surface(surface, vmax=10.0, contour_step=2.0)
    assert sorted(t.get_text() for t in some.texts if t.get_text()) and {
        t.get_text() for t in some.texts
    } <= {"4", "8"}
    every = plot_surface(surface, vmax=10.0, contour_step=2.0, contour_labels="all")
    assert {t.get_text() for t in every.texts} <= {"2", "4", "6", "8", "10"}
    assert len(every.texts) > len(some.texts)
    assert len(plot_surface(surface, vmax=10.0, contour_step=2.0, contour_labels="none").texts) == 0
    with pytest.raises(EstError, match="contour labels"):
        plot_surface(surface, contour_labels="few")


def test_explicit_levels(surface, workdir):
    ax = plot_surface(surface, levels=[0, 2, 5, 10, 20], units="kJ/mol")
    np.testing.assert_allclose(ax.collections[0].levels, [0, 2, 5, 10, 20])
    np.testing.assert_allclose(ax.collections[1].levels, [0, 2, 5, 10, 20])
    with pytest.raises(EstError, match="two contour levels"):
        plot_surface(surface, levels=[1.0])
    surface.save(workdir / "s.dat")
    runner = CliRunner()
    result = runner.invoke(
        app, ["plot", "surface", "s.dat", "--levels", "0,0.05,0.1,0.2", "--units", "eV"]
    )
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["plot", "surface", "s.dat", "--levels", "0,a"]).exit_code == 1


def test_default_units_and_style_come_from_the_project(surface, workdir, monkeypatch):
    import estk.plotting

    seen = {}
    real_surface, real_profile, real_style = (
        estk.plotting.plot_surface, estk.plotting.plot_path_profile, estk.plotting.plot_style,
    )  # fmt: skip

    def spy(name, real):
        def wrapper(*args, **kwargs):
            seen[name] = kwargs.get("units", args[0] if name == "style" else None)
            return real(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(estk.plotting, "plot_surface", spy("surface", real_surface))
    monkeypatch.setattr(estk.plotting, "plot_path_profile", spy("profile", real_profile))
    monkeypatch.setattr(estk.plotting, "plot_style", spy("style", real_style))
    surface.save(workdir / "s.dat")
    runner = CliRunner()
    assert runner.invoke(app, ["mep", "s.dat"]).exit_code == 0

    # no estk.yaml: eV and the paper style
    assert runner.invoke(app, ["plot", "surface", "s.dat"]).exit_code == 0
    assert seen == {"surface": "eV", "style": "paper"}
    # estk.yaml sets both; the options override
    (workdir / "estk.yaml").write_text("plot:\n  units: kj/mol\n  style: slides\n")
    assert runner.invoke(app, ["plot", "path", "s.mep.dat"]).exit_code == 0
    assert seen["profile"] == "kJ/mol" and seen["style"] == "slides"
    result = runner.invoke(app, ["plot", "path", "s.mep.dat", "--units", "kT", "--style", "paper"])
    assert result.exit_code == 0, result.output
    assert seen["profile"] == "kT" and seen["style"] == "paper"


def test_plot_units_setting():
    assert ProjectConfig().plot.units == "eV"
    assert ProjectConfig.model_validate({"plot": {"units": "KCAL/MOL"}}).plot.units == "kcal/mol"
    with pytest.raises(ValueError, match="unknown energy units"):
        ProjectConfig.model_validate({"plot": {"units": "hartree"}})
