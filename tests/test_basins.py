"""Basin populations (estk basins) on a tilted double well with known populations."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import BasinResult, basin_map, basin_populations  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.config import load_config  # noqa: E402
from estk.errors import ConfigError, EstError  # noqa: E402
from estk.grid import Grid  # noqa: E402
from estk.plotting import plot_basins  # noqa: E402
from estk.project import Project  # noqa: E402
from estk.surface import Surface  # noqa: E402

from .test_combine import CONFIG, write_run  # noqa: E402
from .test_reweight import KBT_350 as KT  # noqa: E402
from .test_reweight import fy, sample  # noqa: E402

N = 20000
OPTIONS = {"points": (81, 41)}


def tilted(x):
    """Double well in d0 (barrier 5 kT), the right well 2 kT higher at its bottom."""
    return 5 * KT * (((x - 2.0) / 0.25) ** 2 - 1) ** 2 + KT * (x - 2.0) / 0.25


AXIS = np.linspace(1.3, 2.7, 20001)


def exact_left():
    """Population of the left well within the grid limits 1.5-2.5, split at the barrier top."""
    p = np.exp(-tilted(AXIS) / KT)
    middle = (AXIS > 1.8) & (AXIS < 2.2)
    top = AXIS[np.argmax(np.where(middle, tilted(AXIS), -np.inf))]
    inside = (AXIS >= 1.5) & (AXIS <= 2.5)
    return p[inside & (AXIS < top)].sum() / p[inside].sum()


def write_tilted(folder, seed, offset=0.0, n=N):
    rng = np.random.default_rng(seed)
    x = sample(tilted(AXIS), AXIS, rng, n, 10.0)
    ay = np.linspace(0.0, 3.0, 2001)
    y = sample(fy(ay), ay, rng, n, 10.0)
    write_run(folder, x, y, -0.9 * (tilted(x) + fy(y)) + offset)
    q = np.where(x < 2.0, 0.6, 0.1)  # e.g. a dissociation fraction, higher in the left well
    np.savetxt(folder / "CV-q", np.column_stack([np.arange(n) * 0.05, q]), fmt="%.6f",
               header="FIELDS time q", comments="#! ")  # fmt: skip


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("basins")
    for k in range(3):
        write_tilted(root / f"r{k}", k, offset=-7.0 * k)  # offsets: as independent OPES runs
    (root / "estk.yaml").write_text(CONFIG.format(pattern="r*"))
    return Project(root / "estk.yaml")


@pytest.fixture(scope="module")
def result(project):
    return basin_populations(project, ["d0", "cn"], quantities=["q"], reweight_options=OPTIONS)


def test_populations_match_the_exact_ones(result):
    left, right = result.basins
    assert left.point[0] < 2.0 < right.point[0]
    assert left.population + right.population == pytest.approx(1.0)
    assert result.unassigned == 0.0
    assert left.population == pytest.approx(exact_left(), abs=4 * left.error + 1e-3)
    assert 0 < left.error < 0.01
    assert left.surface_population == pytest.approx(left.population, abs=0.01)
    assert left.run_populations.keys() == {"r0", "r1", "r2"}
    delta, error = result.free_energy_difference(right)
    exact = -KT * np.log((1 - exact_left()) / exact_left())
    assert delta == pytest.approx(exact, abs=4 * error + 0.02 * KT)
    assert result.free_energy_difference(left) == (0.0, 0.0)
    # <q> per basin, and the total = sum over basins of P_A <q>_A
    assert left.averages["q"] == pytest.approx(0.6, abs=0.005)  # the split is on the grid
    assert right.averages["q"] == pytest.approx(0.1, abs=1e-9)
    total, _ = result.totals["q"]
    assert total == pytest.approx(sum(b.population * b.averages["q"] for b in result.basins))
    assert 0.2 < result.outside_fraction < 0.5  # frames past d0 1.5-2.5 / cn 0.75-2.25


def test_min_depth_merges_shallow_basins(project, result):
    surface = project.reweight(["d0", "cn"], **OPTIONS).combined
    labels, minima = basin_map(surface, 0.0)
    assert len(minima) >= 2 and labels.min() == 0
    two, _ = basin_map(surface, 2.0 * result.kbt)
    assert set(np.unique(two)) == {0, 1}
    one, _ = basin_map(surface, 6.0 * result.kbt)  # deeper than the right well (about 3 kT)
    assert set(np.unique(one)) == {0}
    merged = basin_populations(project, ["d0", "cn"], surface=surface, min_depth_kt=6.0)
    assert len(merged.basins) == 1 and merged.basins[0].population == pytest.approx(1.0)


def test_basin_map_by_hand():
    grid = Grid(("x",), (0.0,), (8.0,), (9,))
    values = np.array([3.0, 1.0, 2.0, 2.5, 2.2, 4.0, 0.0, 1.0, 5.0])
    labels, minima = basin_map(Surface(grid, values), 0.0)
    # minima at x = 6 (lowest), 1, and the shallow one at 4 (0.3 deep)
    assert minima == [6, 1, 4]
    # x = 3 (2.5) descends to its lowest neighbour, x = 2 (2.0), not to x = 4 (2.2)
    assert labels.tolist() == [1, 1, 1, 1, 2, 0, 0, 0, 0]
    labels, minima = basin_map(Surface(grid, values), 0.5)  # 4 spills into the basin of 1
    assert minima == [6, 1] and labels.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0]
    values[2] = np.inf  # no data
    labels, _ = basin_map(Surface(grid, values), 0.0)
    assert labels[2] == -1


def test_named_regions_one_run_blocks(tmp_path):
    write_tilted(tmp_path / "r0", 0)
    (tmp_path / "estk.yaml").write_text(
        CONFIG.format(pattern="r*")
        + "basins:\n  left: {d0: [null, 1.95]}\n  right: {d0: [2.05, null], cn: [0.75, 2.25]}\n"
    )
    project = Project(tmp_path / "estk.yaml")
    result = basin_populations(project, ["d0", "cn"], quantities=["q"], blocks=4,
                               reweight_options=OPTIONS)  # fmt: skip
    assert [b.name for b in result.basins] == ["left", "right"]
    assert result.metadata["mode"] == "regions"
    assert 0 < result.unassigned < 0.1  # the barrier strip 1.95-2.05 is in no region
    total = sum(b.population for b in result.basins) + result.unassigned
    assert total == pytest.approx(1.0)
    assert np.isfinite(result.basins[0].error)  # from the time blocks of the one run
    assert result.basins[0].averages["q"] == pytest.approx(0.6)

    # shares of all frames: one run, so exactly the population in the grid scaled down by
    # the reweighted share past the walls; everything adds up to 1
    past, past_error = result.outside_population
    assert 0 < past < 1 and np.isfinite(past_error)
    for b in result.basins:
        assert b.population_of_all == pytest.approx(b.population * (1 - past))
        assert np.isfinite(b.error_of_all)
    total = sum(b.population_of_all for b in result.basins) + result.unassigned_of_all + past
    assert total == pytest.approx(1.0)
    table = project.runs[0].load_cvs(["d0", "cn", "q"])
    w = np.exp(table.bias / table.kbt - (table.bias / table.kbt).max())
    expected = np.sum(w * table["q"]) / w.sum()  # <q> over every frame, OPES reweighted
    assert result.totals_of_all["q"][0] == pytest.approx(expected)

    with pytest.raises(EstError, match="not CVs"):
        basin_populations(project, ["d0", "cn"], regions={"a": {"theta": (0, 1)}},
                          reweight_options=OPTIONS)  # fmt: skip
    with pytest.raises(EstError, match="CV of the basins"):
        basin_populations(project, ["d0", "cn"], quantities=["d0"], reweight_options=OPTIONS)
    with pytest.raises(EstError, match="equal or probability"):
        basin_populations(project, ["d0", "cn"], combine="legacy", reweight_options=OPTIONS)


def test_config_regions_are_checked(tmp_path):
    (tmp_path / "estk.yaml").write_text("basins:\n  a: {d0: [2.0, 1.0]}\n")
    with pytest.raises(ConfigError, match="empty"):
        load_config(tmp_path / "estk.yaml")
    (tmp_path / "estk.yaml").write_text("basins:\n  a: {}\n")
    with pytest.raises(ConfigError, match="at least one CV"):
        load_config(tmp_path / "estk.yaml")


def test_probability_combination(project, result):
    other = basin_populations(project, ["d0", "cn"], combine="probability",
                              reweight_options=OPTIONS)  # fmt: skip
    weights = other.metadata["run_weights"]
    assert weights["r0"] > weights["r1"] > weights["r2"]  # the bias offsets of the runs
    expected = sum(weights[r] * p for r, p in other.basins[0].run_populations.items())
    assert other.basins[0].population == pytest.approx(expected)


def test_file_and_plot(result, project, tmp_path):
    back = BasinResult.load(result.save(tmp_path / "basins.dat"))
    assert [b.name for b in back.basins] == [b.name for b in result.basins]
    for a, b in zip(back.basins, result.basins, strict=True):
        assert a.population == pytest.approx(b.population)
        assert a.averages["q"] == pytest.approx(b.averages["q"])
        assert a.run_populations == pytest.approx(b.run_populations)
        assert a.population_of_all == pytest.approx(b.population_of_all)
    assert back.totals["q"][0] == pytest.approx(result.totals["q"][0])
    assert back.totals_of_all["q"][0] == pytest.approx(result.totals_of_all["q"][0])
    assert back.outside_population == pytest.approx(result.outside_population)
    assert back.grid == result.grid and back.metadata["mode"] == "automatic"
    labels = Surface.load(result.label_surface().save(tmp_path / "map.dat"))
    np.testing.assert_array_equal(labels.values, result.labels)

    surface = project.reweight(["d0", "cn"], **OPTIONS).combined
    ax = plot_basins(result, surface, units="kT")
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("1\n") and t.endswith("%") for t in texts)
    with pytest.raises(EstError, match="grid"):
        plot_basins(result, project.reweight(["d0", "cn"], points=11).combined)


def test_command(project, monkeypatch):
    root = project.config.path.parent
    monkeypatch.chdir(root)
    runner = CliRunner()
    out = runner.invoke(app, ["basins", "d0", "cn", "--quantity", "q", "--points", "81,41"])
    assert out.exit_code == 0, out.output
    for text in ("in the grid", "ΔF", "total <q>", "past the grid limits", "of all frames",
                 "past the walls", "<q>"):  # fmt: skip
        assert text in out.output
    folder = root / "basins-d0-cn"
    for name in ("basins.dat", "basins-map.dat", "basins.png"):
        assert (folder / name).is_file()

    project.reweight(["d0", "cn"], **OPTIONS).combined.save(root / "fes.dat")
    out = runner.invoke(app, ["basins", "d0", "cn", "--surface", "fes.dat", "-o", "again"])
    assert out.exit_code == 0, out.output
    assert (root / "again" / "basins.dat").is_file()
    out = runner.invoke(app, ["basins", "d0", "cn", "--source", "oops"])
    assert out.exit_code == 1 and "--source" in out.output


def test_config_regions_of_other_surfaces_are_skipped(project, tmp_path):
    """estk.yaml can name basins for several surfaces: each surface uses the ones in its CVs."""
    root = project.config.path.parent
    text = (root / "estk.yaml").read_text()
    other = text + "basins:\n  pt-left: {indirect-delta-B: [null, 0]}\n"
    (tmp_path / "estk.yaml").write_text(other.replace("runs:\n", f"runs:\n  root: {root}\n"))
    only_other = Project(tmp_path / "estk.yaml")
    result = basin_populations(only_other, ["d0", "cn"], reweight_options=OPTIONS)
    assert result.metadata["mode"] == "automatic" and len(result.basins) == 2

    both = other + "  left: {d0: [null, 1.95]}\n"
    (tmp_path / "estk.yaml").write_text(both.replace("runs:\n", f"runs:\n  root: {root}\n"))
    mixed = Project(tmp_path / "estk.yaml")
    result = basin_populations(mixed, ["d0", "cn"], reweight_options=OPTIONS)
    assert result.metadata["mode"] == "regions"
    assert [b.name for b in result.basins] == ["left"]
    forced = basin_populations(mixed, ["d0", "cn"], automatic=True, reweight_options=OPTIONS)
    assert forced.metadata["mode"] == "automatic"
