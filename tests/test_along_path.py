"""Quantities averaged along a minimum energy path (estk along-path)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import AlongPathResult, averages_along_path  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.paths import MinimumEnergyPath  # noqa: E402
from estk.plotting import plot_along_path  # noqa: E402
from estk.project import Project  # noqa: E402
from estk.reweight import kde_at_points  # noqa: E402

from .test_combine import CONFIG, frames, write_run  # noqa: E402

H = {"d0": 0.03, "cn": 0.05}  # the bandwidths of CONFIG


def q_of(x, y):
    return 2.0 * x + np.sin(3.0 * y)


@pytest.fixture
def project(tmp_path):
    for k, name in enumerate(("a", "b")):
        x, y, bias = frames(k, n=1500, offset=-5.0 * k)
        write_run(tmp_path / name, x, y, bias)
        t = np.arange(len(x)) * 0.05
        np.savetxt(tmp_path / name / "CV-q", np.column_stack([t, q_of(x, y)]), fmt="%.8f",
                   header="FIELDS time q", comments="#! ")  # fmt: skip
        steps = np.arange(len(x)) * 100  # 0.05 ps with the 0.0005 ps timestep
        np.savetxt(tmp_path / name / "model_devi.out",
                   np.column_stack([steps, np.zeros(len(x)), np.full(len(x), 0.1 + k)]),
                   fmt=["%d", "%.4f", "%.4f"], header="step max_devi_v max_devi_f")  # fmt: skip
    (tmp_path / "estk.yaml").write_text(CONFIG.format(pattern="*"))
    return Project(tmp_path / "estk.yaml")


@pytest.fixture
def path():
    points = np.column_stack([np.linspace(1.8, 2.2, 9), np.linspace(1.2, 1.8, 9)])
    energies = 10 * np.sin(np.linspace(0, np.pi, 9))
    return MinimumEnergyPath(("d0", "cn"), points, energies, metadata={"kbt": 2.91})


def brute(table, point, quantity):
    """sum_i w_i K_i q_i / sum_i w_i K_i, and log sum_i w_i K_i with normalised w, K."""
    lw = table.bias / table.kbt
    lw = lw - lw.max()
    h = np.array([H["d0"], H["cn"]])
    d2 = (((table.array(["d0", "cn"]) - point) / h) ** 2).sum(axis=1)
    k = np.exp(lw - 0.5 * d2)
    norm = np.log(np.sum(np.exp(lw))) + np.log(2 * np.pi * h.prod())
    return (k * table[quantity]).sum() / k.sum(), np.log(k.sum()) - norm


def test_kde_at_points_matches_brute_force(project):
    table = project.runs[0].load_cvs(["d0", "cn", "q"])
    targets = np.array([[2.0, 1.5], [1.9, 1.0]])
    h = np.array([H["d0"], H["cn"]])
    estimate = kde_at_points(targets, table.array(["d0", "cn"]), table.bias / table.kbt, h,
                             quantities={"q": table["q"]})  # fmt: skip
    for i, point in enumerate(targets):
        average, log_p = brute(table, point, "q")
        assert estimate.averages["q"][i] == pytest.approx(average, rel=1e-10)
        assert estimate.log_density[i] == pytest.approx(log_p, abs=1e-10)
    assert np.all(estimate.effective_frames > 1)
    assert np.all(estimate.effective_frames <= len(table))


def test_averages_and_combination(project, path):
    result = averages_along_path(project, path, ["q"])
    tables = {run.name: run.load_cvs(["d0", "cn", "q"]) for run in project.runs}
    k = 4
    for name, table in tables.items():
        assert result.runs[name]["q"][k] == pytest.approx(brute(table, path.points[k], "q")[0])
    # probability: each run's share = its total weight times its density at the point
    shares, means = [], []
    for table in tables.values():
        mean, log_p = brute(table, path.points[k], "q")
        total = np.log(np.sum(np.exp(table.bias / table.kbt - 30))) + 30
        shares.append(total + log_p)
        means.append(mean)
    shares = np.exp(np.array(shares) - max(shares))
    shares /= shares.sum()
    expected = float(np.dot(shares, means))
    assert result.values["q"][k] == pytest.approx(expected, rel=1e-9)
    spread = np.sqrt(np.dot(shares, (np.array(means) - expected) ** 2))
    assert result.spread["q"][k] == pytest.approx(spread, rel=1e-6)
    # the average follows q along the path (kernel smoothing aside)
    np.testing.assert_allclose(result.values["q"], q_of(*path.points.T), atol=0.1)
    assert result.saddle_index == 4 and result.metadata["combined"] == "probability"

    equal = averages_along_path(project, path, ["q"], combine="equal")
    assert not np.allclose(equal.values["q"], result.values["q"])


def test_sources_blocks_and_errors(project, path):
    result = averages_along_path(project, path, ["max_devi_f"],
                                 sources={"max_devi_f": "model_devi.out"})  # fmt: skip
    np.testing.assert_allclose(result.runs["a"]["max_devi_f"], 0.1)
    np.testing.assert_allclose(result.runs["b"]["max_devi_f"], 1.1)
    assert np.all((result.values["max_devi_f"] >= 0.1) & (result.values["max_devi_f"] <= 1.1))

    blocks = averages_along_path(project, path, ["q"], blocks=3)
    assert list(blocks.runs) == ["a.b1", "a.b2", "a.b3", "b.b1", "b.b2", "b.b3"]

    with pytest.raises(EstError, match="CV of the path"):
        averages_along_path(project, path, ["d0"])
    with pytest.raises(EstError, match="combine"):
        averages_along_path(project, path, ["q"], combine="legacy")
    with pytest.raises(EstError, match="no quantities"):
        averages_along_path(project, path, [])


def test_one_run_has_no_spread(tmp_path, path):
    x, y, bias = frames(0, n=1500)
    write_run(tmp_path / "a", x, y, bias)
    np.savetxt(tmp_path / "a" / "CV-q", np.column_stack([np.arange(1500) * 0.05, q_of(x, y)]),
               fmt="%.8f", header="FIELDS time q", comments="#! ")  # fmt: skip
    (tmp_path / "estk.yaml").write_text(CONFIG.format(pattern="*"))
    result = averages_along_path(Project(tmp_path / "estk.yaml"), path, ["q"])
    assert np.all(np.isnan(result.spread["q"]))
    np.testing.assert_allclose(result.values["q"], result.runs["a"]["q"])


def test_file_and_plot(project, path, tmp_path):
    result = averages_along_path(project, path, ["q", "max_devi_f"],
                                 sources={"max_devi_f": "model_devi.out"})  # fmt: skip
    back = AlongPathResult.load(result.save(tmp_path / "p.along.dat"))
    assert back.quantities == ["q", "max_devi_f"] and list(back.runs) == ["a", "b"]
    np.testing.assert_allclose(back.values["q"], result.values["q"])
    np.testing.assert_allclose(back.runs["b"]["max_devi_f"], result.runs["b"]["max_devi_f"])
    np.testing.assert_allclose(back.points, path.points)
    assert back.metadata["bandwidth"]["a"] == H
    with pytest.raises(EstError, match="along-path"):
        AlongPathResult.load(path.save(tmp_path / "p.mep.dat"))

    fig = plot_along_path(result, units="kT")
    assert len(fig.axes) == 3
    assert fig.axes[0].get_ylabel() == "F (kT)" and fig.axes[1].get_ylabel() == "q"
    fig = plot_along_path(result, ["q"], x="cn", runs=False)
    assert len(fig.axes) == 2 and len(fig.axes[1].get_lines()) == 4  # line, 2 markers, saddle
    with pytest.raises(EstError, match="x must be"):
        plot_along_path(result, x="theta")
    with pytest.raises(EstError, match="not in the result"):
        plot_along_path(result, ["theta"])


def test_command(project, path, monkeypatch):
    root = project.config.path.parent
    monkeypatch.chdir(root)
    path.save(root / "combined.mep.dat")
    runner = CliRunner()
    result = runner.invoke(app, ["along-path", "combined.mep.dat", "q", "max_devi_f",
                                 "--source", "max_devi_f=model_devi.out"])  # fmt: skip
    assert result.exit_code == 0, result.output
    for text in ("highest point", "effective frames", "max_devi_f"):
        assert text in result.output
    assert (root / "combined.along.dat").is_file() and (root / "combined.along.png").is_file()
    result = runner.invoke(app, ["along-path", "combined.mep.dat", "q", "--source", "oops"])
    assert result.exit_code == 1 and "--source" in result.output
