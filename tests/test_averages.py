"""Equilibrium averages over whole runs (estk average) and their bar chart."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import AverageResult, ensemble_averages  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.plotting import SERIES, plot_averages  # noqa: E402
from estk.project import Project  # noqa: E402

from .test_basins import write_tilted  # noqa: E402
from .test_combine import CONFIG  # noqa: E402

SITES = ["s1", "s2", "s3"]


def write_sites(folder, seed):
    """Site occupations: 0/1 per frame, more often on the right of the double well."""
    rng = np.random.default_rng(seed)
    colvar = np.loadtxt(folder / "COLVAR", comments="#")
    x = colvar[:, 1]
    columns = [colvar[:, 0]]
    for p in (0.1, 0.3, 0.5):
        columns.append((rng.random(len(x)) < np.where(x < 2.0, p, 2 * p)).astype(float))
    columns[3][::7] = np.nan  # s3 undefined on some frames
    np.savetxt(folder / "CV-sites", np.column_stack(columns), fmt="%.6f",
               header="FIELDS time s1 s2 s3", comments="#! ")  # fmt: skip


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("averages")
    for k in range(3):
        write_tilted(root / f"r{k}", k, offset=-7.0 * k, n=6000)
        write_sites(root / f"r{k}", k)
    sources = "".join(f"  {s}:\n    source: CV-sites\n" for s in SITES)
    (root / "estk.yaml").write_text(CONFIG.format(pattern="r*") + sources)
    return Project(root / "estk.yaml")


def brute(table, q):
    w = np.exp(table.bias / table.kbt - (table.bias / table.kbt).max())
    v = table[q]
    ok = np.isfinite(v)
    return np.sum(w[ok] * v[ok]) / w[ok].sum()


def test_averages(project):
    result = ensemble_averages(project, SITES)
    tables = {r.name: r.load_cvs(SITES, drop_nan=False) for r in project.runs}
    for name, table in tables.items():
        for q in SITES:
            assert result.runs[name][q] == pytest.approx(brute(table, q))
    for q in SITES:
        per_run = [result.runs[r][q] for r in tables]
        assert result.values[q] == pytest.approx(np.mean(per_run))  # equal weights
        assert result.errors[q] == pytest.approx(np.std(per_run, ddof=1) / np.sqrt(3))
    assert np.isfinite(result.values["s3"])  # NaN frames skipped, not propagated
    assert result.values["s1"] < result.values["s2"] < result.values["s3"]

    probability = ensemble_averages(project, SITES, combine="probability")
    weights = probability.metadata["run_weights"]
    expected = sum(weights[r] * probability.runs[r]["s1"] for r in weights)
    assert probability.values["s1"] == pytest.approx(expected)

    with pytest.raises(EstError, match="no quantities"):
        ensemble_averages(project, [])
    with pytest.raises(EstError, match="combine"):
        ensemble_averages(project, SITES, combine="legacy")


def test_one_run_uses_blocks(tmp_path):
    write_tilted(tmp_path / "r0", 0, n=3000)
    write_sites(tmp_path / "r0", 0)
    sources = "".join(f"  {s}:\n    source: CV-sites\n" for s in SITES)
    (tmp_path / "estk.yaml").write_text(CONFIG.format(pattern="r*") + sources)
    project = Project(tmp_path / "estk.yaml")
    assert np.isnan(ensemble_averages(project, ["s1"]).errors["s1"])
    blocked = ensemble_averages(project, ["s1"], blocks=4)
    assert 0 < blocked.errors["s1"] < 0.05 and blocked.metadata["error"] == "between blocks"


def test_file_and_chart(project, tmp_path):
    result = ensemble_averages(project, SITES)
    back = AverageResult.load(result.save(tmp_path / "a.dat"))
    assert back.quantities == SITES and back.runs.keys() == result.runs.keys()
    assert back.values == pytest.approx(result.values)

    groups = {"bridging": ["s2", "s1"], "terminal": ["s3"]}
    ax = plot_averages(result, groups=groups, labels={"s1": "S1"}, ylabel="occupation")
    bars = ax.patches[: len(SITES)]
    assert [to_hex(b.get_facecolor()) for b in bars] == [SERIES[0], SERIES[0], SERIES[1]]
    centres = [b.get_x() + b.get_width() / 2 for b in bars]
    assert centres[1] - centres[0] == pytest.approx(1.0)
    assert centres[2] - centres[1] > 1.4  # the gap between groups
    assert [t.get_text() for t in ax.get_xticklabels()] == ["s2", "S1", "s3"]
    assert [b.get_height() for b in bars] == pytest.approx([result.values[q] for q in
                                                            ("s2", "s1", "s3")])  # fmt: skip
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["bridging", "terminal"]
    assert ax.get_ylabel() == "occupation" and ax.get_ylim()[0] == 0.0

    single = plot_averages(result, quantities=["s1"])
    assert single.get_legend() is None  # one colour, nothing to explain
    with pytest.raises(EstError, match="not in the averages"):
        plot_averages(result, groups={"a": ["nope"]})
    with pytest.raises(EstError, match="two groups"):
        plot_averages(result, groups={"a": ["s1"], "b": ["s1"]})


def test_command(project, monkeypatch):
    root = project.config.path.parent
    monkeypatch.chdir(root)
    runner = CliRunner()
    out = runner.invoke(app, ["average", "s1", "s2", "s3", "--group", "bridging=s1,s2",
                              "--group", "terminal=s3", "--label", "s1=S1",
                              "--ylabel", "occupation", "-o", "sites.png"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "bridging" in out.output and "between runs" in out.output
    assert (root / "sites.png").is_file() and (root / "sites.dat").is_file()
    out = runner.invoke(app, ["average", "s1", "--group", "oops"])
    assert out.exit_code == 1 and "--group" in out.output
