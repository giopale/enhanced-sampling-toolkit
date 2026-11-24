"""Frame selections (--select): parsing, loading, and how each analysis accounts for them."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import basin_populations, ensemble_averages  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.config import load_config  # noqa: E402
from estk.errors import ConfigError, EstError  # noqa: E402
from estk.project import Project  # noqa: E402
from estk.selection import Selection, selection_from  # noqa: E402

from .test_averages import SITES, write_sites  # noqa: E402
from .test_basins import OPTIONS, write_tilted  # noqa: E402
from .test_combine import CONFIG  # noqa: E402

COLUMNS = {
    "d0": np.array([1.7, 1.9, 2.1, 2.3, np.nan]),
    "cn": np.array([2.0, 1.0, 2.0, 1.0, 1.5]),
    "indirect-delta-B": np.array([-0.5, 0.5, 0.5, -0.5, 0.0]),
    "dOOliq.min": np.array([2.5, 2.7, 2.9, 3.1, 3.3]),
}


def mask(text, available=COLUMNS):
    return Selection.parse(text).resolve(available).mask(COLUMNS).tolist()


def test_expressions():
    assert mask("d0 < 2.0") == [True, True, False, False, False]  # NaN: not selected
    assert mask("1.8 < d0 < 2.2") == [False, True, True, False, False]  # chained
    assert mask("cn > 1.5 and d0 < 2.2") == [True, False, True, False, False]
    assert mask("cn > 1.5 or d0 > 2.2") == [True, False, True, True, False]
    assert mask("not cn > 1.5") == [False, True, False, True, True]
    # the channel of the proton transfer: cn and delta of opposite signs
    assert mask("(cn - 1.5) * indirect_delta_B < 0") == [True, True, False, False, False]
    assert mask("(cn - 1.5) * {indirect-delta-B} < 0") == mask("(cn-1.5)*indirect_delta_B<0")
    assert mask("dOOliq.min ^ 2 >= 2.9 ** 2") == [False, False, True, True, True]
    functions = "abs(indirect_delta_B) < sqrt(0.3) and max(cn, 1.2) == 2"
    assert mask(functions) == [True, False, True, False, False]
    assert mask("cn") == [True] * 5  # a number: selected where it is not 0

    sel = Selection.parse("(cn - 1.5) * indirect_delta_B < 0").resolve(COLUMNS)
    assert sorted(sel.names) == ["cn", "indirect-delta-B"]
    for bad in ("__import__('os')", "d0.__class__()", "lambda x: x", "[d0]", "d0 if cn else 1",
                "open('f')", "d0 <"):  # fmt: skip
        with pytest.raises(EstError):
            Selection.parse(bad)
    with pytest.raises(EstError, match="could be any of"):
        Selection.parse("a_b > 0").resolve(["a-b", "a.b"])
    with pytest.raises(EstError, match="not loaded"):
        Selection.parse("theta > 0").mask(COLUMNS)
    named = selection_from("near", {"near": "d0 < 2"})
    assert named.label == "near" and named.describe() == "near: d0 < 2"


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("selection")
    for k in range(2):
        write_tilted(root / f"r{k}", k, offset=-7.0 * k, n=6000)
        write_sites(root / f"r{k}", k)
    sources = "".join(f"  {s}:\n    source: CV-sites\n" for s in SITES)
    (root / "estk.yaml").write_text(
        CONFIG.format(pattern="r*") + sources + "selections:\n  left: 'd0 < 1.95'\n"
    )
    return root


def fresh(root, select=None):
    config = load_config(root / "estk.yaml")
    config.select(select)
    return Project(config)


def test_loading(project):
    run = fresh(project, "left").runs[0]
    table = run.load_cvs(["s1"])  # d0 is loaded for the selection only
    assert "d0" not in table.columns
    every = run.load_cvs(["s1", "d0"], select=False)
    assert len(table) == np.count_nonzero(every["d0"] < 1.95)
    assert table.report.deselected == len(every) - len(table)
    kept = run.load_cvs(["s1", "d0"], drop_deselected=False)
    assert len(kept) == len(every)
    np.testing.assert_array_equal(kept.selected, every["d0"] < 1.95)
    with pytest.raises(ConfigError, match="selections"):
        (project / "bad.yaml").write_text("selections:\n  x: 'd0 <'\n")
        load_config(project / "bad.yaml")


def test_reweight_and_basins(project):
    plain = fresh(project).reweight(["d0", "cn"], **OPTIONS).combined
    left = fresh(project, "left").reweight(["d0", "cn"], **OPTIONS).combined
    assert left.metadata["selection"] == "left: d0 < 1.95"
    assert "selection" not in plain.metadata
    assert left.metadata["frames"] < plain.metadata["frames"]

    result = basin_populations(fresh(project, "left"), ["d0", "cn"], surface=plain)
    first, second = result.basins
    assert first.population == pytest.approx(1.0)  # within the selection: all on the left
    assert second.population == 0.0
    out, error = result.deselected_population
    assert 0.1 < out < 0.5 and np.isfinite(error)  # mostly the right well, plus the barrier
    total = (sum(b.population_of_all for b in result.basins) + result.unassigned_of_all
             + result.outside_population[0] + result.undefined_population[0] + out)  # fmt: skip
    assert total == pytest.approx(1.0)
    everything = basin_populations(fresh(project), ["d0", "cn"], surface=plain)
    assert first.population_of_all == pytest.approx(everything.basins[0].population_of_all,
                                                    abs=0.02)  # fmt: skip
    assert result.metadata["selection"] == "left: d0 < 1.95"


def test_averages(project):
    result = ensemble_averages(fresh(project, "1.8 < d0 < 2.0"), ["s1"])
    shares = []
    for run in fresh(project).runs:
        t = run.load_cvs(["s1", "d0"])
        w = np.exp(t.bias / t.kbt - (t.bias / t.kbt).max())
        chosen = (t["d0"] > 1.8) & (t["d0"] < 2.0)
        assert result.runs[run.name]["s1"] == pytest.approx(
            np.sum(w[chosen] * t["s1"][chosen]) / w[chosen].sum()
        )
        shares.append(w[chosen].sum() / w.sum())
    assert result.metadata["selected_share"][0] == pytest.approx(np.mean(shares))
    with pytest.raises(EstError, match="no frame is selected"):
        ensemble_averages(fresh(project, "d0 > 99"), ["s1"])


def test_commands(project, monkeypatch):
    monkeypatch.chdir(project)
    runner = CliRunner()
    out = runner.invoke(app, ["average", "s1", "s2", "--select", "left", "-o", "left.png"])
    assert out.exit_code == 0, out.output
    assert (
        "selection: left: d0 < 1.95" in out.output and "averages over the selection" in out.output
    )
    assert (project / "left.dat").read_text().count("left: d0 < 1.95") == 1

    out = runner.invoke(app, ["average", "s1", "-o", "all.png"])  # no selection left over
    assert out.exit_code == 0, out.output
    assert "selection" not in out.output

    out = runner.invoke(app, ["reweight", "d0", "cn", "--points", "41,21",
                              "--select", "cn > 1.2 and d0 < 2.4", "-o", "sel"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "estk.selection" in (project / "sel" / "combined.dat").read_text()

    out = runner.invoke(app, ["basins", "d0", "cn", "--surface", "sel/combined.dat",
                              "--select", "left"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "not selected" in out.output

    out = runner.invoke(app, ["average", "s1", "--select", "d0 <"])
    assert out.exit_code == 1 and "can't parse" in out.output
