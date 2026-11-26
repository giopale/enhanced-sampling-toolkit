"""Defaults of every analysis from estk.yaml: analysis.time and analysis.points."""

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.analysis import (
    ConvergenceResult,
    barrier_convergence,
    basin_populations,
    ensemble_averages,
)
from estk.cli.main import app
from estk.config import load_config
from estk.errors import ConfigError
from estk.project import Project

from .test_averages import SITES, write_sites
from .test_basins import write_tilted
from .test_combine import CONFIG


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("defaults")
    for k in range(2):
        write_tilted(root / f"r{k}", k, offset=-7.0 * k, n=4000)  # 0.05 ps per frame: 0.2 ns
        write_sites(root / f"r{k}", k)
    sources = "".join(f"  {s}:\n    source: CV-sites\n" for s in SITES)
    (root / "estk.yaml").write_text(
        CONFIG.format(pattern="r*") + sources + "analysis:\n  time: [0.1, null]\n  points: 31\n"
    )
    return root


def test_config(tmp_path):
    (tmp_path / "estk.yaml").write_text("analysis:\n  time: [2, null]\n  points: [300, 200]\n")
    analysis = load_config(tmp_path / "estk.yaml").analysis
    assert analysis.time == (2.0, None) and analysis.points == [300, 200]
    assert analysis.window(None) == (2.0, None) and analysis.window((0, 1)) == (0, 1)
    (tmp_path / "empty.yaml").write_text("runs:\n  pattern: '*'\n")
    assert load_config(tmp_path / "empty.yaml").analysis.window(None) is None
    for bad, message in (("time: [3, 1]", "before the end"), ("points: 1", "at least 2")):
        (tmp_path / "bad.yaml").write_text(f"analysis:\n  {bad}\n")
        with pytest.raises(ConfigError, match=message):
            load_config(tmp_path / "bad.yaml")


def test_defaults_apply(root):
    project = Project(root / "estk.yaml")
    result = project.reweight(["d0", "cn"])
    assert result.grid.points == (31, 31)
    surface = result.surfaces["r0"]
    assert surface.metadata["time_window_ns"] == [0.1, None]
    assert surface.metadata["frames"] == pytest.approx(2000, abs=2)  # the second half
    assert project.reweight(["d0", "cn"], points=11, time=(0, None)).grid.points == (11, 11)

    averages = ensemble_averages(project, ["s1"])
    assert averages.metadata["time_window_ns"] == [0.1, None]
    assert averages.metadata["frames"] == pytest.approx(4000, abs=4)
    everything = ensemble_averages(project, ["s1"], time=(None, None))
    assert everything.metadata["frames"] == 8000

    basins = basin_populations(project, ["d0", "cn"])
    assert basins.metadata["time_window_ns"] == [0.1, None]

    scan = barrier_convergence(project, ["d0", "cn"], "time", [0.15, 0.2])
    assert scan.metadata["start_ns"] == 0.1  # the scan starts where the window starts
    assert scan.rows[0].frames < scan.rows[1].frames
    assert np.all(np.isfinite(scan.column("barrier")))


def test_command_options_win(root, monkeypatch):
    monkeypatch.chdir(root)
    runner = CliRunner()
    out = runner.invoke(app, ["reweight", "d0", "cn", "-o", "rw"])
    assert out.exit_code == 0, out.output
    assert "31 points" in out.output
    out = runner.invoke(app, ["reweight", "d0", "cn", "--points", "21", "--time", "0", "inf",
                              "-o", "rw2"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "21 points" in out.output and "31 points" not in out.output


def test_time_scan_command_starts_at_the_configured_start(root, monkeypatch):
    monkeypatch.chdir(root)
    out = CliRunner().invoke(app, ["convergence", "time", "d0", "cn", "--until", "0.2",
                                   "--step", "0.05", "-o", "scan"])  # fmt: skip
    assert out.exit_code == 0, out.output
    result = ConvergenceResult.load(root / "scan" / "convergence.dat")
    assert result.column("value").tolist() == pytest.approx([0.15, 0.2])  # windows from 0.1 ns
    assert result.metadata["start_ns"] == 0.1
