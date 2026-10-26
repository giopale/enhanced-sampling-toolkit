import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import ConvergenceResult, barrier_convergence  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.paths import evaluate, minimum_energy_path  # noqa: E402
from estk.plotting import plot_convergence  # noqa: E402
from estk.project import Project  # noqa: E402

from .test_combine import CONFIG, frames, write_run  # noqa: E402


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("convergence")
    write_run(root / "a", *frames(0, n=4000))  # 0.05 ps per frame: 0.2 ns
    write_run(root / "b", *frames(1, n=4000))
    (root / "estk.yaml").write_text(CONFIG.format(pattern="*"))
    return Project(root / "estk.yaml")


OPTIONS = {"points": (21, 15)}


def test_time_scan(project):
    result = barrier_convergence(
        project, ["d0", "cn"], "time", [0.05, 0.1, 0.15, 0.2], reweight_options=OPTIONS
    )
    assert result.parameter == "time"
    assert [r.value for r in result.rows] == [0.05, 0.1, 0.15, 0.2]
    frames_seen = result.column("frames")
    assert np.all(np.diff(frames_seen) > 0)
    assert frames_seen[-1] == 8000
    assert set(result.rows[0].run_barriers) == {"a", "b"}
    assert np.isfinite(result.rows[-1].spread)

    # the last step uses all the data: its barrier is the one along the reference path
    full = project.reweight(["d0", "cn"], time=(None, 0.2), **OPTIONS).combined
    reference = minimum_energy_path(full)
    energies = evaluate(full, reference.points)
    assert result.rows[-1].barrier == pytest.approx(np.max(energies) - energies[0], abs=1e-6)
    np.testing.assert_allclose(result.path.points, reference.points)


def test_relaxed_time_scan_and_start(project):
    result = barrier_convergence(
        project, ["d0", "cn"], "time", [0.15, 0.2], start=0.05, relax=True,
        reweight_options=OPTIONS,
    )  # fmt: skip
    assert result.metadata["relax"] is True
    assert result.metadata["start_ns"] == 0.05
    assert result.rows[-1].frames == 2 * 3000  # frames at 50-199.95 ps in each run
    with pytest.raises(EstError, match="not after the start"):
        barrier_convergence(project, ["d0", "cn"], "time", [0.05], start=0.05,
                            reweight_options=OPTIONS)  # fmt: skip


def test_bandwidth_scan(project, tmp_path):
    result = barrier_convergence(
        project, ["d0", "cn"], "bandwidth", [0.01, 0.02, 0.05], cv="d0",
        reweight_options=OPTIONS, save_steps=tmp_path / "steps",
    )  # fmt: skip
    assert result.label == "bandwidth of d0"
    barriers = result.column("barrier")
    assert len(set(np.round(barriers, 6))) == 3  # the bandwidth changes the barrier
    assert (tmp_path / "steps" / "bandwidth-0.02.dat").is_file()
    assert (tmp_path / "steps" / "bandwidth-0.02.mep.dat").is_file()
    with pytest.raises(EstError, match="which CV"):
        barrier_convergence(project, ["d0", "cn"], "bandwidth", [0.01], cv="x")

    saved = result.save(tmp_path / "convergence.dat")
    back = ConvergenceResult.load(saved)
    assert back.parameter == "bandwidth" and back.label == result.label
    np.testing.assert_allclose(back.column("barrier"), barriers)
    assert back.rows[0].run_barriers.keys() == result.rows[0].run_barriers.keys()
    ax = plot_convergence(back, units="kT")
    assert ax.get_ylabel() == "energy (kT)"


def test_convergence_commands(project, tmp_path, monkeypatch):
    monkeypatch.chdir(project.config.base_dir)
    runner = CliRunner()
    out = tmp_path / "time"
    result = runner.invoke(app, ["convergence", "time", "d0", "cn", "--until", "0.2", "--step",
                                 "0.1", "--points", "11,9", "-o", str(out)])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert {p.name for p in out.iterdir()} >= {
        "convergence.dat",
        "convergence.png",
        "convergence.pdf",
        "reference.mep.dat",
    }
    out = tmp_path / "bw"
    result = runner.invoke(
        app,
        [
            "convergence",
            "bandwidth",
            "d0",
            "cn",
            "--cv",
            "cn",
            "--range",
            "0.02",
            "0.1",
            "--num",
            "3",
            "--points",
            "11,9",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(ConvergenceResult.load(out / "convergence.dat").rows) == 3
    result = runner.invoke(app, ["convergence", "bandwidth", "d0", "cn", "--cv", "cn"])
    assert result.exit_code == 1
    assert "--values or --range" in result.output
