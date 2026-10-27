"""Multiple walkers in one folder (LAMMPS partitions, PLUMED WALKERS_MPI)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import opes_summary, read_timeseries  # noqa: E402
from estk.cli.main import app  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.io.plumed import find_walkers, walker_file  # noqa: E402
from estk.project import Project  # noqa: E402
from estk.surface import Surface  # noqa: E402

from .conftest import LAMMPS_LOG, PLUMED_INPUT  # noqa: E402
from .test_combine import frames  # noqa: E402

SHARED = PLUMED_INPUT.replace("  STATE_WFILE=STATE\n", "  STATE_WFILE=STATE\n  WALKERS_MPI\n")
CONFIG = (
    "runs:\n  pattern: 'sim*'\n{walkers}"
    "cv:\n  d0:\n    limits: [1.5, 2.5]\n    bandwidth: 0.03\n"
    "  cn:\n    limits: [0.75, 2.25]\n    bandwidth: 0.05\n"
)
WALKERS = 3
N = 1200  # frames per walker, every 0.05 ps


def write_walkers(folder, plumed=SHARED, restart=False):
    """A folder as LAMMPS partitions leave it: COLVAR.<i>, log.lammps.<i>, model_devi.<i>.out."""
    folder.mkdir(parents=True)
    (folder / "plumed.plumed").write_text(plumed)
    (folder / "log.lammps").write_text("LAMMPS (2 Aug 2023)\nRunning on 3 partitions\n")
    for i in range(WALKERS):
        x, y, bias = frames(10 + i, n=N)
        t = np.arange(N) * 0.05
        rows = np.column_stack([t, x, y, bias, np.sin(t + i)])
        if restart and i == 1:  # a restart repeats the last 50 frames
            rows = np.vstack([rows, rows[-50:]])
        np.savetxt(folder / f"COLVAR.{i}", rows, fmt="%.8f",
                   header="FIELDS time d0 cn opes.bias opes.rct", comments="#! ")  # fmt: skip
        (folder / f"log.lammps.{i}").write_text(LAMMPS_LOG)
        steps = np.arange(N) * 100  # 0.05 ps with the 0.0005 ps timestep
        np.savetxt(folder / f"model_devi.{i}.out",
                   np.column_stack([steps, np.zeros(N), 0.1 * (i + 1) * np.ones(N)]),
                   fmt=["%d", "%.4f", "%.4f"], header="step max_devi_v max_devi_f")  # fmt: skip
    return folder


@pytest.fixture
def root(tmp_path):
    write_walkers(tmp_path / "sim1", restart=True)
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=""))
    return tmp_path


def test_walker_files(tmp_path):
    for name in ("COLVAR.0", "COLVAR.2", "model_devi.0.out", "model_devi.2.out", "COLVAR"):
        (tmp_path / name).write_text("")
    assert find_walkers(tmp_path, "COLVAR") == [0, 2]
    assert find_walkers(tmp_path, "model_devi.out") == [0, 2]
    assert walker_file(tmp_path, "COLVAR", 2).name == "COLVAR.2"
    assert walker_file(tmp_path, "model_devi.out", 2).name == "model_devi.2.out"
    assert walker_file(tmp_path, "COLVAR", None).name == "COLVAR"
    assert walker_file(tmp_path, "missing.dat", 1).name == "missing.dat.1"  # for the error


def test_shared_bias_pools_the_walkers(root):
    runs = Project(root / "estk.yaml").runs
    assert [r.name for r in runs] == ["sim1"]
    run = runs[0]
    assert run.walkers == (0, 1, 2) and run.metadata.walkers_mpi
    # settings come from the first walker's log (log.lammps only has the partition summary)
    assert run.metadata.temperature == 350 and run.metadata.timestep == 0.0005
    table = run.load_cvs(["d0", "cn"], extra={"max_devi_f": "model_devi.out"})
    assert len(table) == WALKERS * N  # walker 1's repeated frames removed
    assert table.report.duplicates == 50
    np.testing.assert_array_equal(np.bincount(table.walker), [N] * WALKERS)
    # each walker's own model deviation file, joined on its own time
    for i in range(WALKERS):
        np.testing.assert_allclose(table["max_devi_f"][table.walker == i], 0.1 * (i + 1))
    window = run.load_cvs(["d0"], time=(0.01, 0.02))  # ns, in every walker
    assert len(window) == WALKERS * 201


def test_without_walkers_mpi_each_walker_is_a_run(tmp_path):
    write_walkers(tmp_path / "sim1", plumed=PLUMED_INPUT)
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=""))
    runs = Project(tmp_path / "estk.yaml").runs
    assert [r.name for r in runs] == ["sim1.w0", "sim1.w1", "sim1.w2"]
    assert len(runs[1].load_cvs(["d0"])) == N
    # forced either way
    for mode, names in (("pool", ["sim1"]), ("separate", ["sim1.w0", "sim1.w1", "sim1.w2"])):
        (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=f"  walkers: {mode}\n"))
        assert [r.name for r in Project(tmp_path / "estk.yaml").runs] == names


def test_pooled_surface_is_the_probability_combination(root):
    """Pooling walkers that share a bias = reweighting all their frames together."""
    pooled = Project(root / "estk.yaml").reweight(["d0", "cn"], points=(31, 21))
    (root / "estk.yaml").write_text(CONFIG.format(walkers="  walkers: separate\n"))
    separate = Project(root / "estk.yaml").reweight(["d0", "cn"], points=(31, 21))
    assert list(separate.surfaces) == ["sim1.w0", "sim1.w1", "sim1.w2"]
    a, b = pooled.surfaces["sim1"].values, separate.combined.values
    finite = np.isfinite(a)
    np.testing.assert_allclose(a[finite], b[finite], rtol=1e-9, atol=1e-9)


def test_blocks(root):
    project = Project(root / "estk.yaml")
    whole = project.reweight(["d0", "cn"], points=(31, 21)).surfaces["sim1"]
    result = project.reweight(["d0", "cn"], points=(31, 21), blocks=3)
    assert list(result.surfaces) == ["sim1.b1", "sim1.b2", "sim1.b3"]
    assert sum(s.metadata["frames"] for s in result.surfaces.values()) == WALKERS * N
    assert result.surfaces["sim1.b2"].metadata["block_of"] == "sim1"
    ranges = [s.metadata["time_range_ns"] for s in result.surfaces.values()]
    assert ranges[0][0] == 0 and ranges[-1][1] == pytest.approx((N - 1) * 0.05e-3)
    # the probability combination of the blocks is the whole run; their spread the error
    combined = result.combined
    finite = np.isfinite(whole.values)
    np.testing.assert_allclose(combined.values[finite], whole.values[finite], atol=1e-9)
    assert np.nanmax(combined.error) > 0
    with pytest.raises(EstError, match="at least 1"):
        project.reweight(["d0"], blocks=0)
    with pytest.raises(EstError, match="no frames"):
        project.reweight(["d0"], blocks=10_000)


def test_timeseries_of_walkers(root):
    run = Project(root / "estk.yaml").runs[0]
    ts = read_timeseries(run, ["opes.rct"])
    assert len(ts) == WALKERS * N
    segments = ts.segments()
    assert [w for w, _ in segments] == [0, 1, 2]
    first = ts.select(segments[0][1])
    np.testing.assert_allclose(first.values["opes.rct"], np.sin(first.time * 1000), atol=1e-8)
    summary = opes_summary(ts)  # the first walker: with a shared bias they agree
    assert summary.final["opes.rct"] == pytest.approx(np.sin((N - 1) * 0.05))
    devi = read_timeseries(run, ["max_devi_f"], file="model_devi.out")
    assert len(devi) == WALKERS * N and devi.source.name == "model_devi.0.out"


def test_commands(root, monkeypatch):
    monkeypatch.chdir(root)
    runner = CliRunner()
    result = runner.invoke(app, ["runs"])
    assert result.exit_code == 0, result.output
    assert "3 pooled" in result.output
    result = runner.invoke(app, ["cvs", "d0", "cn"])
    assert result.exit_code == 0 and str(WALKERS * N) in result.output

    result = runner.invoke(app, ["check", "model-deviation", "--levels", "0.15,0.25"])
    assert result.exit_code == 0, result.output
    for text in ("sim1.w0", "sim1.w2", "sim1 (all)", "33.3 %"):
        assert text in result.output
    result = runner.invoke(app, ["check", "opes"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["reweight", "d0", "cn", "--points", "21", "--blocks", "2",
                                 "-o", "out"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "2 blocks per run" in result.output
    assert sorted(p.name for p in (root / "out").iterdir()) == [
        "combined.dat", "sim1.b1.dat", "sim1.b2.dat",
    ]  # fmt: skip
    assert Surface.load(root / "out" / "combined.dat").metadata["runs"] == ["sim1.b1", "sim1.b2"]

    # a forced pool without WALKERS_MPI is flagged
    (root / "sim1" / "plumed.plumed").write_text(PLUMED_INPUT)
    (root / "estk.yaml").write_text(CONFIG.format(walkers="  walkers: pool\n"))
    result = runner.invoke(app, ["runs"])
    assert "no WALKERS_MPI" in result.output


def test_init_reads_the_first_walker(tmp_path, monkeypatch):
    write_walkers(tmp_path / "sim1")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["init", "--pattern", "sim*"])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "estk.yaml").read_text()
    assert "  d0:" in text and "  cn:" in text


def test_plain_colvar_wins_over_stray_numbered_files(tmp_path):
    write_walkers(tmp_path / "sim1")
    (tmp_path / "sim1" / "COLVAR").write_text((tmp_path / "sim1" / "COLVAR.0").read_text())
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=""))
    runs = Project(tmp_path / "estk.yaml").runs
    assert [r.name for r in runs] == ["sim1"] and runs[0].walkers is None
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers="  walkers: pool\n"))
    assert Project(tmp_path / "estk.yaml").runs[0].walkers == (0, 1, 2)
