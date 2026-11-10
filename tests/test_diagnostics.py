import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from estk.analysis import (  # noqa: E402
    deviation_stats,
    opes_columns,
    opes_summary,
    read_timeseries,
)
from estk.cli.main import app  # noqa: E402
from estk.errors import EstError  # noqa: E402
from estk.migrate import migrate_case_yaml  # noqa: E402
from estk.plotting.diagnostics import plot_model_deviation, plot_timeseries  # noqa: E402
from estk.project import Project  # noqa: E402

from .conftest import LAMMPS_LOG, PLUMED_INPUT  # noqa: E402

N = 400  # frames, every 0.005 ps (10 MD steps of 0.0005 ps)


def write_run(folder, k=0, *, restart=False):
    folder.mkdir(parents=True)
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)
    t = np.arange(N) * 0.005
    rct = 2.0 + k - np.exp(-t / 0.1) * 50  # start-up transient, then flat
    rows = [f"{ti:.6f} 1.8 1.0 0.5 {r:.6f} {0.01 + 1 / (i + 1):.6f} {i + 1} {i // 5}"
            for i, (ti, r) in enumerate(zip(t, rct, strict=True))]  # fmt: skip
    if restart:  # the last 10 frames written twice, the second time with other values
        rows += [r.replace(" 1.8 ", " 1.9 ") for r in rows[-10:]]
    (folder / "COLVAR").write_text(
        "#! FIELDS time d0 cn opes.bias opes.rct opes.zed opes.neff opes.nker\n"
        + "\n".join(rows) + "\n"
    )  # fmt: skip
    devi = 0.05 * (k + 1) * (1 + np.arange(N) % 10)  # 0.05 .. 0.5 for k = 0
    (folder / "model_devi.out").write_text(
        "#       step         max_devi_v         max_devi_f\n"
        + "\n".join(f"{i * 10:12d} 0.0 {d:.4f}" for i, d in enumerate(devi)) + "\n"
    )  # fmt: skip
    return folder


def _project(root):
    (root / "estk.yaml").write_text("runs:\n  pattern: '0*'\n")
    return Project(root / "estk.yaml")


@pytest.fixture
def project(tmp_path):
    write_run(tmp_path / "000", 0, restart=True)
    write_run(tmp_path / "001", 1)
    return _project(tmp_path)


def test_read_colvar_and_step_indexed_file(project):
    run = project.runs[0]
    ts = read_timeseries(run, ["opes.rct", "d0"])
    assert len(ts) == N  # the restart's repeated times are removed ...
    assert np.all(ts.values["d0"][-10:] == 1.9)  # ... keeping the last occurrence
    np.testing.assert_allclose(ts.time[1], 0.005e-3)  # ns

    devi = read_timeseries(run, ["max_devi_f"], file="model_devi.out")
    assert len(devi) == N
    np.testing.assert_allclose(devi.time, ts.time, atol=1e-12)  # step * timestep

    every = read_timeseries(run)
    assert "time" not in every.values and "opes.nker" in every.values
    with pytest.raises(EstError, match="no column"):
        read_timeseries(run, ["nope"])


def test_opes_columns_and_summary(project):
    run = project.runs[1]
    assert opes_columns(run) == ["opes.rct", "opes.zed", "opes.neff", "opes.nker"]
    summary = opes_summary(read_timeseries(run, opes_columns(run)))
    assert summary.final["opes.neff"] == N
    np.testing.assert_allclose(summary.final["opes.rct"], 3.0, atol=1e-3)
    assert abs(summary.rct_change) < 1e-3  # flat at the end
    np.testing.assert_allclose(summary.time, (N - 1) * 0.005e-3)


def test_opes_columns_missing(tmp_path):
    folder = tmp_path / "000"
    folder.mkdir()
    (folder / "log.lammps").write_text(LAMMPS_LOG)
    (folder / "plumed.plumed").write_text(PLUMED_INPUT)
    (folder / "COLVAR").write_text("#! FIELDS time d0 cn opes.bias\n0 1 1 0\n")
    run = _project(tmp_path).runs[0]
    with pytest.raises(EstError, match="PRINT ARG"):
        opes_columns(run)


def test_deviation_stats(project):
    ts = read_timeseries(project.runs[0], ["max_devi_f"], file="model_devi.out")
    stats = deviation_stats(ts, "max_devi_f", 0.2, 0.35)
    # values 0.05, 0.10, ..., 0.50 equally often
    assert stats.frames == N
    np.testing.assert_allclose([stats.below, stats.between, stats.above], [0.3, 0.3, 0.4])
    np.testing.assert_allclose(stats.maximum, 0.5)
    np.testing.assert_allclose(stats.mean, 0.275)
    with pytest.raises(EstError, match="increasing"):
        deviation_stats(ts, "max_devi_f", 0.35, 0.2)


def test_plots(project):
    series = {run.name: read_timeseries(run, opes_columns(run)) for run in project.runs}
    fig = plot_timeseries(series, ["opes.rct", "opes.zed"], log_columns=["opes.zed"])
    rct, zed = fig.axes
    assert len(rct.get_lines()) == 2 and zed.get_yscale() == "log"
    assert rct.get_ylim()[0] > -40  # the start-up transient does not set the range
    assert rct.get_legend() is not None

    devi = {r.name: read_timeseries(r, ["max_devi_f"], file="model_devi.out") for r in project.runs}
    fig = plot_model_deviation(devi, "max_devi_f", low=0.2, high=0.35)
    ax, hist = fig.axes
    assert ax.get_yscale() == "log"
    assert len(hist.patches) == 2  # one histogram per run
    with pytest.raises(EstError):
        plot_timeseries({}, ["opes.rct"])


def test_check_commands(project, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["check", "opes"])
    assert result.exit_code == 0, result.output
    assert "rct" in result.output and (tmp_path / "opes.png").is_file()

    result = runner.invoke(app, ["check", "model-deviation", "--levels", "0.2,0.35",
                                 "-o", "devi.pdf"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "30.0 %" in result.output and "40.0 %" in result.output
    assert (tmp_path / "devi.pdf").is_file()

    result = runner.invoke(app, ["check", "model-deviation", "--levels", "0.35,0.2"])
    assert result.exit_code == 1


def test_migrate_model_deviation_file(tmp_path):
    src = tmp_path / "case.yaml"
    src.write_text("dir_md: md\nfile_model_devi: devi.out\n")
    text, dropped = migrate_case_yaml(src)
    assert "model_deviation: devi.out" in text
    assert not any(item.startswith("file_model_devi") for item in dropped)
