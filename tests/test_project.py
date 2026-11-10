import numpy as np
import pytest

from estk.config import load_config
from estk.errors import EstError
from estk.io.colvar import read_table
from estk.project import Project

from .conftest import make_run


def project_with_runs(root, names=("000", "001"), config_text="runs:\n  pattern: '0*'\n"):
    for name in names:
        make_run(root / name)
    (root / "estk.yaml").write_text(config_text)
    return Project(root / "estk.yaml")


# --- discovery and metadata --------------------------------------------------------------------


def test_runs_discovered_in_natural_order(tmp_path):
    for name in ("run-10", "run-2", "run-1"):
        make_run(tmp_path / name)
    (tmp_path / "run-3").mkdir()  # no COLVAR: skipped
    (tmp_path / "estk.yaml").write_text("runs:\n  pattern: 'run-*'\n")
    project = Project(tmp_path / "estk.yaml")
    assert [r.name for r in project.runs] == ["run-1", "run-2", "run-10"]
    assert project.run("run-2").path == tmp_path / "run-2"
    with pytest.raises(EstError, match="no run named"):
        project.run("run-3")


def test_no_runs(tmp_path):
    (tmp_path / "estk.yaml").write_text("runs:\n  pattern: 'x*'\n")
    with pytest.raises(EstError, match="no run folders matching 'x\\*'"):
        _ = Project(tmp_path / "estk.yaml").runs


def test_metadata(tmp_path):
    meta = project_with_runs(tmp_path).runs[0].metadata
    assert meta.temperature == 350.0
    assert meta.timestep == 0.0005
    assert meta.bias_column == "opes.bias"
    assert meta.bias_factor == 10.0
    assert meta.walls == {"d0": (1.5, 2.5), "cn": (None, 3.0)}
    assert meta.kbt == pytest.approx(350 * 8.314462618e-3)
    assert meta.sources["temperature"] == "log.lammps"


def test_metadata_overrides(tmp_path):
    config = "runs:\n  pattern: '0*'\nsimulation:\n  temperature: 300\n  bias_factor: 20\n"
    meta = project_with_runs(tmp_path, config_text=config).runs[0].metadata
    assert meta.temperature == 300.0
    assert meta.bias_factor == 20.0
    assert meta.sources["temperature"] == "estk.yaml"


def test_metadata_missing_files(tmp_path):
    project = project_with_runs(tmp_path, names=("000",))
    (tmp_path / "000" / "log.lammps").unlink()
    meta = project.runs[0].metadata
    assert meta.temperature is None
    with pytest.raises(EstError, match="temperature unknown"):
        _ = meta.kbt


# --- reading tables ----------------------------------------------------------------------------


def test_read_table_skips_bad_lines(tmp_path):
    path = tmp_path / "COLVAR"
    path.write_text("#! FIELDS time a b\n0 1 2\n1 3 4\n#! FIELDS time a b\n2 5 6\n3 7\n")
    table = read_table(path)
    assert table.fields == ["time", "a", "b"]
    assert table.data.shape == (3, 3)
    assert table.bad_lines == 1


def test_read_table_plain_header(tmp_path):
    path = tmp_path / "model_devi.out"
    path.write_text("#       step         max_devi_v         max_devi_f\n0 0.1 0.2\n10 0.3 0.4\n")
    table = read_table(path)
    assert table.fields == ["step", "max_devi_v", "max_devi_f"]
    np.testing.assert_allclose(table.column("max_devi_f"), [0.2, 0.4])


# --- loading CV tables -------------------------------------------------------------------------


def test_load_cvs_from_colvar(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    data = run.load_cvs(["d0", "cn"])
    assert len(data) == 100
    assert data.names == ["d0", "cn"]
    np.testing.assert_allclose(data.time[:3], [0.0, 0.005, 0.01])
    np.testing.assert_allclose(data.bias[:3], [0.0, 0.1, 0.2])
    assert data.array().shape == (100, 2)
    assert data.kbt == pytest.approx(2.91, abs=0.01)
    assert data.report.sources["d0"] == run.colvar_path


def test_restarts_keep_last_occurrence(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    with open(run.colvar_path, "a") as f:
        f.write("#! FIELDS time d0 cn opes.bias\n0.490000 9.0 9.0 9.0\n0.495000 8.0 8.0 8.0\n")
    data = run.load_cvs(["d0"])
    assert len(data) == 100
    assert data.report.duplicates == 2
    assert data["d0"][-2:].tolist() == [9.0, 8.0]
    assert np.all(np.diff(data.time) > 0)


def test_cv_from_separate_file_joined_on_time(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    # PLUMED-driver output with every 10th frame missing and times printed differently.
    lines = [f"{i * 0.005:.4f} {i * 1.0}" for i in range(100) if i % 10 != 3]
    (run.path / "CV-dOH").write_text("#! FIELDS time dOH.min\n" + "\n".join(lines) + "\n")
    data = run.load_cvs(["d0", "dOH.min"])
    assert len(data) == 90
    assert data.report.missing == {"CV-dOH": 10}
    # each value still belongs to its own frame
    np.testing.assert_allclose(data["dOH.min"], np.rint(data.time / 0.005))
    np.testing.assert_allclose(data["d0"], 1.5 + 0.01 * np.rint(data.time / 0.005))


def test_cv_source_and_scale_from_config(tmp_path):
    config = "runs:\n  pattern: '0*'\ncv:\n  theta:\n    source: angles.dat\n    scale: 2.0\n"
    run = project_with_runs(tmp_path, config_text=config).runs[0]
    lines = [f"{i * 0.005:.6f} {i * 0.1:.6f} 7" for i in range(100)]
    (run.path / "angles.dat").write_text("#! FIELDS time theta other\n" + "\n".join(lines) + "\n")
    data = run.load_cvs(["theta"])
    np.testing.assert_allclose(data["theta"][:3], [0.0, 0.2, 0.4])


def test_ambiguous_column(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    (run.path / "CV-x").write_text("#! FIELDS time a b\n0 1 2\n")
    with pytest.raises(EstError, match="more than one data column"):
        run.load_cvs(["x"])


def test_extra_property_indexed_by_step(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    # model deviation every 10 MD steps = every 0.005 ps, like COLVAR
    lines = [f"{i * 10} 0.0 {i * 0.01:.4f}" for i in range(50)]
    (run.path / "model_devi.out").write_text(
        "#       step         max_devi_v         max_devi_f\n" + "\n".join(lines) + "\n"
    )
    data = run.load_cvs(["d0"], extra={"max_devi_f": "model_devi.out"})
    assert len(data) == 50
    np.testing.assert_allclose(data["max_devi_f"], np.rint(data.time / 0.005) * 0.01)


def test_time_window_nan_and_filters(tmp_path):
    run = project_with_runs(tmp_path).runs[0]
    lines = [f"{i * 0.005:.6f} {'nan' if i == 5 else i}" for i in range(100)]
    (run.path / "CV-q").write_text("#! FIELDS time q\n" + "\n".join(lines) + "\n")

    data = run.load_cvs(["d0", "q"], time=(0.0001, None), filters={"cn": (0.5, None)})
    # time >= 0.1 ps keeps rows 20..99 (80 rows); q is NaN in row 5 (already outside)
    assert data.report.outside_time == 20
    assert data.report.nan == {}
    assert np.all(data.time >= 0.1)
    assert "cn" not in data.names  # loaded only for filtering
    # cn = (i % 7) * 0.5 >= 0.5 drops rows with i % 7 == 0
    expected = [i for i in range(20, 100) if i % 7 != 0]
    assert len(data) == len(expected)
    assert data.report.filtered == 80 - len(expected)

    data = run.load_cvs(["q"])
    assert data.report.nan == {"q": 1}
    assert len(data) == 99


def test_bias_problems(tmp_path):
    project = project_with_runs(tmp_path, names=("000",))
    run = project.runs[0]
    (run.path / "plumed.plumed").write_text("PRINT FILE=COLVAR ARG=d0\n")
    with pytest.raises(EstError, match="no bias found"):
        run.load_cvs(["d0"])
    assert len(run.load_cvs(["d0"], bias=False)) == 100

    config = load_config(tmp_path / "estk.yaml")
    config.simulation.bias_column = "metad.bias"
    with pytest.raises(EstError, match="no bias column 'metad.bias'"):
        Project(config).runs[0].load_cvs(["d0"])


def test_project_load_cvs(tmp_path):
    tables = project_with_runs(tmp_path).load_cvs(["d0", "cn"], time=(None, 0.0002))
    assert [t.run_name for t in tables] == ["000", "001"]
    assert all(len(t) == 41 for t in tables)
