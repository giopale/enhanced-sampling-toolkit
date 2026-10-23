from typer.testing import CliRunner

from estk import __version__
from estk.cli.main import app
from estk.config import load_config

from .conftest import DATA_DIR, make_run, write_colvar

runner = CliRunner()


def run(*args):
    return runner.invoke(app, list(args))


def test_version():
    result = run("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_init_without_runs(workdir):
    result = run("init")
    assert result.exit_code == 0, result.output
    cfg = load_config(workdir / "estk.yaml")
    assert cfg.cv == {}


def test_init_discovers_cvs(workdir):
    write_colvar(workdir / "000", ["time", "d0", "cn", "opes.bias", "opes.rct", "uwall.bias"])
    write_colvar(workdir / "001", ["time", "other"])
    result = run("init", "--pattern", "00*")
    assert result.exit_code == 0, result.output
    cfg = load_config(workdir / "estk.yaml")
    assert list(cfg.cv) == ["d0", "cn"]
    assert cfg.runs.pattern == "00*"


def test_init_refuses_to_overwrite(workdir):
    (workdir / "estk.yaml").write_text("# mine\n")
    result = run("init")
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert (workdir / "estk.yaml").read_text() == "# mine\n"

    assert run("init", "--force").exit_code == 0
    assert "runs:" in (workdir / "estk.yaml").read_text()


def test_config_validate(workdir):
    assert run("config", "validate").exit_code == 1  # no file

    (workdir / "estk.yaml").write_text("cv:\n  d0:\n    limits: [1.0, 2.0]\n")
    result = run("config", "validate")
    assert result.exit_code == 0
    assert "OK" in result.output

    (workdir / "estk.yaml").write_text("cv:\n  d0:\n    limts: [1.0, 2.0]\n")
    result = run("config", "validate")
    assert result.exit_code == 1
    assert "cv.d0.limts: unknown setting" in result.output


def test_config_option(workdir):
    (workdir / "other.yaml").write_text("runs:\n  pattern: 'job-*'\n")
    result = run("--config", "other.yaml", "config", "show")
    assert result.exit_code == 0, result.output
    assert "job-*" in result.output


def test_config_show_defaults(workdir):
    result = run("config", "show")
    assert result.exit_code == 0, result.output
    assert "using defaults" in result.output
    assert "files.colvar" in result.output


def test_config_migrate(workdir):
    result = run("config", "migrate", str(DATA_DIR / "case_minimal.yaml"), "-o", "estk.yaml")
    assert result.exit_code == 0, result.output
    assert "not migrated" in result.output
    assert load_config(workdir / "estk.yaml").cv["d0"].bandwidth == 0.02161725

    result = run("config", "migrate", str(DATA_DIR / "case_minimal.yaml"), "-o", "estk.yaml")
    assert result.exit_code == 1  # refuses to overwrite


def test_config_migrate_to_stdout(workdir):
    result = run("config", "migrate", str(DATA_DIR / "case_minimal.yaml"))
    assert result.exit_code == 0, result.output
    assert "  d0:" in result.output


def test_runs(workdir):
    make_run(workdir / "000")
    make_run(workdir / "001")
    (workdir / "estk.yaml").write_text("runs:\n  pattern: '0*'\nsimulation:\n  temperature: 300\n")
    result = run("runs")
    assert result.exit_code == 0, result.output
    assert "000" in result.output and "001" in result.output
    assert "opes.bias" in result.output
    assert "set in estk.yaml: temperature" in result.output


def test_runs_none_found(workdir):
    result = run("runs")
    assert result.exit_code == 1
    assert "no run folders" in result.output


def test_cvs(workdir):
    make_run(workdir / "000")
    result = run("cvs", "d0", "cn", "--time", "0", "0.0002")
    assert result.exit_code == 0, result.output
    assert "41" in result.output
    assert "1.5 – 1.9" in result.output


def test_cvs_unknown(workdir):
    make_run(workdir / "000")
    result = run("cvs", "nope")
    assert result.exit_code == 1
    assert "CV-nope" in result.output


def test_reweight(workdir):
    make_run(workdir / "000")
    make_run(workdir / "001")
    (workdir / "estk.yaml").write_text("runs:\n  pattern: '0*'\n")
    result = run(
        "reweight", "d0", "cn", "--points", "11,9", "--limits", "cn=0,", "--bandwidth", "d0=0.05"
    )
    assert result.exit_code == 0, result.output
    assert "grid cn: 0 (command line) to 3 (PLUMED walls), 9 points" in result.output
    from estk.surface import Surface

    surface = Surface.load(workdir / "reweight-d0-cn" / "000.dat")
    assert surface.grid.points == (11, 9)
    assert surface.metadata["run"] == "000"
    assert surface.metadata["bandwidth"]["d0"] == 0.05
    assert surface.metadata["bandwidth_source"]["d0"] == "command line"


def test_reweight_bad_options(workdir):
    make_run(workdir / "000")
    result = run("reweight", "d0", "cn", "--limits", "cn=0.5")
    assert result.exit_code == 1
    assert "--limits: expected e.g. d0=1.5,2.5" in result.output
    result = run("reweight", "d0", "cn", "--points", "ten")
    assert result.exit_code == 1
    assert "--points" in result.output


def test_reweight_combines_and_combine_command(workdir):
    make_run(workdir / "000")
    make_run(workdir / "001")
    (workdir / "estk.yaml").write_text("runs:\n  pattern: '0*'\n")
    result = run("reweight", "d0", "cn", "--points", "6,5")
    assert result.exit_code == 0, result.output
    assert "runs combined with mode 'probability'" in result.output
    from estk.surface import Surface

    combined = Surface.load(workdir / "reweight-d0-cn" / "combined.dat")
    assert combined.metadata["combined"] == "probability"

    files = [str(p) for p in sorted((workdir / "reweight-d0-cn").glob("*.dat"))]
    result = run("combine", *files, "--mode", "equal", "-o", "equal.dat")
    assert result.exit_code == 0, result.output
    assert Surface.load(workdir / "equal.dat").metadata["run_weights"] == {"000": 0.5, "001": 0.5}
    assert run("combine", *files, "-o", "equal.dat").exit_code == 1  # exists
    assert "unknown mode" in run("reweight", "d0", "cn", "--combine", "mean").output

    result = run("reweight", "d0", "cn", "--points", "6,5", "--combine", "none", "-o", "nocomb")
    assert result.exit_code == 0, result.output
    assert not (workdir / "nocomb" / "combined.dat").exists()
