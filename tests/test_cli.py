from typer.testing import CliRunner

from estk import __version__
from estk.cli.main import app
from estk.config import load_config

from .conftest import DATA_DIR, write_colvar

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
