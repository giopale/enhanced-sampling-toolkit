import subprocess
import sys
from pathlib import Path

import pytest

from estk.config import (
    ProjectConfig,
    config_template,
    find_config,
    is_bias_column,
    load_config,
    read_colvar_fields,
)
from estk.errors import ConfigError

from .conftest import write_colvar

VALID = """
[runs]
root = "runs"
pattern = "00*"

[cv.d0]
label = "d(Ti-O)"
limits = [1.5, 2.5]
bandwidth = 0.02

[cv.cn]
min = 0.75

[atoms]
type_map = { 1 = "H", 2 = "O", 3 = "Ti" }
groups = { site_A = 311, site_B = [313, 218] }
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_no_file_gives_defaults(workdir):
    cfg = load_config()
    assert cfg.path is None
    assert cfg.runs.pattern == "*"
    assert cfg.files.colvar == "COLVAR"
    assert cfg.cv == {}
    assert cfg.runs_root == workdir.resolve()


def test_valid_file(workdir):
    cfg = load_config(write(workdir / "estk.toml", VALID))
    assert cfg.cv["d0"].limits == (1.5, 2.5)
    assert cfg.cv["d0"].bandwidth == 0.02
    assert cfg.cv["cn"].limits == (0.75, None)
    assert cfg.atoms.type_map == {1: "H", 2: "O", 3: "Ti"}
    assert cfg.atoms.groups == {"site_A": [311], "site_B": [313, 218]}


def test_relative_root_resolved_against_file(workdir, monkeypatch):
    (workdir / "project").mkdir()
    path = write(workdir / "project" / "estk.toml", VALID)
    monkeypatch.chdir(Path("/"))
    cfg = load_config(path)
    assert cfg.runs_root == (workdir / "project" / "runs").resolve()


def test_find_config_walks_up(workdir, monkeypatch):
    write(workdir / "estk.toml", VALID)
    sub = workdir / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert find_config() == (workdir / "estk.toml").resolve()
    assert load_config().cv["d0"].bandwidth == 0.02


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[runs]\npatern = '00*'\n", "runs.patern: unknown setting"),
        ("[cv.d0]\nlimits = [2.5, 1.5]\n", "cv.d0: min (2.5) must be smaller than max (1.5)"),
        ("[cv.d0]\nlimits = [1, 2, 3]\n", "cv.d0: 'limits' must be a list of two numbers"),
        ("[cv.d0]\nlimits = [1, 2]\nmin = 0\n", "use either 'limits' or 'min'/'max'"),
        ("[cv.d0]\nbandwidth = -1\n", "cv.d0.bandwidth: Input should be greater than 0"),
        ("[atoms]\ngroups = { a = [0, 1] }\n", "atom indices are 1-based"),
        ("[unknown]\nx = 1\n", "unknown: unknown setting"),
    ],
)
def test_invalid_settings(workdir, text, expected):
    path = write(workdir / "estk.toml", text)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert expected in str(exc.value)
    assert str(path) in str(exc.value)


def test_toml_syntax_error(workdir):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(workdir / "estk.toml", "[runs\n"))


def test_explicit_missing_file(workdir):
    with pytest.raises(ConfigError, match="not found"):
        load_config(workdir / "nope.toml")


def test_explain_sources(workdir):
    cfg = load_config(write(workdir / "estk.toml", VALID + "\n[cv.empty]\n"))
    rows = {key: (value, source) for key, value, source in cfg.explain()}
    assert rows["runs.pattern"] == ("00*", "estk.toml")
    assert rows["files.colvar"] == ("COLVAR", "default")
    assert rows["cv.d0.bandwidth"] == (0.02, "estk.toml")
    assert rows["cv.cn.min"] == (0.75, "estk.toml")
    assert "cv.cn.max" not in rows  # only set fields are listed for CVs
    assert rows["cv.empty"] == ("(all defaults)", "estk.toml")


@pytest.mark.parametrize("cvs", [None, ["d0", "cn", "z.min"]])
def test_template_is_valid(workdir, cvs):
    path = write(workdir / "estk.toml", config_template(pattern="00*", cvs=cvs))
    cfg = load_config(path)
    assert cfg.runs.pattern == "00*"
    assert list(cfg.cv) == (cvs or [])


def test_read_colvar_fields(workdir):
    path = write_colvar(workdir, ["time", "d0", "opes.bias"])
    assert read_colvar_fields(path) == ["time", "d0", "opes.bias"]
    assert read_colvar_fields(write(workdir / "plain", "1 2 3\n")) == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("time", True),
        ("opes.bias", True),
        ("opes.rct", True),
        ("uwall.bias", True),
        ("metad.rbias", True),
        ("d0", False),
        ("z.min", False),
    ],
)
def test_is_bias_column(name, expected):
    assert is_bias_column(name) is expected


def test_library_import_does_not_touch_logging():
    code = (
        "import logging, estk, estk.cli.main\n"
        "assert not logging.getLogger().handlers\n"
        "assert not logging.getLogger('estk').handlers\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_defaults_are_independent():
    a, b = ProjectConfig(), ProjectConfig()
    a.atoms.type_map[1] = "H"
    assert b.atoms.type_map == {}
