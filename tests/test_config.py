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
runs:
  root: runs
  pattern: "00*"

cv:
  d0:
    label: d(Ti-O)
    limits: [1.5, 2.5]
    bandwidth: 0.02
  cn:
    limits: [0.75, null]
  theta:
    min: 1.0

atoms:
  type_map: {1: H, 2: O, 3: Ti}
  groups:
    water: &water [153, 154]
    site_A: 311
    site_B: [313, 218]
    site_liquid: *water
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
    cfg = load_config(write(workdir / "estk.yaml", VALID))
    assert cfg.cv["d0"].limits == (1.5, 2.5)
    assert cfg.cv["d0"].bandwidth == 0.02
    assert cfg.cv["cn"].limits == (0.75, None)
    assert cfg.cv["theta"].limits == (1.0, None)
    assert cfg.atoms.type_map == {1: "H", 2: "O", 3: "Ti"}
    assert cfg.atoms.groups == {
        "water": [153, 154],
        "site_A": [311],
        "site_B": [313, 218],
        "site_liquid": [153, 154],
    }


@pytest.mark.parametrize("text", ["", "# only comments\n", "cv:\natoms:\n", "cv:\n  d0:\n"])
def test_empty_file_and_sections(workdir, text):
    cfg = load_config(write(workdir / "estk.yaml", text))
    assert cfg.atoms.type_map == {}
    assert set(cfg.cv) <= {"d0"}


def test_relative_root_resolved_against_file(workdir, monkeypatch):
    (workdir / "project").mkdir()
    path = write(workdir / "project" / "estk.yaml", VALID)
    monkeypatch.chdir(Path("/"))
    cfg = load_config(path)
    assert cfg.runs_root == (workdir / "project" / "runs").resolve()


def test_find_config_walks_up(workdir, monkeypatch):
    write(workdir / "estk.yaml", VALID)
    sub = workdir / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert find_config() == (workdir / "estk.yaml").resolve()
    assert load_config().cv["d0"].bandwidth == 0.02


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("runs:\n  patern: '00*'\n", "runs.patern: unknown setting"),
        ("cv:\n  d0:\n    limits: [2.5, 1.5]\n", "cv.d0: min (2.5) must be smaller than max"),
        ("cv:\n  d0:\n    limits: [1, 2, 3]\n", "cv.d0: 'limits' must be a list of two numbers"),
        ("cv:\n  d0:\n    limits: [1, 2]\n    min: 0\n", "use either 'limits' or 'min'/'max'"),
        ("cv:\n  d0:\n    bandwidth: -1\n", "cv.d0.bandwidth: Input should be greater than 0"),
        ("cv:\n  d0:\n    bandwidth: abc\n", "cv.d0.bandwidth: Input should be a valid number"),
        ("atoms:\n  groups: {a: [0, 1]}\n", "atom indices are 1-based"),
        ("atoms:\n  type_map: {1: no}\n", "atoms.type_map.1: Input should be a valid string"),
        ("unknown:\n  x: 1\n", "unknown: unknown setting"),
    ],
)
def test_invalid_settings(workdir, text, expected):
    path = write(workdir / "estk.yaml", text)
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert expected in str(exc.value)
    assert str(path) in str(exc.value)


def test_yaml_syntax_error(workdir):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(workdir / "estk.yaml", "runs: [\n"))


def test_duplicate_key_is_an_error(workdir):
    text = "cv:\n  d0:\n    bandwidth: 0.1\n  d0:\n    bandwidth: 0.2\n"
    with pytest.raises(ConfigError, match="duplicate key 'd0'"):
        load_config(write(workdir / "estk.yaml", text))


def test_top_level_must_be_mapping(workdir):
    with pytest.raises(ConfigError, match="top level"):
        load_config(write(workdir / "estk.yaml", "- a\n- b\n"))


def test_explicit_missing_file(workdir):
    with pytest.raises(ConfigError, match="not found"):
        load_config(workdir / "nope.yaml")


def test_explain_sources(workdir):
    cfg = load_config(write(workdir / "estk.yaml", VALID.replace("cv:\n", "cv:\n  empty:\n")))
    rows = {key: (value, source) for key, value, source in cfg.explain()}
    assert rows["runs.pattern"] == ("00*", "estk.yaml")
    assert rows["files.colvar"] == ("COLVAR", "default")
    assert rows["cv.d0.bandwidth"] == (0.02, "estk.yaml")
    assert rows["cv.cn.min"] == (0.75, "estk.yaml")
    assert "cv.theta.max" not in rows  # only set fields are listed for CVs
    assert rows["cv.empty"] == ("(all defaults)", "estk.yaml")


@pytest.mark.parametrize("cvs", [None, ["d0", "cn", "z.min", "no", "a:b"]])
def test_template_is_valid(workdir, cvs):
    path = write(workdir / "estk.yaml", config_template(pattern="00*", cvs=cvs))
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


def test_reaction_start(workdir):
    (workdir / "estk.yaml").write_text("reaction:\n  start: {cn: 2.0}\n")
    assert load_config().reaction.start == {"cn": 2.0}
    (workdir / "estk.yaml").write_text("reaction:\n  end: {cn: 1.0}\n")
    with pytest.raises(ConfigError):
        load_config()
    assert "# reaction:\n#   start: {cn: 2.0}" in config_template()
