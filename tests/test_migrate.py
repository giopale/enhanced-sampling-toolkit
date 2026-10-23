import pytest

from estk.config import load_config
from estk.errors import ConfigError
from estk.migrate import migrate_case_yaml

from .conftest import DATA_DIR, REPO_ROOT


def migrate_and_load(src, tmp_path):
    text, dropped = migrate_case_yaml(src)
    out = tmp_path / "estk.toml"
    out.write_text(text)
    return load_config(out), dropped


def test_minimal_case(tmp_path):
    cfg, dropped = migrate_and_load(DATA_DIR / "case_minimal.yaml", tmp_path)

    assert cfg.runs.pattern == "00*"
    assert cfg.cv["d0"].limits == (1.5, 2.5)
    assert cfg.cv["d0"].bandwidth == 0.02161725
    assert cfg.cv["cn"].limits == (0.75, None)
    assert cfg.cv["theta1"].scale == pytest.approx(57.29577951308232)
    assert cfg.cv["indirect-delta"].limits == (-1.5, 1.5)  # YAML anchor resolved
    assert cfg.cv["rmse-f"].limits == (None, None)
    assert cfg.atoms.type_map == {1: "H", 2: "O", 3: "Ti"}
    assert cfg.atoms.groups == {
        "pt_type1": [262],
        "water_oxygens": [153, 154, 155],
        "site_A": [311],
        "site_B": [313, 218],
        "site_liquid": [153, 154, 155],
    }

    keys = [item.split(" -> ")[0] for item in dropped]
    assert set(keys) == {
        "cvs.select",
        "cvs.d0.color",
        "mplstyle",
        "temp",
        "find_structures",
        "proton_transfer",
    }


@pytest.mark.parametrize("name", ["atio2-h2o.yaml", "atio2.yaml", "atio2-h2o.template.yaml"])
def test_real_case_files(tmp_path, name):
    src = REPO_ROOT / "to_refactor" / "config" / "case" / name
    if not src.exists():
        pytest.skip(f"{src} not present")
    cfg, dropped = migrate_and_load(src, tmp_path)
    assert cfg.cv
    assert any(item.startswith("surface_data") for item in dropped)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        migrate_case_yaml(tmp_path / "nope.yaml")


def test_not_a_mapping(tmp_path):
    src = tmp_path / "list.yaml"
    src.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        migrate_case_yaml(src)
