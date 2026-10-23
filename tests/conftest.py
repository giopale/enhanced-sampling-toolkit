from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run the test inside an empty temporary folder."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_colvar(folder: Path, fields: list[str]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "COLVAR"
    path.write_text("#! FIELDS " + " ".join(fields) + "\n" + " ".join("0" for _ in fields) + "\n")
    return path
