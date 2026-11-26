"""Surface and path files: provenance, compression, estk.load and estk info."""

import gzip

import numpy as np
import pytest
from typer.testing import CliRunner

import estk
from estk.cli.main import app
from estk.errors import EstError
from estk.grid import Grid
from estk.io.textfile import base_name, set_command, write_atomic
from estk.paths import MinimumEnergyPath, minimum_energy_path
from estk.surface import Surface

from .test_combine import CONFIG, frames, write_run


@pytest.fixture
def surface():
    grid = Grid(("x", "y"), (-1.0, -1.0), (1.0, 1.0), (41, 31))
    X, Y = grid.mesh()
    values = (X**2 - 0.5) ** 2 * 40 + 10 * Y**2  # two minima at x = +-0.71
    return Surface(grid, values - values.min(), metadata={"kbt": 2.5, "run": "000"})


@pytest.fixture(autouse=True)
def no_command():
    yield
    set_command(None)


def test_compressed_surface_and_path(surface, tmp_path):
    saved = surface.save(tmp_path / "000.dat.gz")
    with open(saved, "rb") as f:
        assert f.read(2) == b"\x1f\x8b"  # gzip
    with gzip.open(saved, "rt") as f:
        assert f.readline() == "#! FIELDS x y fes\n"
    back = estk.load(saved)
    assert isinstance(back, Surface) and back.grid == surface.grid
    np.testing.assert_allclose(back.values, surface.values, rtol=1e-9)  # 10 digits in text

    path = minimum_energy_path(surface, images=11, max_steps=50)
    loaded = estk.load(path.save(tmp_path / "p.mep.dat.gz"))
    assert isinstance(loaded, MinimumEnergyPath)
    np.testing.assert_allclose(loaded.energies, path.energies)
    # non-finite metadata (the global minimum has infinite depth) comes back as a number
    assert loaded.metadata["minima"][0]["depth"] == float("inf")


def test_provenance_is_that_of_the_last_save(surface, tmp_path):
    set_command(["estk", "reweight", "x", "y"])
    first = surface.save(tmp_path / "a.dat")
    text = first.read_text()
    assert f'#! SET estk.version "{estk.__version__}"' in text
    assert '#! SET estk.command "estk reweight x y"' in text
    loaded = Surface.load(first)
    assert loaded.metadata["command"] == "estk reweight x y"

    set_command(None)  # saved again from Python: the old command must not be kept
    again = Surface.load(loaded.save(tmp_path / "b.dat"))
    assert "command" not in again.metadata
    assert again.metadata["version"] == estk.__version__


def test_load_errors(surface, tmp_path):
    surface_file = surface.save(tmp_path / "s.dat")
    with pytest.raises(EstError, match="not a path file"):
        MinimumEnergyPath.load(surface_file)
    (tmp_path / "none.dat").write_text("1 2 3\n")
    with pytest.raises(EstError, match="no FIELDS"):
        estk.load(tmp_path / "none.dat")
    with pytest.raises(EstError, match="not found"):
        estk.load(tmp_path / "missing.dat")


def test_base_name(tmp_path):
    assert base_name(tmp_path / "combined.dat.gz") == "combined"
    assert base_name(tmp_path / "combined.dat") == "combined"
    assert base_name(tmp_path / "combined.mep.dat") == "combined.mep"


@pytest.fixture(scope="module")
def project_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("files")
    write_run(root / "a", *frames(0, n=1500))
    write_run(root / "b", *frames(1, n=1500), t0=10.0)
    (root / "estk.yaml").write_text(CONFIG.format(pattern="*"))
    return root


def test_compressed_reweight_mep_and_info(project_root, monkeypatch):
    monkeypatch.chdir(project_root)
    runner = CliRunner()
    result = runner.invoke(app, ["reweight", "d0", "cn", "--points", "41", "--compress",
                                 "-o", "out"])  # fmt: skip
    assert result.exit_code == 0, result.output
    names = sorted(p.name for p in (project_root / "out").iterdir())
    assert names == ["a.dat.gz", "b.dat.gz", "combined.dat.gz"]

    combined = Surface.load(project_root / "out" / "combined.dat.gz")
    assert combined.metadata["run_time_range_ns"] == {"a": [0.0, 0.07495], "b": [0.01, 0.08495]}
    assert combined.metadata["run_bandwidth"]["a"] == {"d0": 0.03, "cn": 0.05}

    # the per-run files next to a compressed combined surface are found
    result = runner.invoke(app, ["mep", "out/combined.dat.gz"])
    assert result.exit_code == 0, result.output
    path = MinimumEnergyPath.load(project_root / "out" / "combined.mep.dat")
    assert sorted(path.profiles) == ["a", "b"]

    # combining again skips the combined file whatever its suffix
    result = runner.invoke(app, ["combine", "out/a.dat.gz", "out/b.dat.gz",
                                 "out/combined.dat.gz", "-m", "equal", "-o", "eq.dat"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert Surface.load(project_root / "eq.dat").metadata["runs"] == ["a", "b"]

    result = runner.invoke(app, ["info", "out/combined.dat.gz", "out/combined.mep.dat"])
    assert result.exit_code == 0, result.output
    for text in ("free energy, 2-D grid", "41", "run_weights", "path on d0, cn", "barrier"):
        assert text in result.output

    result = runner.invoke(app, ["info", "missing.dat"])
    assert result.exit_code == 1


def test_grid_read_from_the_header(tmp_path):
    """estk's grid files are read on the grid of their header: printing noise in the
    coordinates does not matter, a damaged line is named."""
    grid = Grid(("d0", "cn"), (1.5, 0.7), (2.5, 2.2), (300, 300))
    values = np.random.default_rng(0).random(grid.shape)
    path = Surface(grid, values, metadata={"kbt": 2.9}).save(tmp_path / "s.dat")
    lines = path.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line and not line.startswith("#"))

    noisy = list(lines)
    x, *rest = noisy[first + 7].split()
    noisy[first + 7] = " ".join([repr(float(x) * (1 + 1e-12)), *rest])  # last-digit noise
    (tmp_path / "noisy.dat").write_text("\n".join(noisy) + "\n")
    back = Surface.load(tmp_path / "noisy.dat")
    assert back.grid == grid
    np.testing.assert_allclose(back.values, values, rtol=1e-9)

    damaged = list(lines)
    x, *rest = damaged[first + 7].split()
    damaged[first + 7] = " ".join([str(float(x) + 0.0011), *rest])  # off the grid
    (tmp_path / "damaged.dat").write_text("\n".join(damaged) + "\n")
    with pytest.raises(EstError, match="grid line 8 is not a point of the grid"):
        Surface.load(tmp_path / "damaged.dat")

    twice = [*lines, lines[first]]  # one line written twice, as by an interrupted rewrite
    (tmp_path / "twice.dat").write_text("\n".join(twice) + "\n")
    with pytest.raises(EstError, match="more than once"):
        Surface.load(tmp_path / "twice.dat")


def test_atomic_writes_and_damaged_files(tmp_path):
    """Files are written under a temporary name and renamed: no partial or mixed files are
    left; a file with a broken line is reported by its line number."""
    target = tmp_path / "x.dat"
    target.write_text("old\n")
    with pytest.raises(RuntimeError):
        with write_atomic(target) as f:
            f.write("half")
            raise RuntimeError("interrupted")
    assert target.read_text() == "old\n"  # the old file is untouched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.dat"]  # no temporary left
    with write_atomic(tmp_path / "y.dat.gz") as f:
        f.write("1 2\n")
    assert gzip.open(tmp_path / "y.dat.gz", "rt").read() == "1 2\n"

    grid = Grid(("x", "y"), (0.0, 0.0), (1.0, 1.0), (5, 4))
    path = Surface(grid, np.zeros(grid.shape)).save(tmp_path / "s.dat")
    lines = path.read_text().splitlines()
    (tmp_path / "bad.dat").write_text("\n".join([*lines, "3.499764899"]) + "\n")
    with pytest.raises(EstError, match=r"line \d+ has 1 column\(s\) instead of 3"):
        Surface.load(tmp_path / "bad.dat")
