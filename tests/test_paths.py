import numpy as np
import pytest

from estk.errors import EstError
from estk.grid import Grid
from estk.paths import (
    MinimumEnergyPath,
    evaluate,
    find_minima,
    initial_string,
    minimum_energy_path,
)
from estk.surface import Surface

from .conftest import DATA_DIR

# --- Müller-Brown potential: minima and saddles are known -------------------------------------

A = (-200, -100, -170, 15)
a = (-1, -1, -6.5, 0.7)
b = (0, 0, 11, 0.6)
c = (-10, -10, -6.5, 0.7)
x0 = (1, 0, -0.5, -1)
y0 = (0, 0.5, 1.5, 1)


def muller_brown(x, y):
    return sum(
        A[k]
        * np.exp(
            a[k] * (x - x0[k]) ** 2 + b[k] * (x - x0[k]) * (y - y0[k]) + c[k] * (y - y0[k]) ** 2
        )
        for k in range(4)
    )


MB_MINIMA = {
    "A": (-0.558, 1.442, -146.70),
    "B": (0.623, 0.028, -108.17),
    "C": (-0.050, 0.467, -80.77),
}
MB_SADDLES = {"AC": (-0.822, 0.624, -40.67), "CB": (0.212, 0.293, -72.25)}


@pytest.fixture(scope="module")
def mb_surface():
    grid = Grid(("x", "y"), (-1.5, -0.2), (1.1, 2.0), (261, 221))
    X, Y = grid.mesh()
    values = muller_brown(X, Y)
    return Surface(grid, values - values.min(), metadata={"kbt": 2.5})


def test_basins_find_the_three_minima(mb_surface):
    offset = -146.6995  # the surface is shifted to 0 at its minimum
    found = find_minima(mb_surface, None, min_depth=1.0)
    assert len(found) == 3
    for minimum, (x, y, energy) in zip(found, MB_MINIMA.values(), strict=True):
        assert minimum.point == pytest.approx((x, y), abs=0.011)
        assert minimum.value + offset == pytest.approx(energy, abs=0.3)
    # persistence: C's basin first joins the lower B's at the C-B saddle (depth ~8.5); the
    # merged basin joins A's at the A-C saddle, which ends B's basin (depth ~67.5)
    a_min, b_min, c_min = found
    assert a_min.depth == float("inf")
    assert c_min.saddle + offset == pytest.approx(MB_SADDLES["CB"][2], abs=0.3)
    assert b_min.saddle + offset == pytest.approx(MB_SADDLES["AC"][2], abs=0.3)
    assert c_min.depth == pytest.approx(MB_SADDLES["CB"][2] - MB_MINIMA["C"][2], abs=0.5)
    # a threshold above C's depth keeps only A and B
    assert len(find_minima(mb_surface, None, min_depth=20.0)) == 2


def test_muller_brown_path_and_barrier(mb_surface):
    offset = -146.6995
    path = minimum_energy_path(
        mb_surface, minima=3, min_depth_kt=0.4, images=40, dt=0.002, max_steps=20000
    )
    assert path.metadata["converged"]
    assert path.metadata["method"] == "basins"
    # A (x = -0.56) -> C -> B (x = 0.62): ends ordered along the first CV
    assert path.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert path.points[-1] == pytest.approx(MB_MINIMA["B"][:2], abs=0.011)
    # the highest point is the A-C saddle
    saddle = path.points[path.saddle_index]
    assert saddle == pytest.approx(MB_SADDLES["AC"][:2], abs=0.03)
    assert path.barrier == pytest.approx(MB_SADDLES["AC"][2] - MB_MINIMA["A"][2], abs=1.0)
    assert path.reaction_energy == pytest.approx(MB_MINIMA["B"][2] - MB_MINIMA["A"][2], abs=0.5)
    assert path.reverse_barrier == pytest.approx(MB_SADDLES["AC"][2] - MB_MINIMA["B"][2], abs=1)
    np.testing.assert_allclose(path.energies + offset, muller_brown(*path.points.T), atol=0.6)
    # the path itself follows the valley: every image is near the true MEP energy profile
    assert path.s[0] == 0 and path.s[-1] == pytest.approx(1)
    reverse = minimum_energy_path(
        mb_surface, minima=3, min_depth_kt=0.4, images=40, dt=0.002, max_steps=20000, reverse=True
    )
    np.testing.assert_allclose(reverse.points[::-1], path.points, atol=1e-6)


def test_given_ends_and_free_ends(mb_surface):
    path = minimum_energy_path(
        mb_surface,
        start=(-0.5, 1.4),
        end=(-0.1, 0.5),
        images=20,
        dt=0.002,
        max_steps=20000,
        fixed_ends=False,
    )
    assert path.metadata["method"] == "given"
    # with free ends the images slide into the minima A and C
    assert path.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.03)
    assert path.points[-1] == pytest.approx(MB_MINIMA["C"][:2], abs=0.03)


def test_one_dimensional_path():
    grid = Grid(("d0",), (1.5,), (2.5,), (201,))
    (x,) = grid.axes()
    kbt = 2.91
    f = 5 * kbt * (((x - 2.0) / 0.25) ** 2 - 1) ** 2 + 0.5 * kbt * (x - 1.5)
    surface = Surface(grid, f - f.min(), metadata={"kbt": kbt})
    path = minimum_energy_path(surface, images=101)
    assert path.points[0, 0] == pytest.approx(1.75, abs=0.006)
    assert path.points[-1, 0] == pytest.approx(2.25, abs=0.006)
    expected = (5 * kbt + 0.5 * kbt * 0.5) - (0.5 * kbt * 0.25)  # top at 2.0, start at 1.75
    assert path.barrier == pytest.approx(expected, rel=0.01)


def test_initial_string():
    s = initial_string([[0, 0], [1, 0], [1, 1]], 5)
    np.testing.assert_allclose(s, [[0, 0], [0.5, 0], [1, 0], [1, 0.5], [1, 1]])
    with pytest.raises(EstError, match="same point"):
        initial_string([[1, 1], [1, 1]], 5)


def test_errors(mb_surface):
    with pytest.raises(EstError, match="found 1 minimum"):
        minimum_energy_path(mb_surface, min_depth_kt=1000)
    with pytest.raises(EstError, match="needs 2 values"):
        minimum_energy_path(mb_surface, start=(0.0,), end=(1.0, 1.0))
    prop = Surface(mb_surface.grid, mb_surface.values, quantity="q")
    with pytest.raises(EstError, match="free-energy surface"):
        minimum_energy_path(prop)


def test_ends_from_reaction_start_and_end(mb_surface):
    options = {"min_depth_kt": 8.0, "images": 15, "max_steps": 300}
    default = minimum_energy_path(mb_surface, **options)
    assert default.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert default.metadata["ends"] == "two lowest, smaller x first"

    # start at the minimum nearest y = 0 (B), whatever x is
    oriented = minimum_energy_path(mb_surface, start_near={"y": 0.0}, **options)
    assert oriented.points[0] == pytest.approx(MB_MINIMA["B"][:2], abs=0.011)
    assert oriented.points[-1] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert oriented.metadata["ends"] == (
        "start nearest y=0 (reaction.start); end the lowest other minimum"
    )
    assert oriented.barrier == pytest.approx(default.reverse_barrier, abs=0.5)

    flipped = minimum_energy_path(mb_surface, start_near={"y": 0.0}, reverse=True, **options)
    assert flipped.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert flipped.metadata["ends"].endswith(", reversed")
    # CVs not on the surface: ignored; given ends: kept
    other = minimum_energy_path(mb_surface, start_near={"cn": 2.0}, **options)
    assert other.metadata["ends"] == "two lowest, smaller x first"
    given = minimum_energy_path(mb_surface, start=MB_MINIMA["A"][:2], end=MB_MINIMA["B"][:2],
                                start_near={"y": 0.0}, **options)  # fmt: skip
    assert given.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert given.metadata["ends"] == "given"


def test_reaction_start_chooses_among_all_minima(mb_surface):
    """The hints pick among every minimum deep enough, not only the two lowest."""
    options = {"min_depth_kt": 1.0, "images": 15, "max_steps": 300}
    c_point = {"x": MB_MINIMA["C"][0], "y": MB_MINIMA["C"][1]}
    from_c = minimum_energy_path(mb_surface, start_near=c_point, **options)
    assert from_c.points[0] == pytest.approx(MB_MINIMA["C"][:2], abs=0.011)
    assert from_c.points[-1] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)  # the lowest
    to_c = minimum_energy_path(mb_surface, start_near={"y": 0.0}, end_near=c_point, **options)
    assert to_c.points[0] == pytest.approx(MB_MINIMA["B"][:2], abs=0.011)
    assert to_c.points[-1] == pytest.approx(MB_MINIMA["C"][:2], abs=0.011)
    assert "end nearest" in to_c.metadata["ends"] and "(reaction.end)" in to_c.metadata["ends"]
    # an end hint alone: the start is the lowest other minimum
    only_end = minimum_energy_path(mb_surface, end_near={"y": 0.0}, **options)
    assert only_end.points[0] == pytest.approx(MB_MINIMA["A"][:2], abs=0.011)
    assert only_end.points[-1] == pytest.approx(MB_MINIMA["B"][:2], abs=0.011)


def test_mep_command_uses_reaction_start(mb_surface, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from estk.cli.main import app

    monkeypatch.chdir(tmp_path)
    mb_surface.save(tmp_path / "mb.dat")
    (tmp_path / "estk.yaml").write_text("reaction:\n  start: {y: 0}\n")
    runner = CliRunner()
    result = runner.invoke(app, ["mep", "mb.dat", "--min-depth", "8", "--max-steps", "300"])
    assert result.exit_code == 0, result.output
    assert "ends: start nearest y=0 (reaction.start)" in result.output
    path = MinimumEnergyPath.load(tmp_path / "mb.mep.dat")
    assert path.points[0] == pytest.approx(MB_MINIMA["B"][:2], abs=0.011)

    (tmp_path / "estk.yaml").write_text("")
    result = runner.invoke(app, ["mep", "mb.dat", "--min-depth", "8", "--max-steps", "300"])
    assert "set reaction.start in estk.yaml" in result.output


def test_path_file_roundtrip(mb_surface, tmp_path):
    other = Surface(mb_surface.grid, mb_surface.values * 1.1, metadata={"kbt": 2.5})
    path = minimum_energy_path(mb_surface, images=15, max_steps=50, profiles={"run1": other})
    np.testing.assert_allclose(path.profiles["run1"], evaluate(other, path.points))
    saved = path.save(tmp_path / "p.mep.dat")
    text = saved.read_text()
    assert text.startswith("#! FIELDS s x y fes fes.run1\n")
    assert "#! SET estk.barrier" in text
    back = MinimumEnergyPath.load(saved)
    np.testing.assert_allclose(back.points, path.points)
    np.testing.assert_allclose(back.energies, path.energies)
    np.testing.assert_allclose(back.profiles["run1"], path.profiles["run1"])
    assert back.barrier == pytest.approx(path.barrier)
    assert back.metadata["images"] == 15


# --- the original scripts' MEPs on the reference runs ------------------------------------------

OLD_MEP = DATA_DIR / "runs" / "old-mep"
EV = 1 / 0.0103636  # kJ/mol per eV, as in the original scripts


@pytest.mark.skipif(not OLD_MEP.is_dir(), reason="no old MEP output in tests/data/runs/old-mep")
@pytest.mark.parametrize(
    ("surface_file", "mep_file"),
    [
        ("d0-cn-rew-fes-AVG.dat", "d0-cn-fes-AVG-MEP.dat"),
        ("000/pot000-d0-cn-rew-fes.dat", "000-d0-cn-fes-MEP.dat"),
        ("001/pot001-d0-cn-rew-fes.dat", "001-d0-cn-fes-MEP.dat"),
        ("002/pot002-d0-cn-rew-fes.dat", "002-d0-cn-fes-MEP.dat"),
    ],
)
def test_split_mode_matches_old_mep(surface_file, mep_file):
    """Same surface, same settings (split along cn, 15 images, dt 0.01, ends fixed, free
    energies capped at 1.5 eV): the old and new paths agree."""
    surface = Surface.load(DATA_DIR / "runs" / surface_file)
    finite = np.isfinite(surface.values)
    surface.values = surface.values - surface.values[finite].min()
    old = np.loadtxt(OLD_MEP / mep_file)
    path = minimum_energy_path(
        surface, method="split", split_axis="cn", images=15, dt=0.01, max_steps=2000, cap=1.5 * EV
    )
    np.testing.assert_allclose(path.points, old[:, :2], atol=1e-4)
    # energies: old = linear interpolation on a triangulation, new = bilinear
    np.testing.assert_allclose(path.energies / EV, old[:, 2], atol=2e-3)
