"""CVs computed from LAMMPS dumps: the dump reader, the built-in CVs and estk cv compute."""

import gzip

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.cli.main import app
from estk.config import load_config
from estk.cv import CVProgram, compute_cvs
from estk.errors import EstError
from estk.io.lammpstrj import Box, iter_frames
from estk.project import Project

from .test_walkers import CONFIG, WALKERS, N, write_walkers

L = np.array([10.39, 11.41, 35.0])
N_ATOMS = 12
DRIVER = """UNITS LENGTH=A TIME=1
target: GROUP ATOMS=1
liquid: GROUP ATOMS=2-8
surface: GROUP ATOMS=9,10,11
dliq: DISTANCES GROUPA=target GROUPB=liquid MIN={BETA=500.}
dsurf:  DISTANCES GROUPA=target GROUPB=surface MIN={BETA=500.}
far: DISTANCES GROUPA=target GROUPB=liquid MAX={BETA=0.01}
d12: DISTANCE ATOMS=1,12
opes: OPES_METAD ARG=dliq.min PACE=100 BARRIER=50
PRINT STRIDE=1 ARG=dliq.min,dsurf.min,far.max,d12 FILE=cv/CV
"""


def positions(step, seed=0):
    rng = np.random.default_rng(seed + step)
    return rng.uniform(0, 1, (N_ATOMS, 3)) * L


def write_dump(path, steps, *, columns="id type element x y z fx fy fz", shuffle=True,
               scaled=False, cut_last=False):  # fmt: skip
    rng = np.random.default_rng(1)
    names = columns.split()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        for step in steps:
            r = positions(step)
            if scaled:
                r = r / L
            order = rng.permutation(N_ATOMS) if shuffle else np.arange(N_ATOMS)
            f.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{N_ATOMS}\n"
                    f"ITEM: BOX BOUNDS pp pp pp\n0 {L[0]}\n0 {L[1]}\n0 {L[2]}\n"
                    f"ITEM: ATOMS {columns}\n")  # fmt: skip
            for i in order:
                cells = {"id": str(i + 1), "type": "2", "element": "O", "fx": "0.1",
                         "fy": "0", "fz": "0", "vx": "0", "vy": "0", "vz": "0"}  # fmt: skip
                for k, axis in enumerate("xyz"):
                    for prefix in (axis, axis + "u", axis + "s", axis + "su"):
                        cells[prefix] = f"{r[i, k]:.8f}"
                f.write(" ".join(cells[c] for c in names) + "\n")
        if cut_last:
            f.write(
                f"ITEM: TIMESTEP\n{steps[-1] + 100}\nITEM: NUMBER OF ATOMS\n{N_ATOMS}\n"
                f"ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS {columns}\n1 2 O"
            )


def smooth_min(d, beta=500.0):
    x = beta / d
    return beta / (x.max() + np.log(np.sum(np.exp(x - x.max()))))


def brute_min_image(a, b):
    d = b - a
    return d - np.round(d / L) * L


# --- reading dumps ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "columns", "scaled"),
    [("traj.lammpstrj", "id type element x y z fx fy fz", False),
     ("traj.lammpstrj.gz", "x y z id", False),
     ("scaled.lammpstrj", "id xs ys zs", True)],
)  # fmt: skip
def test_reader(tmp_path, name, columns, scaled):
    path = tmp_path / name
    write_dump(path, [0, 100, 200], columns=columns, scaled=scaled, cut_last=True)
    frames = list(iter_frames(path))
    assert [f.step for f in frames] == [0, 100, 200]  # the cut-off last frame is dropped
    np.testing.assert_array_equal(frames[1].ids, np.arange(1, N_ATOMS + 1))
    np.testing.assert_allclose(frames[1].positions, positions(100), atol=1e-6)
    subset = next(iter(iter_frames(path, [5, 2])))
    np.testing.assert_allclose(subset.positions, positions(0)[[4, 1]], atol=1e-6)


def test_triclinic_box_and_minimum_image(tmp_path):
    path = tmp_path / "tri.lammpstrj"
    xy, xz, yz = 1.0, -0.5, 0.3
    path.write_text(
        "ITEM: TIMESTEP\n7\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS xy xz yz pp pp ff\n"
        f"{0 + min(0, xy, xz, xy + xz)} {10 + max(0, xy, xz, xy + xz)} {xy}\n"
        f"{0 + min(0, yz)} {12 + max(0, yz)} {xz}\n0 30 {yz}\nITEM: ATOMS id xs ys zs\n"
        "1 0.5 0.5 0.5\n"
    )  # fmt: skip
    frame = next(iter(iter_frames(path)))
    box = frame.box
    np.testing.assert_allclose(box.matrix, [[10, 0, 0], [xy, 12, 0], [xz, yz, 30]])
    np.testing.assert_allclose(frame.positions[0], 0.5 * box.matrix.sum(axis=0))
    assert box.periodic == (True, True, False)
    # a whole cell vector away is the same point; z is not periodic
    np.testing.assert_allclose(box.minimum_image(box.matrix[1] + [0.1, 0, 0]), [0.1, 0, 0],
                               atol=1e-12)  # fmt: skip
    np.testing.assert_allclose(box.minimum_image(np.array([0, 0, 29.0])), [0, 0, 29.0],
                               atol=1e-12)  # fmt: skip


def test_reader_errors(tmp_path):
    with pytest.raises(EstError, match="not found"):
        list(iter_frames(tmp_path / "none.lammpstrj"))
    write_dump(tmp_path / "a.lammpstrj", [0], columns="type x y z")
    with pytest.raises(EstError, match="no 'id' column"):
        list(iter_frames(tmp_path / "a.lammpstrj"))
    write_dump(tmp_path / "b.lammpstrj", [0], columns="id type vx vy vz")
    with pytest.raises(EstError, match="no coordinates"):
        list(iter_frames(tmp_path / "b.lammpstrj"))
    write_dump(tmp_path / "c.lammpstrj", [0])
    with pytest.raises(EstError, match="no atom"):
        list(iter_frames(tmp_path / "c.lammpstrj", [99]))


# --- built-in CVs -----------------------------------------------------------------------------


def test_builtin_cvs_match_brute_force(tmp_path):
    (tmp_path / "driver.plumed").write_text(DRIVER)
    program = CVProgram.from_plumed(tmp_path / "driver.plumed")
    assert list(program.values) == ["dliq.min", "dsurf.min", "far.max", "d12"]
    np.testing.assert_array_equal(program.atoms, np.arange(1, 13))
    r = positions(0)
    box = Box(lo=np.zeros(3), matrix=np.diag(L))
    values = program.evaluate(r, box)
    liquid = np.linalg.norm(brute_min_image(r[0], r[1:8]), axis=1)
    beta = 500.0
    assert values["dliq.min"] == pytest.approx(smooth_min(liquid, beta))
    assert values["dliq.min"] == pytest.approx(liquid.min(), rel=1e-2)  # smooth minimum
    assert values["far.max"] == pytest.approx(liquid.max(), rel=1e-2)
    surface = np.linalg.norm(brute_min_image(r[0], r[8:11]), axis=1)
    assert values["dsurf.min"] == pytest.approx(surface.min(), rel=1e-2)
    assert values["d12"] == pytest.approx(np.linalg.norm(brute_min_image(r[0], r[11])))

    # PLUMED's default length unit is nm; NOPBC uses the raw difference
    (tmp_path / "nm.plumed").write_text("d: DISTANCE ATOMS=1,12 NOPBC\nPRINT ARG=d FILE=x\n")
    nm = CVProgram.from_plumed(tmp_path / "nm.plumed")
    assert nm.evaluate(r[[0, 11]], box)["d"] == pytest.approx(np.linalg.norm(r[11] - r[0]) / 10)


@pytest.mark.parametrize(
    ("text", "message"),
    [("d: DISTANCES GROUPA=1 GROUPB=2,3\nPRINT ARG=d FILE=x\n", "MIN"),
     ("c: COORDINATION GROUPA=1 GROUPB=2\nPRINT ARG=c FILE=x\n", "not computed"),
     ("PRINT ARG=nothing FILE=x\n", "no action defines"),
     ("d: DISTANCE ATOMS=1,2\n", "no PRINT"),
     ("d: DISTANCE ATOMS=1,2,3\nPRINT ARG=d FILE=x\n", "two atoms"),
     ("d: DISTANCE ATOMS=1,@water\nPRINT ARG=d FILE=x\n", "not understood"),
     ("UNITS LENGTH=furlong\nd: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=x\n", "UNITS")],
)  # fmt: skip
def test_builtin_errors(tmp_path, text, message):
    (tmp_path / "p.plumed").write_text(text)
    with pytest.raises(EstError, match=message):
        CVProgram.from_plumed(tmp_path / "p.plumed")


# --- estk cv compute --------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    folder = write_walkers(tmp_path / "sim1")
    for w in range(WALKERS):
        steps = list(range(0, N * 100, 100))
        steps = steps[:500] + steps[450:]  # a restart writes 50 steps twice
        write_dump(folder / f"traj.{w}.lammpstrj", steps)
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=""))
    (tmp_path / "driver.plumed").write_text(DRIVER)
    return tmp_path


def test_compute_for_walkers(root):
    project = Project(root / "estk.yaml")
    written = compute_cvs(project.runs, root / "driver.plumed", jobs=2)
    assert [w.path.name for w in written] == ["CV.0", "CV.1", "CV.2"]
    assert written[0].frames == N + 50 and written[0].last_step == (N - 1) * 100
    text = (root / "sim1" / "cv" / "CV.1").read_text().splitlines()
    assert text[0] == "#! FIELDS time dliq.min dsurf.min far.max d12"
    # time = step x timestep (0.0005 ps), as in COLVAR
    assert float(text[2].split()[0]) == pytest.approx(0.05)

    # read back through the project: joined with COLVAR on time, the repeated steps once
    (root / "estk.yaml").write_text(
        CONFIG.format(walkers="") + "  dliq.min:\n    source: cv/CV\n"
    )  # fmt: skip
    table = Project(root / "estk.yaml").runs[0].load_cvs(["d0", "dliq.min"])
    assert len(table) == WALKERS * N
    rows = table.walker == 2
    expected = []
    for step in range(0, N * 100, 100):
        r = positions(step)
        d = np.linalg.norm(brute_min_image(r[0], r[1:8]), axis=1)
        expected.append(smooth_min(d))
    np.testing.assert_allclose(table["dliq.min"][rows], expected, rtol=1e-8)
    with pytest.raises(EstError, match="every"):
        compute_cvs(project.runs, root / "driver.plumed", every=0)


def test_command_and_along_path(root, monkeypatch):
    monkeypatch.chdir(root)
    runner = CliRunner()
    result = runner.invoke(app, ["cv", "compute", "driver.plumed", "--save", "-j", "2"])
    assert result.exit_code == 0, result.output
    flat = "".join(result.output.split())  # rich wraps at the terminal width
    assert "sim1/cv/CV.2" in flat and "dliq.min->cv/CV" in flat
    config = load_config(root / "estk.yaml")
    assert config.cv["dliq.min"].source == "cv/CV" and config.cv["d0"].bandwidth == 0.03

    assert runner.invoke(app, ["reweight", "d0", "cn", "--points", "31"]).exit_code == 0
    assert runner.invoke(app, ["mep", "reweight-d0-cn/combined.dat"]).exit_code == 0
    result = runner.invoke(app, ["along-path", "reweight-d0-cn/combined.mep.dat", "dliq.min"])
    assert result.exit_code == 0, result.output
    assert (root / "reweight-d0-cn" / "combined.along.dat").is_file()
