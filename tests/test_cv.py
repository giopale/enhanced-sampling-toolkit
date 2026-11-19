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


def plumed_switch(r, r0, nn=6, mm=0, d0=0.0, dmax=None, stretch=True):
    """PLUMED's rational switching function, written out one distance at a time."""
    mm = mm or 2 * nn
    if dmax is None:
        dmax = d0 + r0 * 0.00001 ** (1.0 / (nn - mm))

    def raw(d):
        x = max(d - d0, 0.0) / r0
        if abs(x - 1) < 1e-9:
            return nn / mm
        return (1 - x**nn) / (1 - x**mm)

    if r > dmax:
        return 0.0
    if not stretch or dmax == np.inf:
        return raw(r)
    return (raw(r) - raw(dmax)) / (raw(0.0) - raw(dmax))


def brute_coordination(r, a, b, box_lengths, **switch):
    """a, b: 0-based rows of r; b None: the pairs within a."""
    pairs = [(i, j) for k, i in enumerate(a) for j in a[k + 1:]] if b is None else \
        [(i, j) for i in a for j in b if i != j]  # fmt: skip
    total = 0.0
    for i, j in pairs:
        d = np.linalg.norm(brute_min_image(r[i], r[j][None, :])[0])
        total += plumed_switch(d, **switch)
    return total


COORD = """UNITS LENGTH=A
ox: GROUP ATOMS=1-4
hy: GROUP ATOMS=5-12
cn: COORDINATION GROUPA=ox GROUPB=hy R_0=3.0 NN=24 MM=0
self: COORDINATION GROUPA=ox R_0=4.0
general: COORDINATION GROUPA=ox GROUPB=hy SWITCH={RATIONAL R_0=2.5 NN=6 MM=10 D_0=0.5 D_MAX=6}
plain: COORDINATION GROUPA=ox GROUPB=hy SWITCH={RATIONAL R_0=2.5 NN=8 NOSTRETCH D_MAX=5}
paired: COORDINATION GROUPA=1,2 GROUPB=5,6 PAIR R_0=3.5
overlap: COORDINATION GROUPA=1-6 GROUPB=5-8 R_0=3.0
PRINT ARG=cn,self,general,plain,paired,overlap FILE=x
"""


def test_coordination_matches_plumed_formula(tmp_path):
    (tmp_path / "c.plumed").write_text(COORD)
    program = CVProgram.from_plumed(tmp_path / "c.plumed")
    box = Box(lo=np.zeros(3), matrix=np.diag(L))
    ox, hy = list(range(4)), list(range(4, 12))
    for step in range(5):
        r = positions(step)
        values = program.evaluate(r, box)
        expected = {
            "cn": brute_coordination(r, ox, hy, L, r0=3.0, nn=24),
            "self": brute_coordination(r, ox, None, L, r0=4.0),
            "general": brute_coordination(r, ox, hy, L, r0=2.5, nn=6, mm=10, d0=0.5, dmax=6.0),
            "plain": brute_coordination(r, ox, hy, L, r0=2.5, nn=8, dmax=5.0, stretch=False),
            "overlap": brute_coordination(r, list(range(6)), [4, 5, 6, 7], L, r0=3.0),
        }
        pair_d = [
            np.linalg.norm(brute_min_image(r[i], r[j][None, :])[0]) for i, j in ((0, 4), (1, 5))
        ]
        expected["paired"] = sum(plumed_switch(d, r0=3.5) for d in pair_d)
        for name, value in expected.items():
            assert values[name] == pytest.approx(value, rel=1e-10, abs=1e-12), name
        assert values["cn"] > 0.01  # the test has pairs within reach


def test_coordination_by_hand(tmp_path):
    """One O-H pair at r: 1/(1+(r/r0)^24) stretched to 0 at d_max = r0 * 1e5^(1/24)."""
    (tmp_path / "c.plumed").write_text(
        "c: COORDINATION GROUPA=1 GROUPB=2 R_0=1.2 NN=24 MM=0\nUNITS LENGTH=A\nPRINT ARG=c FILE=x\n"
    )
    program = CVProgram.from_plumed(tmp_path / "c.plumed")
    box = Box(lo=np.zeros(3), matrix=np.diag(L))
    dmax = 1.2 * 1e5 ** (1 / 24)
    sd = 1 / (1 + (dmax / 1.2) ** 24)
    for r, raw in [(0.97, 1 / (1 + (0.97 / 1.2) ** 24)), (1.2, 0.5), (1.2 + 1e-7, None),
                   (dmax + 0.01, None)]:  # fmt: skip
        value = program.evaluate(np.array([[1.0, 1.0, 1.0], [1.0 + r, 1.0, 1.0]]), box)["c"]
        if raw is not None:
            assert value == pytest.approx((raw - sd) / (1 - sd), rel=1e-12)
        elif r > dmax:
            assert value == 0.0
        else:  # continuous through x = 1
            assert value == pytest.approx((0.5 - sd) / (1 - sd), abs=1e-5)


FUNCTIONS = """UNITS LENGTH=A
hydrogens: GROUP ATOMS=5-12
O2cup: GROUP ATOMS=1,2
O2cdw: GROUP ATOMS=3,4
nHup: COORDINATION GROUPA=O2cup GROUPB=hydrogens R_0=3.0 NN=24 MM=0
nHdw: COORDINATION GROUPA=O2cdw GROUPB=hydrogens R_0=3.0 NN=24 MM=0
nHads: COMBINE ARG=nHup,nHdw PERIODIC=NO
covO2c: CUSTOM ARG=nHads FUNC=x/4 PERIODIC=NO
d: DISTANCE ATOMS=1,5
mix: COMBINE ARG=nHup,d COEFFICIENTS=2,3 PARAMETERS=0.5,1 POWERS=2,1 NORMALIZE PERIODIC=NO
f: CUSTOM ARG=covO2c,d VAR=c,r FUNC=c^2+exp(-r)*max(r,2)-step(r-3)+2*pi PERIODIC=NO
PRINT ARG=covO2c,mix,f FILE=x
"""


def test_combine_and_custom(tmp_path):
    (tmp_path / "f.plumed").write_text(FUNCTIONS)
    program = CVProgram.from_plumed(tmp_path / "f.plumed")
    names = list(program.values)  # each function after its arguments
    for later, earlier in [("nHads", "nHup"), ("covO2c", "nHads"), ("mix", "d"), ("f", "covO2c")]:
        assert names.index(earlier) < names.index(later)
    np.testing.assert_array_equal(program.atoms, np.arange(1, 13))
    box = Box(lo=np.zeros(3), matrix=np.diag(L))
    for step in range(3):
        r = positions(step)
        v = program.evaluate(r, box)
        up = brute_coordination(r, [0, 1], list(range(4, 12)), L, r0=3.0, nn=24)
        down = brute_coordination(r, [2, 3], list(range(4, 12)), L, r0=3.0, nn=24)
        assert v["covO2c"] == pytest.approx((up + down) / 4, rel=1e-12)
        d = np.linalg.norm(brute_min_image(r[0], r[4][None, :])[0])
        assert v["mix"] == pytest.approx(0.4 * (up - 0.5) ** 2 + 0.6 * (d - 1), rel=1e-12)
        c = v["covO2c"]
        expected = c**2 + np.exp(-d) * max(d, 2) - (1.0 if d >= 3 else 0.0) + 2 * np.pi
        assert v["f"] == pytest.approx(expected, rel=1e-12)

    # a function of one geometric value needs only its atoms
    (tmp_path / "g.plumed").write_text(
        "UNITS LENGTH=A\nd: DISTANCE ATOMS=3,7\nhalf: CUSTOM ARG=d FUNC=x/2\n"
        "bad: CUSTOM ARG=d FUNC=log(x-100)\nPRINT ARG=half,bad FILE=x\n"
    )
    small = CVProgram.from_plumed(tmp_path / "g.plumed")
    np.testing.assert_array_equal(small.atoms, [3, 7])
    v = small.evaluate(positions(0)[[2, 6]], box)
    assert v["half"] == pytest.approx(v["d"] / 2)
    assert np.isnan(v["bad"])  # log of a negative number: NaN, not a crash


@pytest.mark.parametrize(
    ("text", "message"),
    [("d: DISTANCES GROUPA=1 GROUPB=2,3\nPRINT ARG=d FILE=x\n", "MIN"),
     ("t: TORSION ATOMS=1,2,3,4\nPRINT ARG=t FILE=x\n", "not computed"),
     ("c: COORDINATION GROUPA=1 GROUPB=2\nPRINT ARG=c FILE=x\n", "R_0"),
     ("c: COORDINATION GROUPA=1 GROUPB=2 SWITCH={EXP R_0=1}\nPRINT ARG=c FILE=x\n", "RATIONAL"),
     ("c: COORDINATION GROUPA=1,2 GROUPB=3 PAIR R_0=1\nPRINT ARG=c FILE=x\n", "same length"),
     ("c: COORDINATION GROUPA=1 GROUPB=2 R_0=1 NN=6 MM=6\nPRINT ARG=c FILE=x\n", "NN != MM"),
     ("PRINT ARG=nothing FILE=x\n", "no action defines"),
     ("d: DISTANCE ATOMS=1,2\nf: CUSTOM ARG=d FUNC=__import__('os')\nPRINT ARG=f FILE=x\n",
      "only numbers"),
     ("d: DISTANCE ATOMS=1,2\nf: CUSTOM ARG=d FUNC=x*w\nPRINT ARG=f FILE=x\n", "unknown name 'w'"),
     ("d: DISTANCE ATOMS=1,2\nf: CUSTOM ARG=d FUNC=x.real\nPRINT ARG=f FILE=x\n", "only numbers"),
     ("d: DISTANCE ATOMS=1,2\nf: CUSTOM ARG=d FUNC=(x\nPRINT ARG=f FILE=x\n", "parse"),
     ("d: DISTANCE ATOMS=1,2\nf: CUSTOM ARG=d VAR=a,b FUNC=a\nPRINT ARG=f FILE=x\n", "VAR"),
     ("d: DISTANCE ATOMS=1,2\nf: COMBINE ARG=d,d POWERS=2\nPRINT ARG=f FILE=x\n", "POWERS"),
     ("a: COMBINE ARG=b\nb: COMBINE ARG=a\nPRINT ARG=a FILE=x\n", "depends on itself"),
     ("d: DISTANCE ATOMS=1,2\n", "no PRINT"),
     ("d: DISTANCE ATOMS=1,2,3\nPRINT ARG=d FILE=x\n", "two atoms"),
     ("d: DISTANCE ATOMS=1,@water\nPRINT ARG=d FILE=x\n", "not understood"),
     ("UNITS LENGTH=furlong\nd: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=x\n", "UNITS"),
     ("d: DISTANCE ATOMS=1,2\nPRINT ARG=d FILE=cv/CV.$w\n", "shell variable")],
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
