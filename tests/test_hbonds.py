"""Hydrogen bonds of one oxygen with partner oxygens (estk cv hbonds)."""

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.cli.main import app
from estk.cv import HBonds, compute_cvs
from estk.cv.hbonds import ids_of, parse_axis
from estk.errors import EstError
from estk.io.lammpstrj import Box, Frame
from estk.project import Project

from .test_walkers import CONFIG, WALKERS, write_walkers

L = 10.0
# id: (element, position). Centre O 10 at (5, 5, 5).
ATOMS = {
    10: ("O", (5.0, 5.0, 5.0)),  # the centre
    1: ("H", (5.0, 5.97, 5.0)),  # its H, pointing at 20 along +y: the centre donates
    3: ("H", (5.0, 5.0, 4.03)),  # its other H, pointing at nothing
    20: ("O", (5.0, 7.8, 5.0)),  # partner along +y, 2.8 Å: accepts from the centre
    21: ("O", (7.6, 5.0, 5.0)),  # partner along +x, 2.6 Å: donates to the centre
    2: ("H", (6.62, 5.0, 5.0)),  # 21's H, pointing at the centre
    22: ("O", (5.0, 5.0, 8.2)),  # partner 3.2 Å along +z, no H in between: no bond
    23: ("O", (5.0, 2.0, 1.0)),  # partner too far
    30: ("O", (1.0, 1.0, 1.0)),  # another oxygen, not a partner
}


def frame(atoms=ATOMS, step=0):
    ids = np.array(sorted(atoms))
    positions = np.array([atoms[i][1] for i in ids])
    elements = np.array([atoms[i][0] for i in ids])
    box = Box(lo=np.zeros(3), matrix=np.diag([L, L, L]))
    return Frame(step=step, box=box, ids=ids, positions=positions, columns={"element": elements})


def cv(**kw):
    return HBonds(center=[10], partners=[20, 21, 22, 23], axis=parse_axis(kw.pop("axis", "y")),
                  **kw)  # fmt: skip


def test_bonds_by_hand():
    program = cv()
    n, d_oo, angle, role = program.evaluate(frame())
    assert n == 2 and d_oo == pytest.approx(2.6)  # the shortest: 21, along x
    assert angle == pytest.approx(90.0) and role == -1  # 21 donates to the centre
    assert sorted(program.last_bonds) == [(10, 1, 20), (21, 2, 10)]

    n, d_oo, angle, role = cv(direction="donor").evaluate(frame())
    assert (n, role) == (1, 1) and d_oo == pytest.approx(2.8) and angle == pytest.approx(0.0)
    n, d_oo, _, role = cv(direction="acceptor").evaluate(frame())
    assert (n, role) == (1, -1) and d_oo == pytest.approx(2.6)
    assert cv(axis="x").evaluate(frame())[2] == pytest.approx(0.0)  # centre -> 21 is +x
    assert cv(axis="-1,0,0").evaluate(frame())[2] == pytest.approx(180.0)
    assert cv(distance=2.7).evaluate(frame())[:2] == pytest.approx((1.0, 2.6))

    # the H of 21 turned 40° away from the O-O axis: no longer a bond with a 30° limit
    turned = dict(ATOMS)
    turned[2] = ("H", (7.6 - 0.98 * np.cos(np.radians(40)), 5.0 + 0.98 * np.sin(np.radians(40)),
                       5.0))  # fmt: skip
    assert cv().evaluate(frame(turned))[:2] == pytest.approx((1.0, 2.8))
    assert cv(angle=45.0).evaluate(frame(turned))[0] == 2

    lonely = {k: v for k, v in ATOMS.items() if k not in (1, 2)}
    n, d_oo, angle, role = cv().evaluate(frame(lonely))
    assert n == 0 and np.isnan(d_oo) and np.isnan(angle) and np.isnan(role)


def test_minimum_image():
    shifted = {k: (e, ((x - 4.0) % L, y, z)) for k, (e, (x, y, z)) in ATOMS.items()}
    # 21 and its H now sit across the box edge from the centre
    n, d_oo, angle, role = cv(axis="x").evaluate(frame(shifted))
    assert n == 2 and d_oo == pytest.approx(2.6) and angle == pytest.approx(0.0)


def test_inputs():
    assert ids_of("262", {}) == [262] and ids_of("1-3,7", {}) == [1, 2, 3, 7]
    assert ids_of("Oliq", {"Oliq": [5, 6]}) == [5, 6]
    with pytest.raises(EstError, match="not a group"):
        ids_of("Oliq", {})
    assert parse_axis("Y").tolist() == [0, 1, 0]
    assert parse_axis("0,0,2").tolist() == [0, 0, 1]
    for bad in ("w", "1,2", "0,0,0"):
        with pytest.raises(EstError, match="axis"):
            parse_axis(bad)
    with pytest.raises(EstError, match="direction"):
        cv(direction="sideways")
    with pytest.raises(EstError, match="not in the dump"):
        HBonds(center=[99], partners=[20]).evaluate(frame())


def write_dump(path, steps):
    lines = []
    for step in steps:
        lines += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", str(len(ATOMS))]
        lines += ["ITEM: BOX BOUNDS pp pp pp", *[f"0 {L}"] * 3, "ITEM: ATOMS id element x y z"]
        lines += [f"{i} {e} {x} {y} {z}" for i, (e, (x, y, z)) in ATOMS.items()]
    path.write_text("\n".join(lines) + "\n")


def test_compute_and_command(tmp_path, monkeypatch):
    folder = write_walkers(tmp_path / "sim1")
    for w in range(WALKERS):
        write_dump(folder / f"traj.{w}.lammpstrj", [0, 100, 200])
    (tmp_path / "estk.yaml").write_text(CONFIG.format(walkers=""))
    project = Project(tmp_path / "estk.yaml")
    written = compute_cvs(project.runs, cv(name="liq", file="cv/CV-hb-liq"))
    assert len(written) == WALKERS and written[0].frames == 3
    data = np.loadtxt(written[0].path, comments="#")
    assert data[0, 1:].tolist() == pytest.approx([2, 2.6, 90.0, -1])

    monkeypatch.chdir(tmp_path)
    out = CliRunner().invoke(app, ["cv", "hbonds", "--center", "10", "--partners", "20-23",
                                   "--axis", "y", "--name", "liq", "--save"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "hb-angle-liq" in (tmp_path / "estk.yaml").read_text()
    assert "FIELDS time hb-n-liq hb-doo-liq hb-angle-liq hb-role-liq" in (
        folder / "cv_analysis" / "CV-hb-liq.0").read_text()  # fmt: skip
