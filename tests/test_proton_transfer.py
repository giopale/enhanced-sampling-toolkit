"""Proton-transfer CVs: ECToolkits' reference, hand-built geometries, estk cv proton-transfer."""

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.cli.main import app
from estk.config import load_config
from estk.cv import ProtonTransfer
from estk.errors import EstError
from estk.io.lammpstrj import Box, Frame, iter_frames
from estk.project import Project

from .conftest import DATA_DIR
from .test_combine import CONFIG, frames, write_run

REFERENCE = DATA_DIR / "proton_transfer"
TYPES = {1: "H", 2: "O"}
CASES = {
    "direct-sites": dict(kind="direct", type1=[1], type2=[41, 42, 43]),
    "indirect-sites": dict(kind="indirect", type1=[1], type2=[41, 42, 43],
                           water=list(range(2, 41))),
    "direct-liquid": dict(kind="direct", type1=[1], type2=list(range(2, 41))),
    "indirect-liquid": dict(kind="indirect", type1=[1], type2=list(range(2, 41)),
                            water=list(range(2, 41))),
}  # fmt: skip


def evaluate_all(pt):
    return np.array([pt.evaluate(f) for f in iter_frames(REFERENCE / "traj.lammpstrj.gz",
                                                         None, pt.columns)])  # fmt: skip


@pytest.mark.parametrize("case", list(CASES))
def test_matches_ectoolkits(case):
    reference = np.loadtxt(REFERENCE / f"ref-{case}.txt")
    ours = evaluate_all(ProtonTransfer(**CASES[case], type_map=TYPES, ectoolkits_compat=True))
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(reference))
    found = ~np.isnan(reference[:, 0])
    np.testing.assert_allclose(ours[found], reference[found], atol=5e-6)  # MDAnalysis: float32
    if case != "indirect-liquid":  # no bridge can be the donor or acceptor: same either way
        default = evaluate_all(ProtonTransfer(**CASES[case], type_map=TYPES))
        np.testing.assert_allclose(default, ours, equal_nan=True)


def test_bridges_are_never_the_donor_or_acceptor():
    pt = ProtonTransfer(**CASES["indirect-liquid"], type_map=TYPES)
    compat = ProtonTransfer(**CASES["indirect-liquid"], type_map=TYPES, ectoolkits_compat=True)
    degenerate = 0
    for frame in iter_frames(REFERENCE / "traj.lammpstrj.gz", None, pt.columns):
        if not np.isnan(pt.evaluate(frame)[0]):
            donor, _, bridge, _, acceptor = pt.last_chain
            assert bridge not in (donor, acceptor)
        if not np.isnan(compat.evaluate(frame)[0]):
            donor, _, bridge, _, acceptor = compat.last_chain
            degenerate += bridge in (donor, acceptor)
    assert degenerate > 0  # ECToolkits does pick such chains


# --- hand-built geometries --------------------------------------------------------------------

L = 20.0


def frame_of(atoms):
    """``atoms``: list of (element, xyz); ids 1, 2, ... in that order."""
    elements = np.array([e for e, _ in atoms])
    types = np.where(elements == "H", "1", np.where(elements == "O", "2", "3"))
    return Frame(step=0, box=Box(lo=np.zeros(3), matrix=np.eye(3) * L),
                 ids=np.arange(1, len(atoms) + 1), positions=np.array([p for _, p in atoms]),
                 columns={"type": types})  # fmt: skip


def direct(**kw):
    return ProtonTransfer(kind="direct", type1=[1], type2=[3], type_map={1: "H", 2: "O", 3: "Ti"},
                          **kw)  # fmt: skip


def test_direct_geometry_sign_and_cutoffs():
    # O1 - H2 ... O3 on a line: the proton on the type 1 side, delta = 1.0 - 1.6 < 0
    on_type1 = frame_of([("O", (5, 5, 5)), ("H", (6.0, 5, 5)), ("O", (7.6, 5, 5))])
    assert direct().evaluate(on_type1) == pytest.approx((-0.6, 2.6))
    # the H now belongs to O3 (type 2 donates): delta changes sign
    on_type2 = frame_of([("O", (5, 5, 5)), ("H", (6.7, 5, 5)), ("O", (7.6, 5, 5))])
    assert direct().evaluate(on_type2) == pytest.approx((1.7 - 0.9, 2.6))
    # across the periodic boundary
    wrapped = frame_of([("O", (0.5, 5, 5)), ("H", (19.5, 5, 5)), ("O", (17.9, 5, 5))])
    assert direct().evaluate(wrapped) == pytest.approx((-0.6, 2.6))
    # O-O beyond 3.5 Å, or an H 40° off the O-O line: no hydrogen bond
    far = frame_of([("O", (5, 5, 5)), ("H", (6.0, 5, 5)), ("O", (8.6, 5, 5))])
    assert np.isnan(direct().evaluate(far)).all()
    off = np.radians(40)
    bent = frame_of([("O", (5, 5, 5)), ("H", (5 + np.cos(off), 5 + np.sin(off), 5)),
                     ("O", (7.6, 5, 5))])  # fmt: skip
    assert np.isnan(direct().evaluate(bent)).all()
    # Ti is neither O nor H
    with_ti = frame_of([("O", (5, 5, 5)), ("H", (6.0, 5, 5)), ("O", (7.6, 5, 5)),
                        ("Ti", (6.0, 5.5, 5))])  # fmt: skip
    assert direct().evaluate(with_ti) == pytest.approx((-0.6, 2.6))


def test_indirect_geometry():
    # O1 - H2 ... O3 (bridge) - H4 ... O5 on a line
    atoms = [("O", (2, 5, 5)), ("H", (3.0, 5, 5)), ("O", (4.6, 5, 5)), ("H", (5.6, 5, 5)),
             ("O", (7.4, 5, 5))]  # fmt: skip
    pt = ProtonTransfer(kind="indirect", type1=[1], type2=[5], water=[3], type_map=TYPES)
    delta, d = pt.evaluate(frame_of(atoms))
    assert delta == pytest.approx(((1.0 - 1.6) + (1.0 - 1.8)) / 2)
    assert d == pytest.approx((2.6 + 2.8) / 2)
    assert pt.last_chain == (1, 2, 3, 4, 5)


def test_errors():
    with pytest.raises(EstError, match="direct or indirect"):
        ProtonTransfer(kind="both", type1=[1], type2=[2])
    with pytest.raises(EstError, match="bridge water"):
        ProtonTransfer(kind="indirect", type1=[1], type2=[2])
    with pytest.raises(EstError, match="donor and acceptor"):
        ProtonTransfer(kind="direct", type1=[], type2=[2])
    frame = frame_of([("O", (5, 5, 5)), ("H", (6.0, 5, 5)), ("O", (7.6, 5, 5))])
    with pytest.raises(EstError, match="type_map"):
        ProtonTransfer(kind="direct", type1=[1], type2=[3], type_map={2: "O"}).evaluate(frame)
    with pytest.raises(EstError, match="not in the dump"):
        ProtonTransfer(kind="direct", type1=[1], type2=[9], type_map=TYPES).evaluate(frame)


# --- the command ------------------------------------------------------------------------------


def test_command(tmp_path, monkeypatch):
    write_run(tmp_path / "a", *frames(0, n=300))
    # the reference frames as this run's dump, 0.05 ps apart like its COLVAR
    text = (REFERENCE / "traj.lammpstrj.gz").read_bytes()
    (tmp_path / "a" / "traj.lammpstrj.gz").write_bytes(text)
    waters = str(list(range(2, 41)))
    groups = (
        "atoms:\n  type_map: {1: H, 2: O}\n  groups:\n    ads: [1]\n    site_A: [41, 42, 43]\n"
        f"    site_liquid: {waters}\n    water_oxygens: {waters}\n"
    )
    (tmp_path / "estk.yaml").write_text(CONFIG.format(pattern="a") + groups)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [
        "cv", "proton-transfer", "--kind", "indirect", "--donor", "ads", "--acceptor", "site_A",
        "--acceptor", "site_liquid", "--water", "water_oxygens", "--dump", "traj.lammpstrj.gz",
        "--save",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    written = tmp_path / "a" / "cv_analysis"
    assert sorted(p.name for p in written.iterdir()) == ["CV-indirect-A", "CV-indirect-liquid"]
    assert (written / "CV-indirect-A").read_text().startswith(
        "#! FIELDS time indirect-delta-A indirect-d-A\n"
    )  # fmt: skip
    config = load_config(tmp_path / "estk.yaml")
    assert config.cv["indirect-delta-liquid"].source == "cv_analysis/CV-indirect-liquid"
    # read back, joined with COLVAR on time
    table = Project(config).runs[0].load_cvs(["d0", "indirect-d-A"], bias=False)
    reference = np.loadtxt(REFERENCE / "ref-indirect-sites.txt")[:, 1]
    keep = ~np.isnan(reference)
    np.testing.assert_allclose(table["indirect-d-A"], reference[keep], atol=5e-6)

    bad = CliRunner().invoke(app, ["cv", "proton-transfer", "--kind", "direct", "--donor",
                                   "nowhere", "--acceptor", "site_A"])  # fmt: skip
    assert bad.exit_code == 1 and "not a group" in bad.output
