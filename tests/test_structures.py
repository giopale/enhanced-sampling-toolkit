"""Structures sampled from the basins (estk structures basins) and written as extxyz."""

import shlex

import numpy as np
import pytest
from typer.testing import CliRunner

from estk.analysis.structures import key_name, sample_basins
from estk.cli.main import app
from estk.errors import EstError
from estk.io.lammpstrj import iter_frames, scan_steps
from estk.project import Project

from .test_basins import OPTIONS, write_tilted
from .test_combine import CONFIG

LO = np.array([-1.0, 2.0, 0.5])
L = np.array([10.0, 11.0, 30.0])
N = 3000


def write_dump(path, steps, *, element=True, lammps_default=False):
    """3 atoms, written in reverse id order; atom 1 sits at x = step / 1e5 (from the cell
    origin) so that a structure can be traced back to its step. ``lammps_default``: the
    element column says C for every atom, as LAMMPS writes it without dump_modify element."""
    lines = []
    for step in steps:
        lines += ["ITEM: TIMESTEP", str(step), "ITEM: NUMBER OF ATOMS", "3",
                  "ITEM: BOX BOUNDS pp pp pp"]  # fmt: skip
        lines += [f"{lo} {lo + length}" for lo, length in zip(LO, L, strict=True)]
        lines.append("ITEM: ATOMS id type" + (" element" if element else "") + " x y z")
        atoms = [(1, 2, "O", step / 1e5), (2, 1, "H", 1.0), (3, 1, "H", 2.0)]
        for atom_id, kind, symbol, x in reversed(atoms):
            name = f" {'C' if lammps_default else symbol}" if element else ""
            lines.append(f"{atom_id} {kind}{name} {LO[0] + x:.8f} {LO[1] + 1:.8f} {LO[2] + 1:.8f}")
    path.write_text("\n".join(lines) + "\n")


def read_extxyz(path):
    """[(header dict, species, positions)] of every structure."""
    text = path.read_text().splitlines()
    out, i = [], 0
    while i < len(text):
        n = int(text[i])
        header = dict(item.split("=", 1) for item in shlex.split(text[i + 1]))
        atoms = [line.split() for line in text[i + 2 : i + 2 + n]]
        out.append((header, [a[0] for a in atoms], np.array([a[1:] for a in atoms], float)))
        i += 2 + n
    return out


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("structures")
    for k in range(2):
        folder = root / f"r{k}"
        write_tilted(folder, k, offset=-7.0 * k, n=N)
        steps = np.arange(0, N, 2) * 100  # every second COLVAR frame (0.05 ps, dt 0.0005 ps)
        write_dump(folder / "traj.lammpstrj", steps, element=(k == 0))
    (root / "estk.yaml").write_text(
        CONFIG.format(pattern="r*") + "atoms:\n  type_map: {1: H, 2: O}\n"
    )
    return Project(root / "estk.yaml")


@pytest.fixture(scope="module")
def surface(project):
    return project.reweight(["d0", "cn"], **OPTIONS).combined


def test_dump_scan_and_step_filter(project):
    path = project.runs[0].path / "traj.lammpstrj"
    steps = scan_steps(path)
    assert steps[:3].tolist() == [0, 200, 400] and len(steps) == N // 2
    frames = list(iter_frames(path, steps={400, 1000}))
    assert [f.step for f in frames] == [400, 1000]
    assert frames[0].ids.tolist() == [1, 2, 3]

    cut = path.with_name("cut.lammpstrj")  # a job stopped while writing the last frame
    cut.write_text("\n".join(path.read_text().splitlines()[:-2]) + "\n")
    assert len(scan_steps(cut)) == N // 2 - 1
    assert list(iter_frames(cut, steps={int(steps[-1])})) == []


def test_sample_follows_the_basins_and_the_dump(project, surface, tmp_path):
    sample = sample_basins(project, ["d0", "cn"], surface, n=25, quantities=["q"])
    assert {f.basin for f in sample.frames} == {"1", "2"}
    assert sum(f.basin == "1" for f in sample.frames) == 25
    for f in sample.frames:
        assert f.step % 200 == 0  # only frames in the dump
        assert (f.values["d0"] < 2.0) == (f.basin == "1")  # the left well is basin 1
        assert f.values["q"] == pytest.approx(0.6 if f.values["d0"] < 2.0 else 0.1)
        assert np.isfinite(f.free_energy)
    assert len({(f.run, f.step) for f in sample.frames}) == len(sample.frames)  # no repeats
    again = sample_basins(project, ["d0", "cn"], surface, n=25, quantities=["q"])
    assert [(f.run, f.step) for f in again.frames] == [(f.run, f.step) for f in sample.frames]

    path = sample.write(project, tmp_path / "out.extxyz")
    structures = read_extxyz(path)
    assert len(structures) == len(sample.frames)
    for (header, species, positions), f in zip(structures, sample.frames, strict=True):
        assert species == ["O", "H", "H"]  # id order; from 'element' or atoms.type_map
        assert positions[0, 0] == pytest.approx(f.step / 1e5)  # the right frame
        assert positions[1].tolist() == pytest.approx([1.0, 1.0, 1.0])  # shifted to the cell
        assert float(header["d0"]) == pytest.approx(f.values["d0"])
        assert header["basin"] == f.basin and header["run"] == f.run
        assert int(header["step"]) == f.step and header["sampling"] == "boltzmann"
        assert header["Lattice"] == "10 0 0 0 11 0 0 0 30" and header["pbc"] == "T T T"
        assert "q" in header and float(header["fes"]) == pytest.approx(f.free_energy)
        assert header["fes_units"] == "kJ/mol"


def test_boltzmann_prefers_low_free_energy(project, surface):
    kw = {"n": 150, "basins": ["2"]}
    boltzmann = sample_basins(project, ["d0", "cn"], surface, seed=1, **kw)
    uniform = sample_basins(project, ["d0", "cn"], surface, mode="uniform", seed=1, **kw)
    mean = [np.mean([f.free_energy for f in s.frames]) for s in (boltzmann, uniform)]
    assert mean[0] < mean[1] - 0.5 * surface.metadata["kbt"]
    assert boltzmann.effective["2"] > 1 and boltzmann.candidates["2"] > 150
    weights = [f.log_weight for f in boltzmann.frames]
    assert weights == sorted(weights, reverse=True)  # the most probable first


def test_errors(project, surface):
    with pytest.raises(EstError, match="no basin"):
        sample_basins(project, ["d0", "cn"], surface, basins=["9"])
    with pytest.raises(EstError, match="mode"):
        sample_basins(project, ["d0", "cn"], surface, mode="flat")
    with pytest.raises(EstError, match="dump file not found"):
        sample_basins(project, ["d0", "cn"], surface, dump="nope.lammpstrj")
    assert key_name("dOOliq.min") == "dOOliq_min" and key_name("2x") == "_2x"


def test_command(project, surface, monkeypatch):
    root = project.config.path.parent
    monkeypatch.chdir(root)
    surface.save(root / "fes.dat")
    runner = CliRunner()
    out = runner.invoke(app, ["structures", "basins", "d0", "cn", "--surface", "fes.dat",
                              "-n", "5", "--basin", "1", "--quantity", "q"])  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "wrote 5 structures" in out.output
    structures = read_extxyz(root / "structures-d0-cn.extxyz")
    assert len(structures) == 5 and all(h["basin"] == "1" for h, _, _ in structures)


def test_species(project, surface, tmp_path, monkeypatch):
    """atoms.type_map wins; the element column is used without it, unless it is LAMMPS's
    default (C for every type); --type-map gives the map on the command line."""
    root = tmp_path / "proj"
    write_tilted(root / "r0", 0, n=600)
    write_dump(root / "r0" / "traj.lammpstrj", np.arange(0, 600, 2) * 100, lammps_default=True)
    (root / "estk.yaml").write_text(CONFIG.format(pattern="r*"))
    plain = Project(root / "estk.yaml")
    sample = sample_basins(plain, ["d0", "cn"], surface, n=3)
    with pytest.raises(EstError, match="dump_modify element"):
        sample.write(plain, tmp_path / "x.extxyz")
    path = sample.write(plain, tmp_path / "x.extxyz", type_map={1: "H", 2: "O"})
    assert all(s == ["O", "H", "H"] for _, s, _ in read_extxyz(path))
    with pytest.raises(EstError, match="no element for type"):
        sample.write(plain, tmp_path / "x.extxyz", type_map={1: "H"})

    write_dump(root / "r0" / "traj.lammpstrj", np.arange(0, 600, 2) * 100)  # real elements
    path = sample.write(plain, tmp_path / "y.extxyz")
    assert all(s == ["O", "H", "H"] for _, s, _ in read_extxyz(path))

    write_dump(root / "r0" / "traj.lammpstrj", np.arange(0, 600, 2) * 100, lammps_default=True)
    monkeypatch.chdir(root)
    surface.save(root / "fes.dat")
    runner = CliRunner()
    args = ["structures", "basins", "d0", "cn", "--surface", "fes.dat", "-n", "2"]
    out = runner.invoke(app, args)
    assert out.exit_code == 1 and "dump_modify element" in out.output
    out = runner.invoke(app, [*args, "--type-map", "1=H,2=O"])
    assert out.exit_code == 0, out.output
    assert all(s == ["O", "H", "H"] for _, s, _ in read_extxyz(root / "structures-d0-cn.extxyz"))
    out = runner.invoke(app, [*args, "--type-map", "H,O"])
    assert out.exit_code == 1 and "--type-map" in out.output
