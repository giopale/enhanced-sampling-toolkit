import math

import numpy as np
import pytest

from estk import units
from estk.errors import EstError, MissingFileError
from estk.io.lammps import read_log_metadata
from estk.io.plumed import parse_actions, read_input, read_state_bias_factor

from .conftest import LAMMPS_LOG, PLUMED_INPUT

# --- LAMMPS -----------------------------------------------------------------------------------


def write(path, text):
    path.write_text(text)
    return path


def test_lammps_log(tmp_path):
    meta = read_log_metadata(write(tmp_path / "log.lammps", LAMMPS_LOG))
    assert meta.units == "metal"
    assert meta.timestep == 0.0005
    assert meta.temperature == 350.0
    assert meta.warnings == []


@pytest.mark.parametrize(
    ("fix", "expected"),
    [
        ("fix nvt all nvt temp 300 300 0.1", 300.0),
        ("fix 2 mobile npt temp 280.5 280.5 0.1 iso 1 1 1", 280.5),
        ("fix 3 all langevin 310 310 0.1 48279", 310.0),
        ("fix 4 all temp/csvr 290 290 0.1 1234", 290.0),
    ],
)
def test_lammps_thermostat_styles(tmp_path, fix, expected):
    meta = read_log_metadata(write(tmp_path / "log", f"units real\ntimestep 0.5\n{fix}\nrun 10\n"))
    assert meta.temperature == expected


def test_lammps_last_run_wins(tmp_path):
    text = (
        "units metal\ntimestep 0.001\nfix eq all nvt temp 100 300 0.1\nrun 100\n"
        "unfix eq\ntimestep 0.0005\nfix prod all nvt temp 350 350 0.1\nrun 1000\n"
        "fix later all nvt temp 999 999 0.1\n"  # defined after the last run: ignored
    )
    meta = read_log_metadata(write(tmp_path / "log", text))
    assert meta.timestep == 0.0005
    assert meta.temperature == 350.0


def test_lammps_ramp_and_several_thermostats_warn(tmp_path):
    text = "fix a all nvt temp 300 400 0.1\nfix b all langevin 300 300 0.1 1\nrun 1\n"
    meta = read_log_metadata(write(tmp_path / "log", text))
    assert meta.temperature == 300.0
    assert len(meta.warnings) == 2


def test_lammps_unresolved_variable(tmp_path):
    meta = read_log_metadata(write(tmp_path / "log", "fix 1 all nvt temp ${T} ${T} 0.1\nrun 1\n"))
    assert meta.temperature is None


def test_lammps_missing(tmp_path):
    with pytest.raises(MissingFileError):
        read_log_metadata(tmp_path / "nope")


# --- PLUMED -----------------------------------------------------------------------------------


def test_plumed_input(tmp_path):
    p = read_input(write(tmp_path / "plumed.plumed", PLUMED_INPUT))
    assert p.time_units == "ps"
    assert p.energy_units == "kj/mol"
    assert p.bias is not None
    assert p.bias.label == "opes"
    assert p.bias.bias_column == "opes.bias"
    assert p.bias.args == ["d0", "cn"]
    assert p.bias.bias_factor == 10.0
    assert p.bias.state_file == "STATE"
    assert p.walls == {"d0": (1.5, 2.5), "cn": (None, 3.0)}
    assert p.wall_bias_columns == ["uwall.bias", "lwall.bias"]


def test_plumed_several_walls_on_one_side(tmp_path):
    # as in the real input: an inner and an outer upper wall on d0
    text = (
        "UPPER_WALLS ARG=d0 AT=2.2 KAPPA=300 EXP=2\n"
        "LOWER_WALLS ARG=d0 AT=1.5 KAPPA=300 EXP=2\n"
        "UPPER_WALLS ARG=d0 AT=3.0 KAPPA=300 EXP=2\n"
        "LOWER_WALLS ARG=d0 AT=1.2 KAPPA=300\n"
    )
    p = read_input(write(tmp_path / "plumed.dat", text))
    assert p.walls == {"d0": (1.5, 2.2)}  # innermost on each side
    assert [(w.side, w.at) for w in p.wall_terms] == [
        ("upper", 2.2),
        ("lower", 1.5),
        ("upper", 3.0),
        ("lower", 1.2),
    ]
    wall = p.wall_terms[0]
    assert wall.energy(np.array([2.0, 2.3]))[1] == pytest.approx(300 * 0.1**2)


def test_plumed_include_and_braces(tmp_path):
    write(tmp_path / "walls.dat", "UPPER_WALLS ARG=d0 AT=2.0 KAPPA=1 LABEL=w\n")
    write(
        tmp_path / "plumed.dat",
        "UNITS ENERGY=eV TIME=fs\n"
        "INCLUDE FILE=walls.dat\n"
        "m: METAD ARG=d0 SIGMA=0.1 HEIGHT=1 PACE=10 BIASFACTOR=inf "
        "TEMP=300 FMT={%f  %g}\n"
        "ENDPLUMED\n"
        "LOWER_WALLS ARG=d0 AT=0 KAPPA=1\n",
    )
    p = read_input(tmp_path / "plumed.dat")
    assert p.energy_units == "eV"
    assert p.time_units == "fs"
    assert p.walls == {"d0": (None, 2.0)}  # LOWER_WALLS is after ENDPLUMED
    assert p.bias.bias_factor == math.inf
    assert p.bias.temperature == 300.0
    actions = parse_actions(tmp_path / "plumed.dat")
    assert actions[-1].get("FMT") == "%f  %g"


def test_plumed_bias_factor_from_state_file(tmp_path):
    write(tmp_path / "STATE", "#! FIELDS time sigma0_d0\n#! SET biasfactor 12.5\n0 0.1\n")
    write(tmp_path / "plumed.dat", "opes: OPES_METAD ARG=d0 PACE=10 STATE_WFILE=STATE\n")
    assert read_input(tmp_path / "plumed.dat").bias.bias_factor == 12.5
    assert read_state_bias_factor(tmp_path / "missing") is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("w: UPPER_WALLS ARG=a,b AT=1\n", "2 ARG but 1 AT"),
        ("o: OPES_METAD ...\n ARG=a\n", "unterminated"),
        ("UNITS NATURAL\n", "NATURAL"),
        ("INCLUDE FILE=plumed.dat\n", "INCLUDE loop"),
        ("o: OPES_METAD ARG=a BIASFACTOR=ten\n", "BIASFACTOR"),
    ],
)
def test_plumed_errors(tmp_path, text, message):
    with pytest.raises(EstError, match=message):
        read_input(write(tmp_path / "plumed.dat", text))


# --- units ------------------------------------------------------------------------------------


def test_units():
    assert units.kbt(300) == pytest.approx(2.494, abs=1e-3)  # kJ/mol
    assert units.kbt(300, "eV") == pytest.approx(0.02585, abs=1e-5)
    assert units.time_factor("ns") == 1000.0
    assert units.time_factor("0.5") == 0.5
    assert units.lammps_time_factor("real") == 1e-3
    with pytest.raises(EstError, match="unknown energy unit"):
        units.energy_factor("hartree/particle")
    with pytest.raises(EstError, match="simulation.timestep"):
        units.lammps_time_factor("lj")
