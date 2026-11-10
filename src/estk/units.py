"""Physical constants and unit conversions.

Internally estk uses ps for time and K for temperature. Energies keep the units of the file they
come from (PLUMED's energy units, kJ/mol by default) and are converted with ``energy_factor``.
"""

from __future__ import annotations

from estk.errors import EstError

# Molar gas constant in kJ/(mol K) (CODATA 2018).
R_KJ_MOL_K = 8.314462618e-3

# Energy units understood by PLUMED's UNITS ENERGY=..., as kJ/mol per unit.
_ENERGY_TO_KJ_MOL = {
    "kj/mol": 1.0,
    "j/mol": 1e-3,
    "kcal/mol": 4.184,
    "ev": 96.48533212,
}

# Time units understood by PLUMED's UNITS TIME=..., as ps per unit.
_TIME_TO_PS = {"ps": 1.0, "fs": 1e-3, "ns": 1e3, "atomic": 2.418884326509e-5}

# Time unit of the LAMMPS `timestep` command for each `units` style, as ps per unit.
_LAMMPS_TIME_TO_PS = {
    "metal": 1.0,
    "real": 1e-3,
    "electron": 1e-3,
    "si": 1e12,
    "cgs": 1e12,
    "micro": 1e6,
    "nano": 1e3,
}


def energy_factor(units: str) -> float:
    """kJ/mol per unit of ``units`` (a PLUMED energy unit name or a number in kJ/mol)."""
    return _factor(units, _ENERGY_TO_KJ_MOL, "energy")


def time_factor(units: str) -> float:
    """ps per unit of ``units`` (a PLUMED time unit name or a number in ps)."""
    return _factor(units, _TIME_TO_PS, "time")


def lammps_time_factor(units: str) -> float:
    """ps per unit of the LAMMPS timestep for the given LAMMPS ``units`` style."""
    try:
        return _LAMMPS_TIME_TO_PS[units.lower()]
    except KeyError:
        raise EstError(
            f"can't convert the LAMMPS timestep to ps for 'units {units}'; "
            "set simulation.timestep (in ps) in estk.yaml"
        ) from None


def kbt(temperature: float, energy_units: str = "kj/mol") -> float:
    """Thermal energy k_B T in ``energy_units``."""
    return R_KJ_MOL_K * temperature / energy_factor(energy_units)


def _factor(units: str, table: dict[str, float], kind: str) -> float:
    key = units.strip().lower()
    if key in table:
        return table[key]
    try:
        return float(key)
    except ValueError:
        known = ", ".join(table)
        raise EstError(f"unknown {kind} unit '{units}' (known: {known}, or a number)") from None
