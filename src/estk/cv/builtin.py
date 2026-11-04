"""CVs from a PLUMED input, computed in numpy (no PLUMED needed).

Supported: ``UNITS`` (LENGTH), ``GROUP ATOMS=...``, ``DISTANCE ATOMS=a,b``, ``DISTANCES
GROUPA=... GROUPB=...`` with ``MIN={BETA=b}`` or ``MAX={BETA=b}``, and ``PRINT ARG=... FILE=...
STRIDE=...``. Distances use the minimum image (``NOPBC`` to switch it off). The smooth minimum
and maximum are PLUMED's: ``b / log(sum_i exp(b / r_i))`` and ``b log(sum_i exp(r_i / b))``.
Other actions are ignored unless a printed value needs them.

Atom numbers are PLUMED's serial numbers, which ``fix plumed`` takes from the LAMMPS atom ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from estk.errors import EstError
from estk.io.lammpstrj import Box
from estk.io.plumed import Action, parse_actions

# Length units of PLUMED's UNITS LENGTH=..., in Å.
_LENGTH_ANGSTROM = {"nm": 10.0, "a": 1.0, "ang": 1.0, "angstrom": 1.0, "um": 1e4, "bohr": 0.529177}


@dataclass
class Print:
    file: str
    args: list[str]
    stride: int = 1


@dataclass
class _Distance:
    a: np.ndarray  # atom ids
    b: np.ndarray
    pbc: bool
    reduce: str | None = None  # "min", "max" or None (one pair)
    beta: float = 1.0


@dataclass
class CVProgram:
    """The CVs that a PLUMED input prints, ready to be evaluated on frames."""

    values: dict[str, _Distance]  # printed value name (label or label.min) -> definition
    prints: list[Print]
    length_unit: float = 10.0  # Å per PLUMED length unit (PLUMED's default is nm)
    atoms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    @classmethod
    def from_plumed(cls, path: Path | str) -> CVProgram:
        path = Path(path)
        actions = parse_actions(path)
        groups: dict[str, list[int]] = {}
        definitions: dict[str, tuple[Action, str]] = {}  # value name -> (action, reduction)
        prints: list[Print] = []
        length_unit = 10.0
        for action in actions:
            if action.name == "UNITS" and action.get("LENGTH"):
                unit = action.get("LENGTH").lower()  # type: ignore[union-attr]
                if unit not in _LENGTH_ANGSTROM:
                    raise EstError(f"{path}: UNITS LENGTH={action.get('LENGTH')} not supported")
                length_unit = _LENGTH_ANGSTROM[unit]
            elif action.name == "GROUP" and action.label:
                groups[action.label] = _atoms(action.get("ATOMS") or "", groups, path)
            elif action.name == "PRINT":
                prints.append(Print(
                    file=action.get("FILE") or "COLVAR",
                    args=action.get_list("ARG"),
                    stride=int(action.get("STRIDE") or 1),
                ))  # fmt: skip
            elif action.label:
                for reduction in ("MIN", "MAX"):
                    if action.get(reduction) is not None:
                        definitions[f"{action.label}.{reduction.lower()}"] = (action, reduction)
                definitions.setdefault(action.label, (action, ""))
        if not prints:
            raise EstError(f"{path}: no PRINT action, so nothing to compute")
        values = {}
        for p in prints:
            for arg in p.args:
                if arg not in values:
                    values[arg] = _define(arg, definitions, groups, path)
        atoms = np.unique(np.concatenate([np.r_[d.a, d.b] for d in values.values()]))
        return cls(values=values, prints=prints, length_unit=length_unit, atoms=atoms)

    def evaluate(self, positions: np.ndarray, box: Box) -> dict[str, float]:
        """Every printed value for one frame; ``positions`` (Å) are those of ``self.atoms``."""
        index = {int(a): k for k, a in enumerate(self.atoms)}
        out = {}
        for name, d in self.values.items():
            ra = positions[[index[int(a)] for a in d.a]]
            rb = positions[[index[int(b)] for b in d.b]]
            delta = rb[None, :, :] - ra[:, None, :]
            if d.pbc:
                delta = box.minimum_image(delta)
            r = np.linalg.norm(delta, axis=-1).reshape(-1) / self.length_unit
            if d.reduce == "min":
                x = d.beta / r
                peak = x.max()
                out[name] = float(d.beta / (peak + np.log(np.sum(np.exp(x - peak)))))
            elif d.reduce == "max":
                x = r / d.beta
                peak = x.max()
                out[name] = float(d.beta * (peak + np.log(np.sum(np.exp(x - peak)))))
            else:
                out[name] = float(r[0])
        return out


def _define(
    arg: str, definitions: dict[str, tuple[Action, str]], groups: dict[str, list[int]], path: Path
) -> _Distance:
    if arg not in definitions:
        raise EstError(f"{path}: PRINT ARG={arg}: no action defines it")
    action, reduction = definitions[arg]
    pbc = "NOPBC" not in action.flags
    if action.name == "DISTANCE":
        atoms = _atoms(action.get("ATOMS") or "", groups, path)
        if len(atoms) != 2:
            raise EstError(f"{path}: DISTANCE {action.label} needs ATOMS with two atoms")
        return _Distance(np.array(atoms[:1]), np.array(atoms[1:]), pbc)
    if action.name == "DISTANCES":
        if not reduction:
            raise EstError(f"{path}: DISTANCES {action.label}: give MIN={{BETA=...}} or "
                           "MAX={BETA=...} and print .min or .max")  # fmt: skip
        if action.get("GROUPA") is None or action.get("GROUPB") is None:
            raise EstError(f"{path}: DISTANCES {action.label}: only GROUPA/GROUPB is supported")
        spec = action.get(reduction) or ""
        match = re.fullmatch(r"\s*BETA\s*=\s*([^\s}]+)\s*", spec)
        if match is None:
            raise EstError(f"{path}: {action.label}: {reduction}={{{spec}}}: expected BETA=...")
        return _Distance(
            np.array(_atoms(action.get("GROUPA") or "", groups, path)),
            np.array(_atoms(action.get("GROUPB") or "", groups, path)),
            pbc, reduction.lower(), float(match.group(1)),
        )  # fmt: skip
    raise EstError(f"{path}: {action.name} ({arg}) is not computed by estk's built-in CVs "
                   "(DISTANCE, DISTANCES with MIN/MAX); use plumed driver for it")  # fmt: skip


def _atoms(spec: str, groups: dict[str, list[int]], path: Path) -> list[int]:
    """``1-152,310,O2cup`` -> atom numbers (ranges ``a-b`` or ``a-b:stride``, group labels)."""
    atoms: list[int] = []
    for item in (s.strip() for s in spec.split(",") if s.strip()):
        if item in groups:
            atoms += groups[item]
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", item)
        if match is None:
            raise EstError(f"{path}: atoms '{item}' not understood (numbers, a-b, a-b:n, groups)")
        lo = int(match.group(1))
        hi = int(match.group(2) or lo)
        atoms += list(range(lo, hi + 1, int(match.group(3) or 1)))
    if not atoms:
        raise EstError(f"{path}: empty atom list")
    return atoms
