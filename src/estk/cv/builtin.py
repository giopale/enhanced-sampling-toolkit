"""CVs from a PLUMED input, computed in numpy (no PLUMED needed).

Supported: ``UNITS`` (LENGTH), ``GROUP ATOMS=...``, ``DISTANCE ATOMS=a,b``, ``DISTANCES
GROUPA=... GROUPB=...`` with ``MIN={BETA=b}`` or ``MAX={BETA=b}``, ``COORDINATION`` (below), and
``PRINT ARG=... FILE=... STRIDE=...``. Distances use the minimum image (``NOPBC`` to switch it
off). The smooth minimum and maximum are PLUMED's: ``b / log(sum_i exp(b / r_i))`` and
``b log(sum_i exp(r_i / b))``. Other actions are ignored unless a printed value needs them.

``COORDINATION GROUPA=... [GROUPB=...] [PAIR]`` is PLUMED's: the sum over pairs of the rational
switching function ``s(r) = (1 - x^n) / (1 - x^m)``, ``x = (r - D_0) / R_0`` (1 for r < D_0).
The pairs are every A atom with every B atom (the same atom twice is skipped), the pairs within
A when there is no GROUPB, or A[i] with B[i] with ``PAIR``. The switching function is given as
``R_0= NN= MM= D_0=`` (NN default 6, MM default 2 NN, D_0 default 0) or as
``SWITCH={RATIONAL R_0= NN= MM= D_0= D_MAX=}``. As in PLUMED (checked against its source,
2.5 to 2.9), the function is cut at d_max and stretched, ``(s(r) - s(d_max)) / (s(0) -
s(d_max))``, so that it goes smoothly to 0 there: d_max is ``D_0 + R_0 * 1e-5^(1/(NN-MM))``
(where s is about 1e-5) for the first form, D_MAX for the second (no cut without D_MAX; no
stretch with NOSTRETCH).

Atom numbers are PLUMED's serial numbers, which ``fix plumed`` takes from the LAMMPS atom ids.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from estk.errors import EstError
from estk.io.lammpstrj import Box, Frame
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
class _Rational:
    """PLUMED's rational switching function, in PLUMED length units."""

    r0: float
    nn: int = 6
    mm: int = 0  # 0: 2 nn
    d0: float = 0.0
    dmax: float = np.inf
    stretch: bool = False

    def __post_init__(self) -> None:
        if self.mm == 0:
            self.mm = 2 * self.nn
        if self.r0 <= 0 or self.nn <= 0 or self.mm <= 0 or self.nn == self.mm:
            raise EstError(f"switching function: need R_0 > 0 and NN != MM > 0, got {self}")

    def raw(self, r: np.ndarray) -> np.ndarray:
        x = np.clip((np.asarray(r, dtype=float) - self.d0) / self.r0, 0.0, None)
        n, m = self.nn, self.mm
        out = np.empty_like(x)
        near = np.abs(x - 1.0) < 5e10 * np.finfo(float).eps  # 0/0 at x = 1: PLUMED's expansion
        dx = x[near] - 1.0
        second = n * (m * m - 3.0 * m * (n - 1) + n * (2 * n - 3)) / (6.0 * m)
        out[near] = n / m + dx * (0.5 * n * (n - m) / m + 0.5 * dx * second)
        low = (x < 1.0) & ~near
        out[low] = (1.0 - x[low] ** n) / (1.0 - x[low] ** m)
        high = (x > 1.0) & ~near  # in 1/x, which can't overflow
        y = 1.0 / x[high]
        out[high] = y ** (m - n) * (1.0 - y**n) / (1.0 - y**m)
        return out

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        value = self.raw(r)
        if self.stretch and np.isfinite(self.dmax):
            at_zero, at_max = self.raw(np.array([0.0, self.dmax]))
            value = (value - at_max) / (at_zero - at_max)
        return np.where(r > self.dmax, 0.0, value)


@dataclass
class _Coordination:
    a: np.ndarray  # atom ids
    b: np.ndarray  # atom ids; empty: the pairs within a
    pbc: bool
    switch: _Rational
    pair: bool = False  # a[i] with b[i] only

    def value(self, ra: np.ndarray, rb: np.ndarray, box: Box, length_unit: float) -> float:
        if self.pair:
            delta = rb - ra
            keep = self.a != self.b
        elif len(self.b) == 0:
            delta = ra[None, :, :] - ra[:, None, :]
            keep = np.triu(np.ones((len(self.a), len(self.a)), dtype=bool), k=1)
        else:
            delta = rb[None, :, :] - ra[:, None, :]
            keep = self.a[:, None] != self.b[None, :]
        if self.pbc:
            delta = box.minimum_image(delta)
        r = np.linalg.norm(delta, axis=-1)[keep] / length_unit
        return float(self.switch(r).sum())


@dataclass
class CVProgram:
    """The CVs that a PLUMED input prints, ready to be evaluated on frames."""

    values: dict[str, _Distance | _Coordination]  # printed name (label, label.min) -> definition
    prints: list[Print]
    length_unit: float = 10.0  # Å per PLUMED length unit (PLUMED's default is nm)
    atoms: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    columns: tuple[str, ...] = ()  # other dump columns needed (none)

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
                file = action.get("FILE") or "COLVAR"
                if "$" in file:
                    raise EstError(
                        f"{path}: PRINT FILE={file} contains '$', probably a shell variable left "
                        "from a script; give the plain name (estk adds .<walker> for walkers)"
                    )
                prints.append(Print(
                    file=file,
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

    def evaluate_frame(self, frame: Frame) -> dict[str, float]:
        return self.evaluate(frame.positions, frame.box)

    def evaluate(self, positions: np.ndarray, box: Box) -> dict[str, float]:
        """Every printed value for one frame; ``positions`` (Å) are those of ``self.atoms``."""
        index = {int(a): k for k, a in enumerate(self.atoms)}
        out = {}
        for name, d in self.values.items():
            ra = positions[[index[int(a)] for a in d.a]]
            rb = positions[[index[int(b)] for b in d.b]]
            if isinstance(d, _Coordination):
                out[name] = d.value(ra, rb, box, self.length_unit)
                continue
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
) -> _Distance | _Coordination:
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
    if action.name == "COORDINATION":
        if reduction:
            raise EstError(f"{path}: COORDINATION {action.label}: MIN/MAX not supported")
        return _coordination(action, groups, path, pbc)
    raise EstError(f"{path}: {action.name} ({arg}) is not computed by estk's built-in CVs "
                   "(DISTANCE, DISTANCES with MIN/MAX, COORDINATION); use plumed driver for "
                   "it")  # fmt: skip


def _coordination(
    action: Action, groups: dict[str, list[int]], path: Path, pbc: bool
) -> _Coordination:
    where = f"{path}: COORDINATION {action.label}"
    if action.get("GROUPA") is None:
        raise EstError(f"{where}: needs GROUPA (and GROUPB, or none for pairs within A)")
    a = np.array(_atoms(action.get("GROUPA") or "", groups, path))
    b = np.array(_atoms(action.get("GROUPB") or "", groups, path)) if action.get("GROUPB") \
        else np.empty(0, dtype=np.int64)  # fmt: skip
    pair = "PAIR" in action.flags
    if pair and len(a) != len(b):
        raise EstError(f"{where}: PAIR needs GROUPA and GROUPB of the same length")
    # NLIST, NL_CUTOFF, NL_STRIDE only speed PLUMED up; estk uses every pair
    spec = action.get("SWITCH")
    try:
        if spec is not None:
            words = spec.split()
            if not words or words[0].upper() != "RATIONAL":
                raise EstError(f"{where}: SWITCH={{{spec}}}: only RATIONAL is supported")
            options = dict(w.split("=", 1) for w in words[1:] if "=" in w)
            options = {k.upper(): v for k, v in options.items()}
            known = {"R_0", "NN", "MM", "D_0", "D_MAX"}
            if set(options) - known:
                raise EstError(f"{where}: SWITCH: {', '.join(sorted(set(options) - known))} "
                               "not supported")  # fmt: skip
            if "R_0" not in options:
                raise EstError(f"{where}: SWITCH needs R_0")
            switch = _Rational(
                r0=float(options["R_0"]), nn=int(options.get("NN", 6)),
                mm=int(options.get("MM", 0)), d0=float(options.get("D_0", 0.0)),
                dmax=float(options.get("D_MAX", np.inf)),
                stretch="D_MAX" in options and "NOSTRETCH" not in [w.upper() for w in words],
            )  # fmt: skip
        else:
            if action.get("R_0") is None:
                raise EstError(f"{where}: give R_0 (and NN, MM, D_0) or SWITCH={{RATIONAL ...}}")
            switch = _Rational(
                r0=float(action.get("R_0")),  # type: ignore[arg-type]
                nn=int(action.get("NN") or 6),
                mm=int(action.get("MM") or 0),
                d0=float(action.get("D_0") or 0.0),
            )
            # PLUMED cuts this form where the function has fallen to about 1e-5, and stretches
            switch.dmax = switch.d0 + switch.r0 * 1e-5 ** (1.0 / (switch.nn - switch.mm))
            switch.stretch = True
    except ValueError as exc:
        raise EstError(f"{where}: {exc}") from None
    return _Coordination(a, b, pbc, switch, pair)


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
