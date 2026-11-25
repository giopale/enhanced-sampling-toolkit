"""Hydrogen bonds of one oxygen (or a few) with a set of partner oxygens, per frame.

The criterion is that of the proton-transfer CVs (``estk.cv.proton_transfer``): every H
belongs to its nearest O (all oxygens of the system), and donor O - H ... acceptor O is a
hydrogen bond when the O-O distance is below ``distance`` (3.5 Å) and the angle at the donor,
between the H and the acceptor, is at most ``angle`` (30°). Distances use the minimum image.

For the ``center`` oxygen(s) and the ``partners``, each frame gives (with ``<name>``):

- ``hb-n-<name>``: the number of hydrogen bonds (each donor-H-acceptor triple once);
- ``hb-doo-<name>``: the O-O distance of the shortest one (Å);
- ``hb-angle-<name>``: the angle (degrees, 0-180) between that bond's O-O vector, from the
  center to the partner, and ``axis`` (e.g. y);
- ``hb-role-<name>``: +1 if the center donates the proton in that bond, -1 if it accepts it.

``direction`` keeps only the bonds where the center donates (``donor``), accepts
(``acceptor``), or both (default). Without any bond, all but the number are NaN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from estk.cv.builtin import Print
from estk.cv.proton_transfer import HBOND_ANGLE, HBOND_DISTANCE, oxygens_and_hydrogens
from estk.errors import EstError
from estk.io.lammpstrj import Frame

DIRECTIONS = ("both", "donor", "acceptor")
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def parse_axis(text: str) -> np.ndarray:
    """``x``, ``y``, ``z``, or a vector ``a,b,c`` -> a unit vector."""
    if text.lower() in AXES:
        return np.array(AXES[text.lower()])
    try:
        vector = np.array([float(v) for v in text.split(",")])
    except ValueError:
        vector = np.empty(0)
    if vector.shape != (3,) or not np.linalg.norm(vector) > 0:
        raise EstError(f"axis {text!r}: give x, y, z or a vector like 0,1,0")
    return vector / np.linalg.norm(vector)


@dataclass
class HBonds:
    """The hydrogen bonds between ``center`` and ``partners`` (atom ids), on dump frames."""

    center: list[int]
    partners: list[int]
    axis: np.ndarray = field(default_factory=lambda: np.array(AXES["y"]))
    name: str = "hb"
    file: str = "CV-hb"
    direction: str = "both"
    distance: float = HBOND_DISTANCE
    angle: float = HBOND_ANGLE
    type_map: Mapping[int, str] = field(default_factory=dict)
    atoms: None = None  # every atom is needed (the H-O pairing uses all of them)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise EstError(f"direction must be one of {', '.join(DIRECTIONS)}")
        if not self.center or not self.partners:
            raise EstError("hydrogen bonds need center and partner oxygens")
        self.partners = [p for p in self.partners if p not in set(self.center)]
        self.axis = np.asarray(self.axis, dtype=float) / np.linalg.norm(self.axis)
        self._elements: tuple[np.ndarray, np.ndarray] | None = None
        self.last_bonds: list[tuple[int, int, int]] = []  # (donor, H, acceptor), last frame

    @property
    def values(self) -> list[str]:
        return [f"hb-{v}-{self.name}" for v in ("n", "doo", "angle", "role")]

    @property
    def prints(self) -> list[Print]:
        return [Print(file=self.file, args=self.values)]

    @property
    def columns(self) -> tuple[str, ...]:
        return ("type",) if self.type_map else ("element",)

    def evaluate_frame(self, frame: Frame) -> dict[str, float]:
        return dict(zip(self.values, self.evaluate(frame), strict=True))

    def evaluate(self, frame: Frame) -> tuple[float, float, float, float]:
        if self._elements is None:
            self._elements = oxygens_and_hydrogens(frame, self.type_map)
        oxygens, hydrogens = self._elements
        ids, r, box = frame.ids, frame.positions, frame.box
        row = {int(i): k for k, i in enumerate(ids)}
        try:
            centers = [row[a] for a in self.center]
            partners = np.array([row[a] for a in self.partners])
        except KeyError as exc:
            raise EstError(f"atom {exc.args[0]} is not in the dump") from None

        # each H to its nearest O
        to_o = box.minimum_image(r[oxygens][None, :, :] - r[hydrogens][:, None, :])
        nearest = oxygens[np.argmin(np.einsum("hok,hok->ho", to_o, to_o), axis=1)]
        owned: dict[int, list[int]] = {}
        for h, o in zip(hydrogens, nearest, strict=True):
            owned.setdefault(int(o), []).append(int(h))

        cos_max = np.cos(np.radians(self.angle))
        bonds = []  # (d_OO, center->partner vector, role, donor, H, acceptor)
        for c in centers:
            vectors = box.minimum_image(r[partners] - r[c])
            d_oo = np.linalg.norm(vectors, axis=1)
            for k in np.flatnonzero(d_oo < self.distance):
                p, v, d = int(partners[k]), vectors[k], float(d_oo[k])
                if self.direction in ("both", "donor"):  # c - H ... p
                    for h in owned.get(c, []):
                        to_h = box.minimum_image(r[h] - r[c])
                        if np.dot(to_h, v) >= cos_max * np.linalg.norm(to_h) * d:
                            bonds.append((d, v, 1.0, c, h, p))
                if self.direction in ("both", "acceptor"):  # p - H ... c
                    for h in owned.get(p, []):
                        to_h = box.minimum_image(r[h] - r[p])
                        if np.dot(to_h, -v) >= cos_max * np.linalg.norm(to_h) * d:
                            bonds.append((d, v, -1.0, p, h, c))
        self.last_bonds = [(int(ids[a]), int(ids[h]), int(ids[b])) for *_, a, h, b in bonds]
        if not bonds:
            return 0.0, np.nan, np.nan, np.nan
        d, v, role, *_ = min(bonds, key=lambda b: b[0])
        cos = float(np.clip(np.dot(v, self.axis) / d, -1.0, 1.0))
        return float(len(bonds)), d, float(np.degrees(np.arccos(cos))), role


def ids_of(spec: str, groups: Mapping[str, Sequence[int]]) -> list[int]:
    """A group of atoms.groups, or atom ids like ``262`` or ``153-216,252-261,263``."""
    if spec in groups:
        return list(groups[spec])
    ids: list[int] = []
    try:
        for part in spec.split(","):
            lo, _, hi = part.strip().partition("-")
            ids += list(range(int(lo), int(hi or lo) + 1))
    except ValueError:
        known = ", ".join(groups) or "none in estk.yaml"
        raise EstError(f"{spec!r}: not a group of atoms.groups ({known}) nor atom ids") from None
    return ids
