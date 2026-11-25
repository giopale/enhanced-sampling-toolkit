"""Proton-transfer CVs, as ``ectoolkits.analysis.proton_transfer_cv.ProtonTransferCV``.

For a pair of oxygen sets (type 1, e.g. an adsorbed O; type 2, e.g. a surface site), each frame:

1. every H belongs to its nearest O (all oxygens of the system);
2. candidate chains: donor O - its H - acceptor O (``direct``), or donor O - its H - bridge
   water O - its H - acceptor O (``indirect``), with type 1 donating to type 2 and the
   reverse;
3. a chain counts if every hop has an O-O distance below 3.5 Å and an angle at the donor,
   between the acceptor and the H, of at most 30°;
4. for each chain, ``delta`` is the mean over its hops of d(donor, H) - d(acceptor, H) and
   ``d`` the mean donor-acceptor distance; the chain with the smallest |delta| is taken, its
   delta negated if a type 2 oxygen donates (so delta < 0: the proton is on the type 1 side,
   > 0: on the type 2 side). Without any chain both are NaN.

Bridges and acceptors farther than 3.5 Å never pass step 3, so they are left out before
building the chains; the chains keep ECToolkits' order, so ties are resolved the same way.
Distances use the minimum image. Atom numbers are LAMMPS ids (1-based).

One difference: when the bridge waters include the donor or the acceptor (e.g. ``indirect``
with all waters as acceptors and as bridges), ECToolkits also builds chains whose bridge is
the donor or the acceptor. Such a chain has a hop of length 0, and MDAnalysis' angle for it is
atan2(0, dot) = 0° or 180° depending on the sign of a zero (180° exactly when the three
components of the donor-H vector are all negative), so these chains pass or not by accident,
and when they pass delta and d are half of those of a direct transfer. Here a bridge is never
the donor or the acceptor; ``ectoolkits_compat=True`` reproduces ECToolkits, accident
included, e.g. to compare with CV files written by the old scripts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from estk.cv.builtin import Print
from estk.errors import EstError
from estk.io.lammpstrj import Frame

KINDS = ("direct", "indirect")
HBOND_DISTANCE = 3.5  # Å
HBOND_ANGLE = 30.0  # degrees


@dataclass
class ProtonTransfer:
    """One proton-transfer CV pair (delta, d), evaluated on dump frames."""

    kind: str
    type1: list[int]  # atom ids
    type2: list[int]
    water: list[int] = field(default_factory=list)  # bridge candidates (indirect)
    name: str = "pt"  # values are <kind>-delta-<name> and <kind>-d-<name>
    file: str = "CV-pt"
    type_map: Mapping[int, str] = field(default_factory=dict)  # LAMMPS type -> element
    ectoolkits_compat: bool = False  # also accept bridges equal to the donor (see above)
    atoms: None = None  # every atom is needed (the H-O pairing uses all of them)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise EstError(f"kind must be direct or indirect, got '{self.kind}'")
        if not self.type1 or not self.type2:
            raise EstError("proton transfer needs donor and acceptor atoms")
        if self.kind == "indirect" and not self.water:
            raise EstError("indirect proton transfer needs the bridge water oxygens")
        self._elements: tuple[np.ndarray, np.ndarray] | None = None
        self.last_chain: tuple[int, ...] | None = None  # the chain chosen in the last frame

    @property
    def values(self) -> list[str]:
        return [f"{self.kind}-delta-{self.name}", f"{self.kind}-d-{self.name}"]

    @property
    def prints(self) -> list[Print]:
        return [Print(file=self.file, args=self.values)]

    @property
    def columns(self) -> tuple[str, ...]:
        return ("type",) if self.type_map else ("element",)

    def evaluate_frame(self, frame: Frame) -> dict[str, float]:
        delta, d = self.evaluate(frame)
        return dict(zip(self.values, (delta, d), strict=True))

    def evaluate(self, frame: Frame) -> tuple[float, float]:
        oxygens, hydrogens = self._oxygen_hydrogen(frame)
        ids, r, box = frame.ids, frame.positions, frame.box
        row = {int(i): k for k, i in enumerate(ids)}

        # 1. each H to its nearest O
        to_o = box.minimum_image(r[oxygens][None, :, :] - r[hydrogens][:, None, :])
        nearest = oxygens[np.argmin(np.einsum("hok,hok->ho", to_o, to_o), axis=1)]
        bonded: dict[int, list[int]] = {}
        for h, o in zip(hydrogens, nearest, strict=True):
            bonded.setdefault(int(o), []).append(int(h))

        def rows(atom_ids: Sequence[int]) -> list[int]:
            try:
                return [row[a] for a in atom_ids]
            except KeyError as exc:
                raise EstError(f"atom {exc.args[0]} is not in the dump") from None

        t1, t2, water = rows(self.type1), rows(self.type2), rows(self.water)
        type2_rows = set(t2)
        # O-O distances among the oxygens involved, once per frame
        involved = np.unique(np.array(t1 + t2 + water))
        slot = {int(k): i for i, k in enumerate(involved)}
        o_o = box.minimum_image(r[involved][None, :, :] - r[involved][:, None, :])
        close = np.einsum("ijk,ijk->ij", o_o, o_o) < HBOND_DISTANCE**2

        def near(a: int, others: list[int]) -> list[int]:
            within = close[slot[a]]
            return [o for o in others if within[slot[o]]]

        # 2. candidate chains, in ECToolkits' order: type1 -> type2, then type2 -> type1
        chains = []
        for donors, acceptors in ((t1, t2), (t2, t1)):
            for donor in donors:
                if self.kind == "direct":
                    for acceptor in near(donor, acceptors):
                        for h in bonded.get(donor, []):
                            chains.append((donor, h, acceptor))
                else:
                    others = water if self.ectoolkits_compat else [w for w in water if w != donor]
                    bridges = near(donor, others)
                    reach = {b: set(near(b, acceptors)) for b in bridges}
                    if not self.ectoolkits_compat:
                        reach = {b: a - {b} for b, a in reach.items()}
                    for acceptor in acceptors:
                        for bridge in bridges:
                            if acceptor not in reach[bridge]:
                                continue
                            for h1 in bonded.get(donor, []):
                                for h2 in bonded.get(bridge, []):
                                    chains.append((donor, h1, bridge, h2, acceptor))
        if not chains:
            return np.nan, np.nan

        # 3.-4. hydrogen bonds, delta and d
        c = np.array(chains)
        hops = (c.shape[1] - 1) // 2
        ok = np.ones(len(c), dtype=bool)
        deltas, distances = [], []
        for k in range(hops):
            donor, h, acceptor = r[c[:, 2 * k]], r[c[:, 2 * k + 1]], r[c[:, 2 * k + 2]]
            to_acceptor = box.minimum_image(acceptor - donor)
            to_h = box.minimum_image(h - donor)
            from_acceptor_to_h = box.minimum_image(h - acceptor)
            d_oo = np.linalg.norm(to_acceptor, axis=1)
            d_oh = np.linalg.norm(to_h, axis=1)
            # the angle as MDAnalysis computes it, atan2(|u x v|, u . v); for a hop of length 0
            # (only in ECToolkits-compatible chains) its signed-zero result: 180° if all
            # components of the donor-H vector are negative, else 0°
            cross = np.linalg.norm(np.cross(to_acceptor, to_h), axis=1)
            angle = np.degrees(np.arctan2(cross, np.einsum("ik,ik->i", to_acceptor, to_h)))
            zero = d_oo == 0
            angle[zero] = np.where(np.all(to_h[zero] < 0, axis=1), 180.0, 0.0)
            ok &= (d_oo < HBOND_DISTANCE) & (angle <= HBOND_ANGLE)
            deltas.append(d_oh - np.linalg.norm(from_acceptor_to_h, axis=1))
            distances.append(d_oo)
        if not ok.any():
            return np.nan, np.nan
        delta = np.mean(deltas, axis=0)[ok]
        d = np.mean(distances, axis=0)[ok]
        best = int(np.argmin(np.abs(delta)))
        self.last_chain = tuple(int(ids[k]) for k in c[ok][best])  # atom ids, donor first
        sign = -1.0 if int(c[ok][best, 0]) in type2_rows else 1.0
        return float(sign * delta[best]), float(d[best])

    def _oxygen_hydrogen(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        if self._elements is None:  # all atoms, sorted by id: the same rows in every frame
            if self.type_map:
                types = frame.columns["type"].astype(int)
                unknown = sorted(set(types) - set(self.type_map))
                if unknown:
                    raise EstError(f"atom types {unknown} are not in atoms.type_map")
                elements = np.array([self.type_map[t] for t in types])
            else:
                elements = frame.columns["element"]
            oxygens = np.flatnonzero(elements == "O")
            hydrogens = np.flatnonzero(elements == "H")
            if not len(oxygens) or not len(hydrogens):
                raise EstError("no O or no H atoms found; check atoms.type_map in estk.yaml")
            self._elements = (oxygens, hydrogens)
        return self._elements
