"""Reading LAMMPS text dumps (``dump custom``, ``*.lammpstrj``), frame by frame.

Any custom dump works: the ``ITEM: ATOMS`` line says which columns there are. Needed are
``id`` and one set of coordinates: ``x y z`` (wrapped), ``xu yu zu`` (unwrapped), ``xs ys zs``
or ``xsu ysu zsu`` (scaled). Other columns (type, element, forces, ...) are ignored, atoms can
be in any order (``dump_modify sort id`` makes reading faster) and the box can be orthogonal or
triclinic. Files ending in ``.gz`` are read compressed.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import IO

import numpy as np

from estk.errors import EstError, MissingFileError

COORDINATES = (("x", "y", "z"), ("xu", "yu", "zu"), ("xs", "ys", "zs"), ("xsu", "ysu", "zsu"))


@dataclass
class Box:
    """A periodic cell: origin ``lo`` and cell vectors as the rows of ``matrix``."""

    lo: np.ndarray  # (3,)
    matrix: np.ndarray  # (3, 3): rows a, b, c
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def minimum_image(self, delta: np.ndarray) -> np.ndarray:
        """Shortest periodic images of displacement vectors (..., 3)."""
        frac = delta @ np.linalg.inv(self.matrix)
        for k in range(3):
            if self.periodic[k]:
                frac[..., k] -= np.round(frac[..., k])
        return frac @ self.matrix


@dataclass
class Frame:
    step: int
    box: Box
    ids: np.ndarray  # atom ids (LAMMPS ids = PLUMED serial numbers) of ``positions``
    positions: np.ndarray  # (atoms, 3) Cartesian, in the order of ``ids``


def iter_frames(path: Path | str, atoms: Sequence[int] | None = None) -> Iterator[Frame]:
    """Yield the frames of a dump; with ``atoms``, only those ids, in that order."""
    path = Path(path)
    if not path.is_file():
        raise MissingFileError(f"dump file not found: {path}")
    wanted = None if atoms is None else np.asarray(atoms, dtype=np.int64)
    with _open(path) as f:
        order_key: str | None = None
        rows: np.ndarray | None = None
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                if line.strip():
                    raise EstError(f"{path}: expected 'ITEM: TIMESTEP', got {line.strip()!r}")
                continue
            step = int(f.readline().split()[0])
            _expect(f, "ITEM: NUMBER OF ATOMS", path)
            n_atoms = int(f.readline())
            box = _read_box(f, path)
            header = f.readline()
            if not header.startswith("ITEM: ATOMS"):
                raise EstError(f"{path}: expected 'ITEM: ATOMS', got {header.strip()!r}")
            columns = header.split()[2:]
            i_id, i_xyz, scaled = _columns(columns, path)
            block = list(islice(f, n_atoms))
            if len(block) < n_atoms:  # the last frame was cut off
                return
            tokens = np.array("".join(block).split()).reshape(n_atoms, len(columns))
            ids_text = tokens[:, i_id]
            key = " ".join(ids_text)  # same atom order as the previous frame?
            if key != order_key or rows is None:
                ids = ids_text.astype(np.int64)
                if wanted is None:
                    rows = np.argsort(ids)
                else:
                    position = {int(i): r for r, i in enumerate(ids)}
                    missing = [int(a) for a in wanted if int(a) not in position]
                    if missing:
                        raise EstError(f"{path}: step {step}: no atom(s) {missing[:5]}")
                    rows = np.array([position[int(a)] for a in wanted])
                order_key = key
                frame_ids = ids[rows]
            xyz = tokens[rows][:, i_xyz].astype(float)
            if scaled:
                xyz = box.lo + xyz @ box.matrix
            yield Frame(step=step, box=box, ids=frame_ids, positions=xyz)


def _open(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")  # type: ignore[return-value]
    return open(path)


def _expect(f: IO[str], start: str, path: Path) -> None:
    line = f.readline()
    if not line.startswith(start):
        raise EstError(f"{path}: expected '{start}', got {line.strip()!r}")


def _read_box(f: IO[str], path: Path) -> Box:
    header = f.readline()
    if not header.startswith("ITEM: BOX BOUNDS"):
        raise EstError(f"{path}: expected 'ITEM: BOX BOUNDS', got {header.strip()!r}")
    words = header.split()[3:]
    triclinic = words[:3] == ["xy", "xz", "yz"]
    flags = words[3:] if triclinic else words
    periodic = tuple(flag.startswith("p") for flag in flags[:3]) if len(flags) >= 3 else (True,) * 3
    bounds = np.array([[float(v) for v in f.readline().split()] for _ in range(3)])
    if triclinic:
        xy, xz, yz = bounds[0, 2], bounds[1, 2], bounds[2, 2]
        xlo = bounds[0, 0] - min(0.0, xy, xz, xy + xz)
        xhi = bounds[0, 1] - max(0.0, xy, xz, xy + xz)
        ylo = bounds[1, 0] - min(0.0, yz)
        yhi = bounds[1, 1] - max(0.0, yz)
        zlo, zhi = bounds[2, 0], bounds[2, 1]
        matrix = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
        lo = np.array([xlo, ylo, zlo])
    else:
        lo = bounds[:, 0]
        matrix = np.diag(bounds[:, 1] - bounds[:, 0])
    return Box(lo=lo, matrix=matrix, periodic=periodic)  # type: ignore[arg-type]


def _columns(columns: list[str], path: Path) -> tuple[int, list[int], bool]:
    if "id" not in columns:
        raise EstError(f"{path}: the dump has no 'id' column (columns: {' '.join(columns)})")
    for names in COORDINATES:
        if all(n in columns for n in names):
            return columns.index("id"), [columns.index(n) for n in names], names[0].startswith("xs")
    raise EstError(f"{path}: no coordinates in the dump (need x y z, xu yu zu, xs ys zs or "
                   f"xsu ysu zsu; columns: {' '.join(columns)})")  # fmt: skip
