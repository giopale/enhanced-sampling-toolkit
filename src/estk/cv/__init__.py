"""CVs computed after the simulation, from the trajectory (instead of ``plumed driver``).

``compute_cvs`` reads a run's LAMMPS dumps frame by frame (each walker's own dump), evaluates
the values printed by a PLUMED input with the built-in CVs (``estk.cv.builtin``), and writes
each PRINT's file in the run folder (``FILE.<i>`` for walker i), with time = step x timestep
in the time units of the run's COLVAR. Times come from the dump's step numbers, not from
counting frames, so they line up with COLVAR even when a restart wrote some steps twice
(estk keeps the last occurrence when reading).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from estk.cv.builtin import CVProgram
from estk.cv.proton_transfer import ProtonTransfer
from estk.errors import EstError
from estk.io.lammpstrj import Frame, iter_frames
from estk.io.plumed import walker_file
from estk.logging import get_logger

if TYPE_CHECKING:
    from estk.project import Run

log = get_logger(__name__)

Program = CVProgram | ProtonTransfer

__all__ = ["CVProgram", "ComputedFile", "ProtonTransfer", "compute_cvs"]


@dataclass
class ComputedFile:
    run: str
    walker: int | None
    path: Path
    frames: int
    first_step: int | None
    last_step: int | None


def compute_cvs(
    runs: Sequence[Run],
    programs: Program | Path | str | Sequence[Program | Path | str],
    *,
    dump: str = "traj.lammpstrj",
    every: int = 1,
    jobs: int = 1,
) -> list[ComputedFile]:
    """Evaluate the CVs of ``programs`` (PLUMED input files, ``CVProgram`` or
    ``ProtonTransfer`` objects) on every frame of each run's dump(s), reading each dump once.

    ``dump`` is the dump's name in the run folder; a walker's is found like its other files
    (``traj.3.lammpstrj`` or ``traj.lammpstrj.3``). ``every`` keeps every n-th frame.
    """
    if isinstance(programs, (str, Path, CVProgram, ProtonTransfer)):
        programs = [programs]
    programs = [CVProgram.from_plumed(p) if isinstance(p, (str, Path)) else p for p in programs]
    if every < 1:
        raise EstError(f"every must be at least 1, got {every}")
    prints = [p for program in programs for p in program.prints]
    files = [p.file for p in prints]
    if len(set(files)) != len(files):
        raise EstError(f"two outputs have the same file: {', '.join(files)}")
    tasks = []
    for run in runs:
        meta = run.metadata
        if meta.timestep is None:
            raise EstError(f"{run.name}: timestep unknown; set simulation.timestep in estk.yaml")
        time_per_step = meta.timestep / meta.time_factor  # in the time units of COLVAR
        for walker in run.walker_ids:
            source = walker_file(run.path, dump, walker)
            # a walker writes FILE.<i>, the name estk looks for first (as PLUMED's replicas)
            outputs = [
                (run.path / p.file).with_name(f"{Path(p.file).name}.{walker}")
                if walker is not None else run.path / p.file
                for p in prints
            ]  # fmt: skip
            tasks.append((run.name, walker, source, outputs, time_per_step))

    def submit(pool: ProcessPoolExecutor | None):
        for name, walker, source, outputs, dt in tasks:
            args = (programs, source, outputs, dt, every)
            yield name, walker, outputs, (pool.submit(_compute, *args) if pool else _compute(*args))

    results = []
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            pending = list(submit(pool))
            for name, walker, outputs, future in pending:
                frames, first, last = future.result()
                results += [ComputedFile(name, walker, o, frames, first, last) for o in outputs]
    else:
        for name, walker, outputs, (frames, first, last) in submit(None):
            results += [ComputedFile(name, walker, o, frames, first, last) for o in outputs]
    return results


def _compute(
    programs: list[Program],
    source: Path,
    outputs: list[Path],
    time_per_step: float,
    every: int,
) -> tuple[int, int | None, int | None]:
    # read only the atoms needed when there is one program that names them; else all atoms
    single = programs[0] if len(programs) == 1 and programs[0].atoms is not None else None
    columns = sorted({c for program in programs for c in program.columns})
    names = [n for program in programs for n in program.values]
    rows: list[list[float]] = []
    steps: list[int] = []
    for k, frame in enumerate(iter_frames(source, single.atoms if single else None, columns)):
        if k % every:
            continue
        values: dict[str, float] = {}
        for program in programs:
            values.update(program.evaluate_frame(_subset(frame, program.atoms)))
        steps.append(frame.step)
        rows.append([frame.step * time_per_step] + [values[n] for n in names])
    if not rows:
        raise EstError(f"{source}: no frames")
    table = np.array(rows)
    prints = [p for program in programs for p in program.prints]
    for p, output in zip(prints, outputs, strict=True):
        keep = np.arange(len(table)) % p.stride == 0
        index = [0] + [1 + names.index(a) for a in p.args]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output, table[keep][:, index], fmt="%.10g",
                   header=f"FIELDS time {' '.join(p.args)}", comments="#! ")  # fmt: skip
    log.info("%s: %d frames, steps %d to %d", source.name, len(rows), steps[0], steps[-1])
    return len(rows), steps[0], steps[-1]


def _subset(frame: Frame, atoms: np.ndarray | None) -> Frame:
    """The frame restricted to ``atoms`` (ids), in that order; the frame itself if it holds
    exactly those atoms or ``atoms`` is None."""
    if atoms is None or np.array_equal(frame.ids, atoms):
        return frame
    rows = np.searchsorted(frame.ids, atoms)  # frames of all atoms are sorted by id
    return Frame(step=frame.step, box=frame.box, ids=frame.ids[rows],
                 positions=frame.positions[rows],
                 columns={c: v[rows] for c, v in frame.columns.items()})  # fmt: skip
