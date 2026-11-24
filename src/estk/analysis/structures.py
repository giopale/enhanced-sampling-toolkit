"""Configurations sampled from the free-energy basins, written as extended XYZ.

Within each basin, frames are drawn without replacement from the frames whose coordinates are
in the dump (``mode``):

- ``boltzmann`` (default): with probability proportional to the reweighting factor
  ``w_i = exp(V_i / kT)``, so that the sample follows the equilibrium (unbiased) distribution
  in the basin, as its free energy says: low-F parts more often than high-F ones. Each run's
  frames are normalised by the run's total weight and the runs combined like surfaces
  (``equal`` or ``probability``).
- ``uniform``: every frame of the basin equally likely: the biased ensemble, which also covers
  the higher-F parts of the basin (e.g. for training sets).

Each structure's header holds the CVs (and any other quantities) of that frame, its basin, run,
walker, step, time, and the free energy of the surface at its CVs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.analysis.basins import NO_DATA, BasinDefinition, Region, define_basins
from estk.errors import EstError
from estk.io.lammpstrj import Frame, dump_columns, iter_frames, scan_steps
from estk.io.plumed import walker_file
from estk.logging import get_logger
from estk.paths import evaluate
from estk.reweight.core import logsumexp
from estk.surface import Surface

if TYPE_CHECKING:
    from estk.project import Project

log = get_logger(__name__)

MODES = ("boltzmann", "uniform")


@dataclass
class SampledFrame:
    basin: str
    run: str
    walker: int | None
    step: int
    time_ps: float
    values: dict[str, float]  # CVs and quantities of the frame
    free_energy: float  # of the surface at the frame's CVs (surface units)
    log_weight: float  # V / kT


@dataclass
class BasinSample:
    """Frames drawn from each basin; ``write`` reads their coordinates and writes extxyz."""

    cvs: tuple[str, ...]
    quantities: list[str]
    frames: list[SampledFrame]
    units: str
    candidates: dict[str, int] = field(default_factory=dict)  # basin -> frames to draw from
    effective: dict[str, float] = field(default_factory=dict)  # basin -> Kish n of the draw
    metadata: dict[str, Any] = field(default_factory=dict)

    def write(
        self,
        project: Project,
        path: Path | str,
        *,
        dump: str = "traj.lammpstrj",
        type_map: Mapping[int, str] | None = None,
        units: str | None = None,
    ) -> Path:
        """Read the sampled frames from the runs' dumps and write them to one extxyz file.

        Species come from the dump's ``element`` column, else from its ``type`` column with
        ``type_map`` (default ``atoms.type_map`` of estk.yaml). Positions are those of the dump
        (wrapped or not), shifted so that the cell starts at the origin; atoms are in id order.
        ``units``: energy units of ``fes`` in the headers (default: those of the surface).
        """
        from estk.plotting.core import energy_scale

        factor, label = energy_scale(self.units, units, self.metadata.get("kbt"))
        path = Path(path)
        type_map = dict(type_map if type_map is not None else project.config.atoms.type_map)
        runs = {run.name: run for run in project.runs}
        wanted: dict[tuple[str, int | None], set[int]] = {}
        for f in self.frames:
            wanted.setdefault((f.run, f.walker), set()).add(f.step)
        read: dict[tuple[str, int | None, int], tuple[Frame, list[str]]] = {}
        for (run_name, walker), steps in wanted.items():
            source = walker_file(runs[run_name].path, dump, walker)
            columns = _species_columns(source)
            for frame in iter_frames(source, columns=columns, steps=steps):
                read[(run_name, walker, frame.step)] = (frame, _species(frame, type_map, source))
        missing = [f for f in self.frames if (f.run, f.walker, f.step) not in read]
        if missing:
            m = missing[0]
            raise EstError(f"{m.run}: step {m.step} not found in its dump ({len(missing)} missing)")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as out:
            for f in self.frames:
                frame, species = read[(f.run, f.walker, f.step)]
                out.write(_extxyz(frame, species, self._header(f, factor, label)))
        return path

    def _header(self, f: SampledFrame, factor: float, units: str) -> dict[str, Any]:
        header: dict[str, Any] = {key_name(n): v for n, v in f.values.items()}
        header.update({
            "basin": f.basin, "run": f.run, "step": f.step, "time_ps": f.time_ps,
            "fes": f.free_energy * factor, "fes_units": units, "sampling": self.metadata["mode"],
        })  # fmt: skip
        if f.walker is not None:
            header["walker"] = f.walker
        return header


def sample_basins(
    project: Project,
    cvs: Sequence[str],
    surface: Surface,
    *,
    n: int = 10,
    basins: Sequence[str] | None = None,
    regions: Mapping[str, Region] | None = None,
    min_depth_kt: float = 2.0,
    automatic: bool = False,
    quantities: Sequence[str] = (),
    sources: Mapping[str, str | Path] | None = None,
    time: tuple[float | None, float | None] | None = None,
    mode: str = "boltzmann",
    combine: str = "equal",
    dump: str = "traj.lammpstrj",
    seed: int | None = 0,
) -> BasinSample:
    """Draw ``n`` frames from each basin (all, or those named in ``basins``) of ``surface``.

    The basins are those of ``estk basins`` (``regions``/estk.yaml, or automatic with
    ``min_depth_kt``). Only frames inside the grid limits and present in the run's dump (each
    walker's own) can be drawn. ``quantities`` (with ``sources`` for other files) are written
    to the headers next to the CVs.
    """
    cvs = list(cvs)
    if mode not in MODES:
        raise EstError(f"mode must be one of {', '.join(MODES)}, got '{mode}'")
    if combine not in ("equal", "probability"):
        raise EstError(f"combine must be equal or probability, got '{combine}'")
    if n < 1:
        raise EstError(f"n must be at least 1, got {n}")
    definition: BasinDefinition = define_basins(
        project, cvs, surface, regions=regions, min_depth_kt=min_depth_kt, automatic=automatic
    )
    chosen = list(basins) if basins else list(definition.names)
    unknown = [b for b in chosen if b not in definition.names]
    if unknown:
        raise EstError(f"no basin {', '.join(unknown)} (basins: {', '.join(definition.names)})")
    quantities = [q for q in dict.fromkeys(quantities) if q not in cvs]
    extra = {q: f for q, f in (sources or {}).items() if q in quantities}
    loaded = [q for q in quantities if q not in extra]

    pool: list[dict[str, Any]] = []  # per run: the frames that can be drawn
    run_log_totals = []
    for run in project.runs:
        meta = run.metadata
        if meta.timestep is None:
            raise EstError(f"{run.name}: timestep unknown; set simulation.timestep in estk.yaml")
        table = run.load_cvs(cvs + loaded, time=time, extra=extra or None)
        if table.bias is None or table.kbt is None:
            raise EstError(f"{run.name}: no bias or temperature; cannot weight the frames")
        labels = definition.frame_labels(table.array(cvs))
        logw = table.bias / table.kbt
        inside = labels >= NO_DATA  # in the grid, CVs defined
        if not inside.any():
            continue
        log_total = logsumexp(logw[inside])
        run_log_totals.append(log_total)
        if table.walker is not None:
            walkers = np.asarray(table.walker)
        else:  # one walker: none (-1), or the walker of a run split per walker
            only = run.walker_ids[0]
            walkers = np.full(len(table), -1 if only is None else only)
        steps = np.rint(table.time / meta.timestep).astype(np.int64)
        available = np.zeros(len(table), dtype=bool)
        for walker in run.walker_ids:
            in_dump = scan_steps(walker_file(run.path, dump, walker))
            rows = walkers == (-1 if walker is None else walker)
            available[rows] = np.isin(steps[rows], in_dump)
        if not available.any():
            log.warning("%s: no frame of the CV files is in the dump %s", run.name, dump)
        keep = inside & available
        pool.append({
            "run": run.name, "labels": labels[keep], "logw": logw[keep] - log_total,
            "walker": walkers[keep], "step": steps[keep], "time": table.time[keep],
            "values": {c: np.asarray(table[c], dtype=float)[keep] for c in cvs + quantities},
        })  # fmt: skip
    if not pool:
        raise EstError("no frames inside the grid limits")
    if combine == "probability":
        totals = np.array(run_log_totals)
        run_share = totals - logsumexp(totals)
    else:
        run_share = np.full(len(pool), -math.log(len(pool)))

    rng = np.random.default_rng(seed)
    frames: list[SampledFrame] = []
    candidates, effective = {}, {}
    for name in chosen:
        k = definition.names.index(name)
        rows = [(i, np.nonzero(p["labels"] == k)[0]) for i, p in enumerate(pool)]
        log_p = np.concatenate([
            (pool[i]["logw"][r] + run_share[i]) if mode == "boltzmann"
            else np.full(len(r), run_share[i] - math.log(max(len(r), 1)))
            for i, r in rows
        ]) if rows else np.empty(0)  # fmt: skip
        owner = np.concatenate([np.full(len(r), i) for i, r in rows])
        row = np.concatenate([r for _, r in rows])
        candidates[name] = len(row)
        if not len(row):
            log.warning("basin %s: no frame in the dumps to draw", name)
            continue
        p = np.exp(log_p - log_p.max())
        p /= p.sum()
        effective[name] = float(1.0 / np.sum(p**2))
        count = min(n, int(np.count_nonzero(p > 0)))
        if count < n:
            log.warning("basin %s: only %d frames to draw from (asked for %d)", name, count, n)
        if mode == "boltzmann" and count > effective[name] / 2:
            log.warning(
                "basin %s: %d structures from %.0f effective frames: the last ones are drawn "
                "from the rare, high-F frames",
                name, count, effective[name],
            )  # fmt: skip
        picks = rng.choice(len(row), size=count, replace=False, p=p)
        picks = picks[np.argsort(-p[picks], kind="stable")]  # the most probable first
        for j in picks:
            src, i = pool[owner[j]], row[j]
            point = np.array([[src["values"][c][i] for c in cvs]])
            walker = int(src["walker"][i])
            frames.append(SampledFrame(
                basin=name, run=src["run"], walker=None if walker < 0 else walker,
                step=int(src["step"][i]), time_ps=float(src["time"][i]),
                values={c: float(v[i]) for c, v in src["values"].items()},
                free_energy=float(evaluate(surface, point)[0]),
                log_weight=float(src["logw"][i]),
            ))  # fmt: skip
    return BasinSample(
        cvs=tuple(cvs), quantities=quantities, frames=frames, units=surface.units,
        candidates=candidates, effective=effective,
        metadata={"mode": mode, "combined": combine, "seed": seed, "basin_mode": definition.mode,
                  "time_window_ns": list(time) if time else None, "kbt": definition.kbt},
    )  # fmt: skip


# --- extended XYZ -----------------------------------------------------------------------------


def key_name(name: str) -> str:
    """A header key extxyz readers accept: ``dOOliq.min`` -> ``dOOliq_min``."""
    key = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return key if re.match(r"[A-Za-z_]", key) else f"_{key}"


_TYPE_MAP_HINT = (
    "set atoms.type_map in estk.yaml (e.g. type_map: {1: H, 2: O, 3: Ti}) or pass --type-map "
    "1=H,2=O,3=Ti; the order of the elements in the LAMMPS pair_coeff line (pair_coeff * * "
    "model H O Ti) gives the types 1, 2, 3"
)


def _species_columns(source: Path) -> list[str]:
    """The dump's ``type`` and ``element`` columns (whichever it has)."""
    names = [c for c in ("type", "element") if c in dump_columns(source)]
    if not names:
        raise EstError(f"{source}: the dump has neither a 'type' nor an 'element' column")
    return names


def _species(frame: Frame, type_map: Mapping[int, str], source: Path) -> list[str]:
    """Element of each atom: from ``type_map`` (it wins when set), else the dump's ``element``
    column, unless that is LAMMPS's default: the same name for atoms of different types (C for
    every type without ``dump_modify element``)."""
    types = frame.columns["type"].astype(int) if "type" in frame.columns else None
    if type_map and types is not None:
        missing = sorted(set(types.tolist()) - set(type_map))
        if missing:
            raise EstError(f"{source}: atoms.type_map has no element for type(s) {missing}")
        return [type_map[t] for t in types]
    if "element" in frame.columns:
        elements = [str(e) for e in frame.columns["element"]]
        if types is not None and len(set(elements)) == 1 and len(set(types.tolist())) > 1:
            raise EstError(
                f"{source}: the element column says {elements[0]} for every atom although they "
                f"have {len(set(types.tolist()))} types: LAMMPS writes that without dump_modify "
                f"element. To get the elements, {_TYPE_MAP_HINT}"
            )
        return elements
    raise EstError(f"{source}: the dump has atom types but no elements; {_TYPE_MAP_HINT}")


def _format(value: Any) -> str:
    if isinstance(value, str):
        return f'"{value}"' if re.search(r"\s|=|\"", value) or not value else value
    if isinstance(value, (bool, np.bool_)):
        return "T" if value else "F"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.10g}"


def _extxyz(frame: Frame, species: list[str], header: Mapping[str, Any]) -> str:
    lattice = " ".join(f"{v:.10g}" for v in frame.box.matrix.reshape(-1))
    pbc = " ".join("T" if p else "F" for p in frame.box.periodic)
    fields = [f'Lattice="{lattice}"', "Properties=species:S:1:pos:R:3", f'pbc="{pbc}"']
    fields += [f"{k}={_format(v)}" for k, v in header.items()]
    positions = frame.positions - frame.box.lo
    lines = [str(len(species)), " ".join(fields)]
    lines += [f"{s:<2} {x:16.8f} {y:16.8f} {z:16.8f}"
              for s, (x, y, z) in zip(species, positions, strict=True)]  # fmt: skip
    return "\n".join(lines) + "\n"
