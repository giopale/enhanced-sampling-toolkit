"""Populations of the free-energy basins, and averages of quantities within each basin.

Basins are either found on a free-energy surface (``basin_map``) or named regions of CV space
(``basins:`` in estk.yaml). Automatic basins are a watershed of the surface: every grid point
belongs to the minimum it descends to, and minima shallower than ``min_depth`` are merged into
the neighbouring basin they spill into (the same persistence ``estk mep`` uses to choose its
ends).

Populations are counted on the frames, without kernels: each frame of each run is assigned to
the basin of its grid cell (or the named region it is in) and

    P_A = sum_{i in A} w_i / sum_i w_i,   w_i = exp(V_i / kT),

over the frames inside the grid limits (the walls, as for the surfaces): frames past them are
reported, not counted. The runs are combined like surfaces (``equal``: same weight per run;
``probability``: by their total weight). The error is the standard error between runs, or
between time blocks with one run. The populations integrated from the surface,
``sum_{cells in A} exp(-F/kT)``, are given as a check.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.errors import EstError, MissingFileError
from estk.grid import Grid
from estk.io.textfile import PREFIX, decode, metadata_lines, open_text, read_header
from estk.logging import get_logger
from estk.reweight.combine import MODES as COMBINE_MODES
from estk.reweight.core import logsumexp
from estk.surface import Surface

if TYPE_CHECKING:
    from estk.project import Project

log = get_logger(__name__)

NO_DATA = -1  # label of grid cells (and frames) without data, or in no named region
OUTSIDE = -2  # label of frames outside the grid limits

Region = Mapping[str, tuple[float | None, float | None]]


# --- basins on the grid -----------------------------------------------------------------------


def basin_map(surface: Surface, min_depth: float = 0.0) -> tuple[np.ndarray, list[int]]:
    """Watershed of a free-energy surface: the basin label of every grid point.

    Grid points are added from the lowest free energy up (nearest neighbours along each axis);
    a point joins the basin of its lowest neighbour. When two basins meet, the one whose minimum
    is less than ``min_depth`` (surface units) below the meeting level is merged into the other;
    deeper basins stay apart. Labels are 0, 1, ... with basin 0 the lowest; points without data
    are ``NO_DATA``. Returns the labels and the flat grid index of each basin's minimum.
    """
    values = np.asarray(surface.values, dtype=float)
    shape = values.shape
    flat = values.reshape(-1)
    order = [int(i) for i in np.argsort(flat, kind="stable") if np.isfinite(flat[i])]
    parent = np.full(flat.size, -1)
    strides = [int(np.prod(shape[d + 1 :])) for d in range(len(shape))]
    lowest: dict[int, int] = {}  # root -> flat index of its minimum

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for cell in order:
        coords = np.unravel_index(cell, shape)
        neighbours = []
        for d in range(len(shape)):
            for step in (-1, 1):
                c = coords[d] + step
                if 0 <= c < shape[d]:
                    other = cell + step * strides[d]
                    if parent[other] >= 0:
                        neighbours.append(other)
        if not neighbours:
            parent[cell] = cell
            lowest[cell] = cell
            continue
        roots = {root(n) for n in neighbours}
        level = float(flat[cell])
        # shallow basins spill into the deepest one they touch
        deepest = min(roots, key=lambda r: flat[lowest[r]])
        for r in roots:
            if r != deepest and level - flat[lowest[r]] < min_depth:
                parent[r] = deepest
                del lowest[r]
        # the point itself goes downhill: to the basin of its lowest neighbour
        parent[cell] = root(min(neighbours, key=lambda n: flat[n]))

    minima = sorted(lowest.values(), key=lambda i: flat[i])
    label_of_root = {root(m): k for k, m in enumerate(minima)}
    labels = np.full(flat.size, NO_DATA, dtype=int)
    for cell in order:
        labels[cell] = label_of_root[root(cell)]
    return labels.reshape(shape), minima


def region_map(grid: Grid, regions: Mapping[str, Region]) -> np.ndarray:
    """Label of every grid point: the first named region it is in, or ``NO_DATA``."""
    coordinates = grid.coordinates()
    labels = np.full(len(coordinates), NO_DATA, dtype=int)
    for k, region in reversed(list(enumerate(regions.values()))):
        inside = _in_region(coordinates, list(grid.names), region)
        labels[inside] = k
    return labels.reshape(grid.shape)


def _in_region(points: np.ndarray, names: list[str], region: Region) -> np.ndarray:
    inside = np.ones(len(points), dtype=bool)
    for cv, (lo, hi) in region.items():
        x = points[:, names.index(cv)]
        if lo is not None:
            inside &= x >= lo
        if hi is not None:
            inside &= x <= hi
    return inside


def _frame_labels(
    points: np.ndarray, grid: Grid, labels: np.ndarray, regions: Mapping[str, Region] | None
) -> np.ndarray:
    """Basin of each frame: by its nearest grid point, or by the named regions."""
    lower, upper = np.array(grid.lower), np.array(grid.upper)
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    out = np.full(len(points), OUTSIDE, dtype=int)
    if regions is not None:
        found = np.full(len(points), NO_DATA, dtype=int)
        for k, region in reversed(list(enumerate(regions.values()))):
            found[_in_region(points, list(grid.names), region)] = k
        out[inside] = found[inside]
        return out
    spacing = (upper - lower) / (np.array(grid.points) - 1)
    index = np.rint((points[inside] - lower) / spacing).astype(int)
    index = np.clip(index, 0, np.array(grid.points) - 1)
    out[inside] = labels[tuple(index.T)]
    return out


@dataclass
class BasinDefinition:
    """Which basin every grid point (and so every frame) belongs to."""

    names: list[str]
    labels: np.ndarray  # basin of every grid point (NO_DATA: none)
    grid: Grid
    kbt: float
    mode: str  # "automatic" or "regions"
    regions: dict[str, Region] | None = None

    def frame_labels(self, points: np.ndarray) -> np.ndarray:
        """Basin of each frame (``OUTSIDE`` past the grid limits, ``NO_DATA`` in none)."""
        return _frame_labels(points, self.grid, self.labels, self.regions)


def define_basins(
    project: Project,
    cvs: Sequence[str],
    surface: Surface,
    *,
    regions: Mapping[str, Region] | None = None,
    min_depth_kt: float = 2.0,
) -> BasinDefinition:
    """The basins of ``surface``: the named ``regions`` (default: ``basins:`` of estk.yaml, if
    set) or the automatic basins (``basin_map``, merged below ``min_depth_kt``)."""
    cvs = list(cvs)
    if not surface.is_fes:
        raise EstError(f"the surface is a {surface.quantity} surface, not a free energy")
    if list(surface.grid.names) != cvs:
        raise EstError(f"the surface is in {surface.grid.names}, the basins in {tuple(cvs)}")
    kbt = float(surface.metadata.get("kbt") or 0.0)
    if kbt <= 0:
        raise EstError("the surface has no kT in its metadata")
    if regions is None and project.config.basins:
        regions = {name: dict(r) for name, r in project.config.basins.items()}
    if regions:
        unknown = sorted({cv for r in regions.values() for cv in r} - set(cvs))
        if unknown:
            raise EstError(f"basin regions use {', '.join(unknown)}, not CVs of {tuple(cvs)}")
        return BasinDefinition(list(regions), region_map(surface.grid, regions), surface.grid,
                               kbt, "regions", dict(regions))  # fmt: skip
    labels, minima = basin_map(surface, min_depth_kt * kbt)
    names = [str(k + 1) for k in range(len(minima))]
    return BasinDefinition(names, labels, surface.grid, kbt, "automatic")


# --- result -----------------------------------------------------------------------------------


@dataclass
class Basin:
    name: str
    point: tuple[float, ...]  # the lowest grid point of the basin
    free_energy: float  # surface value there (surface units; the surface's minimum is 0)
    population: float
    error: float  # standard error between runs (or blocks); NaN without either
    surface_population: float  # integrated from the surface, as a check
    effective_frames: float  # Kish, summed over runs
    run_populations: dict[str, float] = field(default_factory=dict)
    averages: dict[str, float] = field(default_factory=dict)  # quantity -> <q> in the basin
    average_errors: dict[str, float] = field(default_factory=dict)


@dataclass
class BasinResult:
    names: tuple[str, ...]  # CVs
    basins: list[Basin]
    labels: np.ndarray  # basin of every grid point (NO_DATA: none)
    grid: Grid
    units: str
    kbt: float
    unassigned: float = 0.0  # population in no named region
    outside_fraction: float = 0.0  # raw fraction of frames outside the grid limits
    totals: dict[str, tuple[float, float]] = field(default_factory=dict)  # q -> (<q>, error)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quantities(self) -> list[str]:
        return list(self.totals)

    def free_energy_difference(self, basin: Basin) -> tuple[float, float]:
        """F_A - F_ref = -kT ln(P_A / P_ref), ref the most populated basin; and its error."""
        ref = max(self.basins, key=lambda b: b.population)
        if basin.population <= 0:
            return math.inf, math.nan
        value = -self.kbt * math.log(basin.population / ref.population)
        if basin is ref:
            return 0.0, 0.0
        relative = [(b.error / b.population) ** 2 for b in (basin, ref)]
        return value, self.kbt * math.sqrt(sum(relative))

    def label_surface(self) -> Surface:
        return Surface(self.grid, self.labels.astype(float), quantity="basin", units="",
                       metadata={"basins": [b.name for b in self.basins]})  # fmt: skip

    def save(self, path: Path | str) -> Path:
        """Write a table: one line per basin (name first), the result's settings in the header."""
        path = Path(path)
        runs = list(self.basins[0].run_populations) if self.basins else []
        fields = ["basin", *self.names, "fes", "population", "error", "surface_population",
                  "n_eff"]  # fmt: skip
        for q in self.quantities:
            fields += [q, f"{q}.err"]
        fields += [f"population.{r}" for r in runs]
        meta = {
            "kind": "basins",
            **self.metadata,
            "units": self.units,
            "kbt": self.kbt,
            "unassigned": self.unassigned,
            "outside_fraction": self.outside_fraction,
            "totals": {q: list(v) for q, v in self.totals.items()},
            "grid": {"names": list(self.grid.names), "lower": list(self.grid.lower),
                     "upper": list(self.grid.upper), "points": list(self.grid.points)},
        }  # fmt: skip
        lines = [f"#! FIELDS {' '.join(fields)}", *metadata_lines(meta)]
        for b in self.basins:
            values = [*b.point, b.free_energy, b.population, b.error, b.surface_population,
                      b.effective_frames]  # fmt: skip
            for q in self.quantities:
                values += [b.averages.get(q, math.nan), b.average_errors.get(q, math.nan)]
            values += [b.run_populations.get(r, math.nan) for r in runs]
            lines.append(" ".join([b.name.replace(" ", "_"), *(f"{v:.10g}" for v in values)]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str, labels: np.ndarray | None = None) -> BasinResult:
        """Read a table written by ``save`` (the label map, if needed, is a separate file)."""
        path = Path(path)
        if not path.is_file():
            raise MissingFileError(f"basins file not found: {path}")
        fields, sets = read_header(path)
        meta = {k[len(PREFIX) :]: decode(v) for k, v in sets.items() if k.startswith(PREFIX)}
        if meta.pop("kind", None) != "basins":
            raise EstError(f"{path}: not a basins file (estk basins)")
        g = meta.pop("grid")
        grid = Grid(tuple(g["names"]), tuple(g["lower"]), tuple(g["upper"]), tuple(g["points"]))
        names = tuple(g["names"])
        totals = {q: (float(v[0]), float(v[1])) for q, v in meta.pop("totals").items()}
        basins = []
        with open_text(path) as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                name, *raw = line.split()
                row = dict(zip(fields[1:], (float(v) for v in raw), strict=True))
                basins.append(Basin(
                    name=name,
                    point=tuple(row[n] for n in names),
                    free_energy=row["fes"],
                    population=row["population"],
                    error=row["error"],
                    surface_population=row["surface_population"],
                    effective_frames=row["n_eff"],
                    run_populations={f[11:]: row[f] for f in fields if f.startswith("population.")},
                    averages={q: row[q] for q in totals},
                    average_errors={q: row[f"{q}.err"] for q in totals},
                ))  # fmt: skip
        return cls(
            names=names, basins=basins,
            labels=labels if labels is not None else np.full(grid.shape, NO_DATA),
            grid=grid, units=meta.pop("units"), kbt=float(meta.pop("kbt")),
            unassigned=float(meta.pop("unassigned")),
            outside_fraction=float(meta.pop("outside_fraction")), totals=totals, metadata=meta,
        )  # fmt: skip


# --- populations ------------------------------------------------------------------------------


@dataclass
class _RunEstimate:
    populations: np.ndarray  # (basins + 1,): last entry = unassigned / no data
    log_total: float  # log of the total weight of the frames inside the grid
    effective: np.ndarray  # (basins,)
    averages: dict[str, np.ndarray]  # q -> (basins,)
    totals: dict[str, float]  # q -> <q> over all frames inside the grid
    outside: float  # raw fraction of frames outside the grid


def _estimate(labels: np.ndarray, logw: np.ndarray, values: Mapping[str, np.ndarray],
              n_basins: int) -> _RunEstimate:  # fmt: skip
    inside = labels != OUTSIDE
    outside = 1.0 - float(inside.mean()) if len(labels) else math.nan
    lab, lw = labels[inside], logw[inside]
    if not len(lab):
        nan = np.full(n_basins, math.nan)
        return _RunEstimate(
            populations=np.full(n_basins + 1, math.nan),
            log_total=-math.inf,
            effective=np.zeros(n_basins),
            averages={q: nan for q in values},
            totals={q: math.nan for q in values},
            outside=outside,
        )
    peak = float(lw.max())
    w = np.exp(lw - peak)
    total = float(w.sum())
    populations = np.zeros(n_basins + 1)
    effective = np.zeros(n_basins)
    averages = {q: np.full(n_basins, math.nan) for q in values}
    for k in range(n_basins):
        mask = lab == k
        wk = w[mask]
        populations[k] = wk.sum() / total
        if wk.sum() > 0:
            effective[k] = wk.sum() ** 2 / np.sum(wk**2)
            for q, v in values.items():
                averages[q][k] = float(np.sum(wk * v[inside][mask]) / wk.sum())
    populations[n_basins] = w[lab == NO_DATA].sum() / total
    totals = {q: float(np.sum(w * v[inside]) / total) for q, v in values.items()}
    return _RunEstimate(populations, peak + math.log(total), effective, averages, totals,
                        outside)  # fmt: skip


def _standard_error(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def basin_populations(
    project: Project,
    cvs: Sequence[str],
    *,
    surface: Surface | None = None,
    regions: Mapping[str, Region] | None = None,
    min_depth_kt: float = 2.0,
    quantities: Sequence[str] = (),
    sources: Mapping[str, str | Path] | None = None,
    time: tuple[float | None, float | None] | None = None,
    blocks: int = 1,
    combine: str = "equal",
    reweight_options: Mapping[str, Any] | None = None,
) -> BasinResult:
    """Populations of the basins of the free energy in ``cvs``, from the frames of every run.

    Parameters
    ----------
    surface:
        Free-energy surface in ``cvs`` that defines the grid (limits) and the automatic basins;
        default: reweight the project now (``reweight_options``: points, limits, bandwidth...).
    regions:
        Named regions ``{name: {cv: (lo, hi)}}`` (None for an open end; CVs left out are not
        restricted) instead of automatic basins. A frame in several belongs to the first.
        Default: ``basins:`` of estk.yaml if set, otherwise automatic basins.
    min_depth_kt:
        Automatic basins shallower than this (kT) are merged into their neighbours.
    quantities, sources:
        Quantities to average in each basin (CVs, ``CV-<name>`` files, or ``{q: file}``).
    time:
        Time window in ns (the surface, when it is computed here, uses the same).
    blocks:
        Time blocks per run for the error bar when there is only one run.
    combine:
        ``equal`` (default) or ``probability``, as for surfaces.
    """
    cvs = list(cvs)
    if combine not in COMBINE_MODES or combine == "legacy":
        raise EstError(f"combine must be equal or probability, got '{combine}'")
    if blocks < 1:
        raise EstError(f"blocks must be at least 1, got {blocks}")
    quantities = list(dict.fromkeys(quantities))
    clash = [q for q in quantities if q in cvs]
    if clash:
        raise EstError(f"{', '.join(clash)} is a CV of the basins: its average is the basin")
    if surface is None:
        options = dict(reweight_options or {})
        surface = project.reweight(cvs, time=time, combine=combine, **options).combined
    definition = define_basins(project, cvs, surface, regions=regions, min_depth_kt=min_depth_kt)
    kbt, grid, labels = definition.kbt, surface.grid, definition.labels
    names, mode, regions = definition.names, definition.mode, definition.regions
    n = len(names)

    extra = {q: f for q, f in (sources or {}).items() if q in quantities}
    loaded = [q for q in quantities if q not in extra]
    estimates: dict[str, _RunEstimate] = {}
    block_estimates: dict[str, list[_RunEstimate]] = {}
    for run in project.runs:
        table = run.load_cvs(cvs + loaded, time=time, extra=extra or None)
        if table.bias is None or table.kbt is None:
            raise EstError(f"{run.name}: no bias or temperature; cannot weight the frames")
        frame_labels = _frame_labels(table.array(cvs), grid, labels, regions)
        logw = table.bias / table.kbt
        values = {q: np.asarray(table[q], dtype=float) for q in quantities}
        estimates[run.name] = _estimate(frame_labels, logw, values, n)
        if blocks > 1:
            order = np.argsort(table.time, kind="stable")
            block_estimates[run.name] = [
                _estimate(frame_labels[i], logw[i], {q: v[i] for q, v in values.items()}, n)
                for i in np.array_split(order, blocks)
            ]  # fmt: skip
    if not estimates:
        raise EstError("no runs found")
    runs = list(estimates)

    # run weights of the combination
    if combine == "probability":
        log_totals = np.array([estimates[r].log_total for r in runs])
        weights = np.exp(log_totals - logsumexp(log_totals))
    else:
        weights = np.full(len(runs), 1.0 / len(runs))
    table_p = np.array([estimates[r].populations for r in runs])  # (runs, basins + 1)
    populations = weights @ table_p

    def spread(per_run: np.ndarray, per_block: list[np.ndarray] | None) -> float:
        if len(runs) > 1:
            return _standard_error(per_run)
        return _standard_error(np.array(per_block)) if per_block else math.nan

    blocks_of_first = block_estimates.get(runs[0])
    values_f = surface.values
    finite = np.isfinite(values_f)
    density = np.where(finite, np.exp(-(values_f - np.min(values_f[finite])) / kbt), 0.0)
    basins = []
    for k, name in enumerate(names):
        cells = labels == k
        if cells.any():
            lowest = np.unravel_index(np.argmin(np.where(cells, values_f, np.inf)), labels.shape)
            point = tuple(float(a[i]) for a, i in zip(grid.axes(), lowest, strict=True))
            fes = float(values_f[lowest])
        else:
            point, fes = tuple(math.nan for _ in cvs), math.nan
        averages, errors = {}, {}
        for q in quantities:
            per_run = np.array([estimates[r].averages[q][k] for r in runs])
            share = weights * table_p[:, k]
            ok = np.isfinite(per_run) & (share > 0)
            averages[q] = float(np.sum(share[ok] * per_run[ok]) / share[ok].sum()) \
                if ok.any() else math.nan  # fmt: skip
            errors[q] = spread(per_run, [b.averages[q][k] for b in blocks_of_first]
                               if blocks_of_first else None)  # fmt: skip
        basins.append(Basin(
            name=name,
            point=point,
            free_energy=fes,
            population=float(populations[k]),
            error=spread(table_p[:, k],
                         [b.populations[k] for b in blocks_of_first] if blocks_of_first else None),
            surface_population=float(density[cells].sum() / density.sum()),
            effective_frames=float(sum(estimates[r].effective[k] for r in runs)),
            run_populations={r: float(table_p[i, k]) for i, r in enumerate(runs)},
            averages=averages,
            average_errors=errors,
        ))  # fmt: skip
    totals = {}
    for q in quantities:
        per_run = np.array([estimates[r].totals[q] for r in runs])
        totals[q] = (float(weights @ per_run),
                     spread(per_run, [b.totals[q] for b in blocks_of_first]
                            if blocks_of_first else None))  # fmt: skip
    outside = float(np.mean([estimates[r].outside for r in runs]))
    for b in basins:
        if b.effective_frames < 10 and b.population > 0:
            log.warning("basin %s rests on %.1f effective frames: its population is uncertain",
                        b.name, b.effective_frames)  # fmt: skip
    metadata = {
        "mode": mode,
        "combined": combine,
        "runs": runs,
        "run_weights": {r: float(w) for r, w in zip(runs, weights, strict=True)},
        "time_window_ns": list(time) if time else None,
        "blocks": blocks,
        "min_depth_kt": min_depth_kt if mode == "automatic" else None,
        "regions": {k: {cv: list(v) for cv, v in r.items()} for k, r in regions.items()}
        if regions else None,
        "temperature": surface.metadata.get("temperature"),
    }  # fmt: skip
    return BasinResult(
        names=tuple(cvs), basins=basins, labels=labels, grid=grid, units=surface.units, kbt=kbt,
        unassigned=float(populations[n]), outside_fraction=outside, totals=totals,
        metadata=metadata,
    )  # fmt: skip
