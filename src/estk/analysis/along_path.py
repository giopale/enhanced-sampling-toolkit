"""Averages of other quantities along a minimum energy path.

At each image of the path, a quantity (another CV, a model deviation, ...) is averaged over
the frames of each run with the frame weights ``exp(bias / kT)`` times a kernel centred on the
image, as in the reweighting (section 6): what the quantity is, on average, at that point of
the free-energy surface. The runs are combined like surfaces (``probability`` or ``equal``),
and the spread between them is kept. ``effective_frames`` says how many frames each average
really rests on: few near the saddle point are common.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from estk.errors import EstError
from estk.io.textfile import PREFIX, decode, metadata_lines, open_text, read_header
from estk.logging import get_logger
from estk.reweight.core import _split_blocks, choose_bandwidths, log_weights, logsumexp
from estk.reweight.kde import kde_at_points

if TYPE_CHECKING:
    from estk.paths import MinimumEnergyPath
    from estk.project import Project

log = get_logger(__name__)

COMBINE_MODES = ("probability", "equal")


@dataclass
class AlongPathResult:
    """Quantities averaged at the images of a path, per run and combined."""

    names: tuple[str, ...]  # CVs of the path
    points: np.ndarray  # (images, ndim)
    s: np.ndarray  # path coordinate, 0 to 1
    energies: np.ndarray  # free energy along the path (path units)
    units: str
    values: dict[str, np.ndarray]  # quantity -> combined average at each image
    spread: dict[str, np.ndarray]  # quantity -> spread between runs (NaN with one run)
    runs: dict[str, dict[str, np.ndarray]]  # run -> quantity -> average at each image
    effective_frames: np.ndarray  # all runs together, at each image
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def quantities(self) -> list[str]:
        return list(self.values)

    @property
    def saddle_index(self) -> int:
        return int(np.nanargmax(self.energies))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        fields = ["s", *self.names, "fes", "n_eff"]
        columns = [self.s, *self.points.T, self.energies, self.effective_frames]
        for q in self.quantities:
            fields += [q, f"{q}.err"]
            columns += [self.values[q], self.spread[q]]
            for run, values in self.runs.items():
                fields.append(f"{q}.{run}")
                columns.append(values[q])
        meta = {"units": self.units, "path_cvs": list(self.names), "quantities": self.quantities,
                "runs": list(self.runs), **self.metadata}  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        with open_text(path, "w") as f:
            f.write("\n".join([f"#! FIELDS {' '.join(fields)}", *metadata_lines(meta)]) + "\n")
            np.savetxt(f, np.column_stack(columns), fmt="%.10g")
        return path

    @classmethod
    def load(cls, path: Path | str) -> AlongPathResult:
        path = Path(path)
        fields, sets = read_header(path)
        meta = {k[len(PREFIX) :]: decode(v) for k, v in sets.items() if k.startswith(PREFIX)}
        if "quantities" not in meta or "path_cvs" not in meta:
            raise EstError(f"{path}: not a file written by estk along-path")
        data = np.loadtxt(path, comments="#", ndmin=2)
        column = {name: data[:, j] for j, name in enumerate(fields)}
        names = tuple(meta.pop("path_cvs"))
        quantities = meta.pop("quantities")
        runs = meta.pop("runs")
        return cls(
            names=names,
            points=np.column_stack([column[n] for n in names]),
            s=column["s"],
            energies=column["fes"],
            units=meta.pop("units"),
            values={q: column[q] for q in quantities},
            spread={q: column[f"{q}.err"] for q in quantities},
            runs={r: {q: column[f"{q}.{r}"] for q in quantities} for r in runs},
            effective_frames=column["n_eff"],
            metadata=meta,
        )


def averages_along_path(
    project: Project,
    path: MinimumEnergyPath,
    quantities: Sequence[str],
    *,
    sources: Mapping[str, str | Path] | None = None,
    time: tuple[float | None, float | None] | None = None,
    bandwidth: Mapping[str, float] | None = None,
    bandwidth_method: str | None = None,
    include_walls: bool = False,
    combine: str = "probability",
    blocks: int = 1,
) -> AlongPathResult:
    """Average ``quantities`` at each image of ``path`` (a path computed by ``estk mep``).

    ``quantities`` are loaded like CVs (COLVAR, ``CV-<name>``, ``cv.<name>.source``), or from
    ``sources`` (``{quantity: file}``, e.g. ``{"max_devi_f": "model_devi.out"}``). The kernel
    is over the path's CVs, with the bandwidths of the reweighting (``bandwidth`` >
    ``bandwidth_method`` > estk.yaml > Silverman). ``combine`` and ``blocks`` work as in
    ``estk.reweight.reweight``.
    """
    if combine not in COMBINE_MODES:
        raise EstError(f"combine must be one of {', '.join(COMBINE_MODES)}, got '{combine}'")
    quantities = list(dict.fromkeys(quantities))
    if not quantities:
        raise EstError("no quantities to average")
    cvs = list(path.names)
    overlap = [q for q in quantities if q in cvs]
    if overlap:
        raise EstError(f"{', '.join(overlap)} is a CV of the path: its value is the path itself")
    extra = {q: f for q, f in (sources or {}).items() if q in quantities}
    loaded = [q for q in quantities if q not in extra]

    tables = {
        run.name: run.load_cvs(cvs + loaded, time=time, extra=extra or None,
                               include_walls=include_walls)
        for run in project.runs
    }  # fmt: skip
    if blocks > 1:
        tables, _ = _split_blocks(tables, blocks)
    configured = {n: project.config.cv[n].bandwidth for n in cvs if n in project.config.cv}

    per_run: dict[str, dict[str, np.ndarray]] = {}
    log_density, log_total, n_eff_terms, used_bandwidths = {}, {}, {}, {}
    for name, table in tables.items():
        if table.bias is None or table.kbt is None:
            raise EstError(f"{name}: no bias or temperature; cannot weight the frames")
        raw = log_weights(table.bias, table.kbt)
        log_total[name] = logsumexp(raw)
        weights = np.exp(raw - log_total[name])
        h, _ = choose_bandwidths(table, cvs, weights, explicit=bandwidth, configured=configured,
                                 method=bandwidth_method)  # fmt: skip
        used_bandwidths[name] = {n: float(v) for n, v in zip(cvs, h, strict=True)}
        estimate = kde_at_points(
            path.points, table.array(cvs), raw, h, quantities={q: table[q] for q in quantities}
        )
        per_run[name] = estimate.averages
        log_density[name] = estimate.log_density
        n_eff_terms[name] = estimate.effective_frames

    # each run's share of the density at each image, as in the combination of surfaces
    names = list(tables)
    run_log_weight = np.array([log_total[n] for n in names]) if combine == "probability" \
        else np.zeros(len(names))  # fmt: skip
    run_log_weight -= logsumexp(run_log_weight)
    terms = np.array([run_log_weight[k] + log_density[n] for k, n in enumerate(names)])
    with np.errstate(invalid="ignore"):
        local = np.exp(terms - np.max(terms, axis=0))
        local = np.where(np.isfinite(local), local, 0.0)
        local /= np.where(local.sum(axis=0) > 0, local.sum(axis=0), np.nan)

    values, spread = {}, {}
    for q in quantities:
        stack = np.array([per_run[n][q] for n in names])  # (runs, images)
        share = np.where(np.isfinite(stack), local, 0.0)
        norm = share.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (share * np.nan_to_num(stack)).sum(axis=0) / norm
            var = (share * np.nan_to_num(stack - mean) ** 2).sum(axis=0) / norm
        values[q] = mean
        spread[q] = np.sqrt(var) if len(names) > 1 else np.full_like(mean, np.nan)
    # frames effectively used at each image, all runs together (sum over independent runs)
    effective = np.sum([n_eff_terms[n] for n in names], axis=0)

    first = tables[names[0]]
    meta = {
        "combined": combine,
        "kbt": first.kbt,
        "bandwidth": used_bandwidths,
        "time_window_ns": list(time) if time else None,
        "walls_in_bias": include_walls,
        "blocks": blocks,
        **{k: path.metadata[k] for k in ("temperature", "ends") if k in path.metadata},
    }
    thin = int(np.count_nonzero(effective < 10))
    if thin:
        log.warning("%d of %d images have fewer than 10 effective frames: their averages are "
                    "uncertain", thin, len(effective))  # fmt: skip
    return AlongPathResult(
        names=tuple(cvs), points=np.asarray(path.points), s=path.s, energies=path.energies,
        units=path.units, values=values, spread=spread, runs=per_run,
        effective_frames=effective, metadata=meta,
    )  # fmt: skip
