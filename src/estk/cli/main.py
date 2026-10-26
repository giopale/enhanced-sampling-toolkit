"""Entry point of the ``estk`` command."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from estk import __version__
from estk.config import (
    CONFIG_FILENAME,
    config_template,
    find_config,
    is_bias_column,
    load_config,
    read_colvar_fields,
)
from estk.errors import EstError
from estk.logging import get_logger, setup_cli_logging

app = typer.Typer(
    help="Analysis tools for MD + enhanced sampling simulations.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
config_app = typer.Typer(help="Inspect, check and migrate estk.yaml.", no_args_is_help=True)
app.add_typer(config_app, name="config")
plot_app = typer.Typer(help="Figures of surfaces and paths.", no_args_is_help=True)
app.add_typer(plot_app, name="plot")

console = Console()
err_console = Console(stderr=True)
log = get_logger(__name__)

# Options given before the subcommand (estk -v --config x.yaml ...).
_state: dict[str, Any] = {"verbosity": 0, "config": None}

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    """Print estk errors as a short message and exit with code 1."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except EstError as exc:
            if _state["verbosity"] > 0:
                err_console.print_exception()
            err_console.print(f"[bold red]error:[/bold red] {exc}", highlight=False)
            raise typer.Exit(code=1) from None

    return wrapper  # type: ignore[return-value]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"estk {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug messages.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Show warnings only.")] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help=f"Project file (default: nearest {CONFIG_FILENAME})."),
    ] = None,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    _state["verbosity"] = 1 if verbose else -1 if quiet else 0
    _state["config"] = config
    setup_cli_logging(_state["verbosity"])


@app.command()
@handle_errors
def init(
    root: Annotated[str, typer.Option(help="Folder containing the run folders.")] = ".",
    pattern: Annotated[str, typer.Option(help="Glob pattern matching the run folders.")] = "*",
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a commented estk.yaml in the current folder.

    If a run folder with a COLVAR file is found, a section is added for each of its CVs.
    """
    target = Path.cwd() / CONFIG_FILENAME
    if target.exists() and not force:
        raise EstError(f"{target} already exists (use --force to overwrite)")

    cvs = _discover_cvs(Path(root), pattern)
    target.write_text(config_template(root=root, pattern=pattern, cvs=cvs))
    if cvs:
        log.info("found CVs: %s", ", ".join(cvs))
    else:
        log.info("no COLVAR found in %s/%s; added an example CV section", root, pattern)
    console.print(f"wrote {target}")


def _discover_cvs(root: Path, pattern: str) -> list[str]:
    colvar = load_config(_state["config"]).files.colvar if _state["config"] else "COLVAR"
    for folder in sorted(p for p in root.glob(pattern) if p.is_dir()):
        path = folder / colvar
        if path.is_file():
            log.debug("reading CV names from %s", path)
            return [name for name in read_colvar_fields(path) if not is_bias_column(name)]
    return []


@app.command()
@handle_errors
def runs() -> None:
    """List the runs and the simulation settings read for each one."""
    from estk.project import Project

    project = Project(load_config(_state["config"]))
    table = Table("run", "T (K)", "timestep (ps)", "bias", "bias factor", "walls", "units")
    for run in project.runs:
        meta = run.metadata
        walls = ", ".join(f"{cv} [{_fmt(lo)}, {_fmt(hi)}]" for cv, (lo, hi) in meta.walls.items())
        table.add_row(
            run.name,
            _fmt(meta.temperature),
            _fmt(meta.timestep),
            meta.bias_column or "-",
            _fmt(meta.bias_factor),
            walls or "-",
            f"{meta.time_units}, {meta.energy_units}",
        )
    console.print(table)
    overridden = sorted(
        {k for r in project.runs for k, v in r.metadata.sources.items() if v == CONFIG_FILENAME}
    )
    if overridden:
        console.print(f"set in {CONFIG_FILENAME}: {', '.join(overridden)}")


@app.command()
@handle_errors
def cvs(
    names: Annotated[list[str], typer.Argument(help="CVs or other COLVAR columns to load.")],
    time: Annotated[
        tuple[float, float] | None,
        typer.Option(help="Time window in ns, e.g. --time 0 20 (use inf for no end)."),
    ] = None,
) -> None:
    """Load CVs from every run and show how many rows were kept and the range of each CV."""
    from estk.project import Project

    project = Project(load_config(_state["config"]))
    table = Table("run", "rows", "time (ns)", *names, "past walls", "notes")
    for run in project.runs:
        data = run.load_cvs(names, time=time, bias=False)
        ranges = [f"{data[n].min():.4g} – {data[n].max():.4g}" for n in names]
        span = f"{data.time[0] / 1000:.3g} – {data.time[-1] / 1000:.3g}"
        notes = data.report.summary().split(", ", 1)
        walls = data.report.wall_summary() or "-"
        table.add_row(
            run.name,
            str(len(data)),
            span,
            *ranges,
            walls,
            notes[1] if len(notes) > 1 else "",
        )
    console.print(table)


@app.command()
@handle_errors
def reweight(
    names: Annotated[list[str], typer.Argument(help="CVs of the grid: one (d0) or more (d0 cn).")],
    prop: Annotated[
        str | None,
        typer.Option("--property", help="Average this quantity instead of the free energy."),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(help="File holding --property when it is not a CV (e.g. model_devi.out)."),
    ] = None,
    time: Annotated[
        tuple[float, float] | None,
        typer.Option(help="Time window in ns, e.g. --time 0 20 (use inf for no end)."),
    ] = None,
    points: Annotated[
        str, typer.Option(help="Grid points: one number for all CVs, or one per CV (100,80).")
    ] = "100",
    limits: Annotated[
        list[str] | None,
        typer.Option(help="Grid limits of a CV, e.g. --limits d0=1.5,2.5 or cn=0.75, (repeat)."),
    ] = None,
    bandwidth: Annotated[
        list[str] | None,
        typer.Option(help="Kernel bandwidth of a CV, e.g. --bandwidth d0=0.02 (repeat)."),
    ] = None,
    include_walls: Annotated[
        bool, typer.Option("--include-walls", help="Add the PLUMED wall energies to the bias.")
    ] = False,
    jobs: Annotated[int, typer.Option("--jobs", "-j", help="Runs computed in parallel.")] = 1,
    skip_failed: Annotated[
        bool, typer.Option("--skip-failed", help="Leave out runs that fail instead of stopping.")
    ] = False,
    combine: Annotated[
        str,
        typer.Option(
            help="Combine the runs: probability (default; = all frames together), equal "
            "(every run counts the same), legacy (as the original scripts), none."
        ),
    ] = "probability",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o", help="Output folder (default: reweight-<cvs>[-<property>])."
        ),
    ] = None,
) -> None:
    """Reweight every run to a free-energy surface (or property average) on a common grid.

    Writes one file per run, <output>/<run>.dat, and the combination of all runs,
    <output>/combined.dat, in PLUMED grid format.
    """
    from estk.project import Project

    mode = _combine_mode(combine)
    project = Project(load_config(_state["config"]))
    result = project.reweight(
        names,
        quantity=prop,
        source=source,
        time=time,
        points=_parse_points(points),
        limits=_parse_assignments(limits, "limits", pair=True),
        bandwidth=_parse_assignments(bandwidth, "bandwidth", pair=False),
        include_walls=include_walls,
        jobs=jobs,
        skip_failed=skip_failed,
        combine=mode,
    )
    folder = output or Path("reweight-" + "-".join([*names, *([prop] if prop else [])]))
    paths = result.save(folder)

    grid = result.grid
    for name, lo, hi, n in zip(grid.names, grid.lower, grid.upper, grid.points, strict=True):
        src_lo, src_hi = result.limit_sources[name]
        console.print(f"grid {name}: {lo:g} ({src_lo}) to {hi:g} ({src_hi}), {n} points")
    weights = result.combined.metadata["run_weights"] if result.combined is not None else {}
    table = Table("run", "frames", "effective", *[f"h({n})" for n in names], "weight", "past walls")
    for run_name, surface in result.surfaces.items():
        meta = surface.metadata
        report = result.tables[run_name].report
        table.add_row(
            run_name,
            str(meta["frames"]),
            f"{meta['effective_frames']:.0f}",
            *[f"{meta['bandwidth'][n]:.4g} ({meta['bandwidth_source'][n]})" for n in names],
            f"{weights[run_name]:.3f}" if run_name in weights else "-",
            report.wall_summary() or "-",
        )
    console.print(table)
    if mode:
        console.print(f"runs combined with mode '{mode}' -> combined.dat")
    if include_walls:
        console.print("wall energies are included in the bias")
    for run_name, reason in result.failed.items():
        err_console.print(f"[yellow]skipped {run_name}:[/yellow] {reason}", highlight=False)
    console.print(f"wrote {len(paths)} file(s) to {folder}/")


@app.command("combine")
@handle_errors
def combine_files(
    files: Annotated[list[Path], typer.Argument(help="Per-run surfaces written by estk reweight.")],
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="probability (default), equal or legacy.")
    ] = "probability",
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file.")] = Path(
        "combined.dat"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite the output file.")] = False,
) -> None:
    """Combine per-run surfaces (e.g. reweight-d0-cn/0*.dat) into one, with another mode."""
    from estk.reweight import combine
    from estk.surface import Surface

    chosen = _combine_mode(mode)
    if chosen is None:
        raise EstError("--mode none makes no sense here")
    if output.exists() and not force:
        raise EstError(f"{output} already exists (use --force to overwrite)")
    surfaces = {}
    for path in files:
        if path.name == "combined.dat":
            continue
        surface = Surface.load(path)
        surfaces[str(surface.metadata.get("run", path.stem))] = surface
    result = combine(surfaces, mode=chosen)
    result.save(output)
    table = Table("run", "weight")
    for run_name, weight in result.metadata["run_weights"].items():
        table.add_row(run_name, f"{weight:.3f}")
    console.print(table)
    console.print(f"wrote {output} (mode '{chosen}')")


@app.command()
@handle_errors
def mep(
    file: Annotated[
        Path, typer.Argument(help="Free-energy surface, e.g. reweight-d0-cn/combined.dat")
    ],
    start: Annotated[str | None, typer.Option(help="Start point, e.g. 2.09,1.98.")] = None,
    end: Annotated[str | None, typer.Option(help="End point, e.g. 1.86,1.0.")] = None,
    via: Annotated[
        list[str] | None, typer.Option(help="Point the initial path goes through (repeat).")
    ] = None,
    method: Annotated[
        str, typer.Option(help="How to find the ends: basins (default) or split (old scripts).")
    ] = "basins",
    minima: Annotated[int, typer.Option(help="Minima to use (3: the middle one is a via).")] = 2,
    split_axis: Annotated[
        str | None, typer.Option(help="CV to split along with --method split (default: first).")
    ] = None,
    min_depth: Annotated[
        float, typer.Option(help="Minimum depth of a basin, in kT (--method basins).")
    ] = 2.0,
    reverse: Annotated[bool, typer.Option("--reverse", help="Swap start and end.")] = False,
    images: Annotated[int, typer.Option(help="Images (points) along the path.")] = 15,
    dt: Annotated[float, typer.Option(help="String step, in CV units.")] = 0.01,
    max_steps: Annotated[int, typer.Option(help="Maximum string steps.")] = 2000,
    free_ends: Annotated[
        bool, typer.Option("--free-ends", help="Let the ends move (default: fixed on minima).")
    ] = False,
    cap: Annotated[
        float | None,
        typer.Option(help="Cap free energies above this (surface units) for the relaxation."),
    ] = None,
    runs: Annotated[
        bool,
        typer.Option(
            "--runs/--no-runs",
            help="Also evaluate the path on the per-run files next to a combined surface.",
        ),
    ] = True,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file (default: <file>.mep.dat).")
    ] = None,
) -> None:
    """Minimum energy path between two minima of a free-energy surface, and its barriers."""
    from estk.paths import find_minima, minimum_energy_path
    from estk.surface import Surface
    from estk.units import energy_factor

    surface = Surface.load(file)
    names = surface.grid.names
    profiles = {}
    if runs:
        for run_name in surface.metadata.get("runs", []):
            sibling = file.parent / f"{run_name}.dat"
            if sibling.is_file():
                profiles[run_name] = Surface.load(sibling)
    path = minimum_energy_path(
        surface,
        start=_parse_point(start, "start", len(names)),
        end=_parse_point(end, "end", len(names)),
        via=[_parse_point(v, "via", len(names)) for v in via or []],
        method=method,
        minima=minima,
        split_axis=split_axis if split_axis is not None else 0,
        min_depth_kt=min_depth,
        reverse=reverse,
        images=images,
        dt=dt,
        max_steps=max_steps,
        fixed_ends=not free_ends,
        cap=cap,
        profiles=profiles,
    )
    target = output or file.with_name(file.stem + ".mep.dat")
    path.save(target)

    kbt = surface.metadata.get("kbt")
    to_ev = energy_factor(surface.units) / energy_factor("ev")

    def energy(value: float) -> str:
        text = f"{value:.3f} {surface.units} = {value * to_ev:.4f} eV"
        return text + (f" = {value / kbt:.2f} kT" if kbt else "")

    if method == "basins" and start is None and kbt:
        found = find_minima(surface, None, min_depth=min_depth * kbt)
        table = Table("minimum", *names, f"F ({surface.units})", "depth (kT)")
        for i, m in enumerate(found):
            depth = "global" if m.depth == float("inf") else f"{m.depth / kbt:.2f}"
            table.add_row(str(i + 1), *[f"{v:.4g}" for v in m.point], f"{m.value:.3f}", depth)
        console.print(table)
    start_point = ", ".join(f"{n}={v:.4g}" for n, v in zip(names, path.points[0], strict=True))
    end_point = ", ".join(f"{n}={v:.4g}" for n, v in zip(names, path.points[-1], strict=True))
    console.print(f"path from ({start_point}) to ({end_point}), {images} images")
    if not path.metadata["converged"]:
        err_console.print(f"[yellow]not converged in {max_steps} steps[/yellow]")
    console.print(f"barrier          {energy(path.barrier)}")
    console.print(f"reverse barrier  {energy(path.reverse_barrier)}")
    console.print(f"reaction         {energy(path.reaction_energy)}")
    if path.profiles:
        barriers = path.profile_barriers()
        values = np.array(list(barriers.values()))
        listed = ", ".join(f"{k}: {v:.2f}" for k, v in barriers.items())
        console.print(f"barrier on each run along this path ({surface.units}): {listed}")
        if len(values) > 1:
            console.print(f"  spread between runs: {np.std(values):.2f} {surface.units}")
    console.print(f"wrote {target}")


UnitsOption = Annotated[
    str | None, typer.Option(help="Energy units shown: kJ/mol, kcal/mol, eV or kT.")
]
StyleOption = Annotated[
    str, typer.Option(help="Plot style: paper (default), slides, or a .mplstyle file.")
]
OutputsOption = Annotated[
    list[Path] | None,
    typer.Option("--output", "-o", help="Output file(s); the extension sets the format (repeat)."),
]


@plot_app.command("surface")
@handle_errors
def plot_surface_command(
    files: Annotated[
        list[Path], typer.Argument(help="Surface file(s); several = small multiples.")
    ],
    path: Annotated[
        Path | None, typer.Option("--path", help="Path file (.mep.dat) to draw on top.")
    ] = None,
    units: UnitsOption = None,
    vmax: Annotated[
        float | None, typer.Option(help="Top of the colour scale (default 20 kT).")
    ] = None,
    step: Annotated[
        float | None,
        typer.Option(help="Contour lines at fixed intervals from 0, e.g. 0.1 (0: no lines)."),
    ] = None,
    levels: Annotated[
        str | None,
        typer.Option(help="Contour levels, e.g. 0,0.1,0.2,0.4 (also sets the colour range)."),
    ] = None,
    ncols: Annotated[int, typer.Option(help="Columns of small multiples.")] = 3,
    contour_labels: Annotated[
        str, typer.Option(help="Numbers on the contour lines: all, some (default) or none.")
    ] = "some",
    style: StyleOption = "paper",
    output: OutputsOption = None,
) -> None:
    """Free-energy map (or property map / 1D profile), optionally with a path on top."""
    import matplotlib.pyplot as plt

    from estk.paths import MinimumEnergyPath
    from estk.plotting import labels_from_config, plot_style, plot_surface, plot_surfaces
    from estk.plotting.core import use_agg
    from estk.surface import Surface

    use_agg()
    level_list = None
    if levels is not None:
        try:
            level_list = [float(v) for v in levels.split(",") if v.strip()]
        except ValueError:
            raise EstError(f"--levels: expected numbers like 0,0.1,0.2, got {levels!r}") from None
    labels = labels_from_config(load_config(_state["config"]))
    mep_path = MinimumEnergyPath.load(path) if path else None
    surfaces = {f.stem: Surface.load(f) for f in files}
    targets = output or [files[0].with_suffix(".png")]
    with plot_style(style):
        if len(surfaces) == 1:
            ax = plot_surface(
                next(iter(surfaces.values())), path=mep_path, units=units, vmax=vmax,
                contour_step=step, levels=level_list, contour_labels=contour_labels, labels=labels,
            )  # fmt: skip
            fig = ax.figure
        else:
            fig = plot_surfaces(
                surfaces, ncols=ncols, path=mep_path, units=units, vmax=vmax, contour_step=step,
                levels=level_list, contour_labels=contour_labels, labels=labels,
            )  # fmt: skip
        for target in targets:
            fig.savefig(target)
        plt.close(fig)
    console.print(f"wrote {', '.join(str(t) for t in targets)}")


@plot_app.command("path")
@handle_errors
def plot_path_command(
    file: Annotated[Path, typer.Argument(help="Path file written by estk mep (.mep.dat).")],
    units: UnitsOption = None,
    runs: Annotated[
        bool, typer.Option("--runs/--no-runs", help="Draw each run's profile too.")
    ] = True,
    annotate: Annotated[
        bool, typer.Option("--annotate/--no-annotate", help="Write the barriers in the plot.")
    ] = True,
    style: StyleOption = "paper",
    output: OutputsOption = None,
) -> None:
    """Free energy along a minimum energy path, with each run's profile and the barriers."""
    import matplotlib.pyplot as plt

    from estk.paths import MinimumEnergyPath
    from estk.plotting import plot_path_profile, plot_style
    from estk.plotting.core import use_agg

    use_agg()
    mep_path = MinimumEnergyPath.load(file)
    name = file.name.removesuffix(".dat")
    targets = output or [file.with_name(name + ".png")]
    with plot_style(style):
        ax = plot_path_profile(mep_path, units=units, runs=runs, annotate=annotate)
        for target in targets:
            ax.figure.savefig(target)
        plt.close(ax.figure)
    console.print(f"wrote {', '.join(str(t) for t in targets)}")


def _parse_point(text: str | None, option: str, ndim: int) -> list[float] | None:
    if text is None:
        return None
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != ndim:
        raise EstError(f"--{option}: expected {ndim} comma-separated numbers, got {text!r}")
    return values


def _combine_mode(text: str) -> str | None:
    from estk.reweight import COMBINE_MODES

    if text.lower() == "none":
        return None
    if text.lower() not in COMBINE_MODES:
        raise EstError(f"unknown mode '{text}' (choose from {', '.join(COMBINE_MODES)}, none)")
    return text.lower()


def _parse_points(text: str) -> int | list[int]:
    try:
        values = [int(v) for v in text.split(",")]
    except ValueError:
        raise EstError(f"--points: expected integers like 100 or 100,80, got {text!r}") from None
    return values[0] if len(values) == 1 else values


def _parse_assignments(
    items: list[str] | None, option: str, *, pair: bool
) -> dict[str, Any] | None:
    """Parse ``NAME=value`` (or ``NAME=lo,hi`` with ``pair``; either side may be empty)."""
    if not items:
        return None
    parsed: dict[str, Any] = {}
    for item in items:
        name, sep, value = item.partition("=")
        try:
            if not sep or not name:
                raise ValueError
            if pair:
                lo, comma, hi = value.partition(",")
                if not comma:
                    raise ValueError
                parsed[name] = (float(lo) if lo else None, float(hi) if hi else None)
            else:
                parsed[name] = float(value)
        except ValueError:
            example = "d0=1.5,2.5" if pair else "d0=0.02"
            raise EstError(f"--{option}: expected e.g. {example}, got {item!r}") from None
    return parsed


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:g}"


@config_app.command("show")
@handle_errors
def config_show() -> None:
    """Print the settings in effect and where each one comes from."""
    cfg = load_config(_state["config"])
    if cfg.path is None:
        console.print(f"no {CONFIG_FILENAME} found; using defaults")
    else:
        console.print(f"configuration: {cfg.path}")

    table = Table("setting", "value", "source")
    for key, value, source in cfg.explain():
        style = None if source == "default" else "bold"
        table.add_row(key, _format_value(value), source, style=style)
    table.add_row("(runs root, resolved)", str(cfg.runs_root), "", style="dim")
    console.print(table)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items()) or "{}"
    return str(value)


@config_app.command("validate")
@handle_errors
def config_validate() -> None:
    """Check estk.yaml; exits with code 1 if it is invalid."""
    path = _state["config"] or find_config()
    if path is None:
        raise EstError(f"no {CONFIG_FILENAME} found in this folder or its parents")
    load_config(path)
    console.print(f"{path}: OK")


@config_app.command("migrate")
@handle_errors
def config_migrate(
    case_file: Annotated[Path, typer.Argument(help="Old Hydra case file (config/case/*.yaml).")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write here instead of printing.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite the output file.")] = False,
) -> None:
    """Convert an old Hydra case file to estk.yaml."""
    from estk.migrate import migrate_case_yaml

    text, dropped = migrate_case_yaml(case_file)
    if output is None:
        console.print(text, highlight=False, markup=False, soft_wrap=True)
    else:
        if output.exists() and not force:
            raise EstError(f"{output} already exists (use --force to overwrite)")
        output.write_text(text)
        load_config(output)  # make sure the result is valid
        err_console.print(f"wrote {output}")

    if dropped:
        err_console.print(f"\n[yellow]not migrated ({len(dropped)}):[/yellow]")
        for item in dropped:
            err_console.print(f"  {item}", highlight=False, markup=False)
