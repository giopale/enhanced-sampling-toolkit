"""Entry point of the ``estk`` command."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

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
