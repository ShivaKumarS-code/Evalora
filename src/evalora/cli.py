"""Thin Typer command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from evalora import __version__
from evalora.config import DEFAULT_CONFIG_YAML, load_config


class AppState:
    def __init__(self, *, no_cache: bool = False) -> None:
        self.no_cache = no_cache


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate RAG test suites and gate RAG quality in CI.",
)


@app.callback()
def main(
    ctx: typer.Context,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Bypass the LLM response cache.")
    ] = False,
) -> None:
    ctx.obj = AppState(no_cache=no_cache)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"evalora {__version__}")


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Option("--path", "-p", help="Where to write the starter config."),
    ] = Path("evalora.yaml"),
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing config file.")
    ] = False,
) -> None:
    """Write a starter evalora.yaml."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
    load_config(path)
    typer.echo(f"Wrote {path}")
