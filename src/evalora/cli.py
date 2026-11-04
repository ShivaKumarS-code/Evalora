"""Thin Typer command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from evalora import __version__
from evalora.config import DEFAULT_CONFIG_YAML, ConfigError, load_config
from evalora.generate.generator import (
    QUESTION_TYPE_ORDER,
    GenerationError,
    generate_test_set,
)
from evalora.ingest.loaders import DocumentLoadError
from evalora.llm.base import LLMError
from evalora.llm.factory import build_llm_client


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


@app.command()
def generate(
    ctx: typer.Context,
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the evalora config file."),
    ] = Path("evalora.yaml"),
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the generated JSONL test set."),
    ] = Path("test_set.jsonl"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Cap the total number of generated cases."),
    ] = None,
) -> None:
    """Generate a filtered test set from the configured documents."""
    state = ctx.obj if isinstance(ctx.obj, AppState) else AppState()
    try:
        config = load_config(config_path)
        client = build_llm_client(model=config.generation.model, no_cache=state.no_cache)
        result = asyncio.run(generate_test_set(config, client, out=out, limit=limit))
    except (ConfigError, DocumentLoadError, GenerationError, LLMError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {len(result.cases)} cases to {out}")
    for question_type in QUESTION_TYPE_ORDER:
        typer.echo(
            f"  {question_type.value}: "
            f"{result.report.kept.get(question_type, 0)}/"
            f"{result.report.requested.get(question_type, 0)}"
        )
    if result.report.dropped:
        drop_summary = ", ".join(
            f"{reason.value}={count}"
            for reason, count in sorted(
                result.report.dropped.items(),
                key=lambda item: item[0].value,
            )
        )
        typer.echo(f"  dropped: {drop_summary}")
    usage = result.report.token_usage
    typer.echo(
        f"  tokens: {usage.total_tokens} "
        f"(prompt {usage.prompt_tokens}, completion {usage.completion_tokens}), "
        f"cost ${usage.cost_usd:.6f}"
    )
    typer.echo(f"  report: {out.with_name('generation_report.json')}")
