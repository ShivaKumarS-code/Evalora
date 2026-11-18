"""Thin Typer command line interface."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from evalora import __version__
from evalora.config import (
    DEFAULT_CONFIG_YAML,
    ConfigError,
    GateConfig,
    ThresholdConfig,
    load_config,
    redacted_config_snapshot,
)
from evalora.evaluate.aggregate import DEFAULT_RESAMPLES, DEFAULT_SEED, aggregate_run
from evalora.evaluate.judges import judge_case
from evalora.evaluate.prompts import CLAIM_VERIFICATION_PROMPTS, JUDGE_PROMPT_VERSIONS
from evalora.evaluate.retrieval import RetrievalSummary, aggregate_retrieval
from evalora.gate.baseline import (
    BaselineFileError,
    export_baseline,
    import_baseline,
    read_baseline_file,
    write_baseline_file,
)
from evalora.gate.compare import (
    DEFAULT_RESAMPLES as GATE_RESAMPLES,
)
from evalora.gate.compare import (
    DEFAULT_SEED as GATE_SEED,
)
from evalora.gate.compare import GateError, compare_runs
from evalora.gate.markdown import render_markdown
from evalora.generate.generator import (
    QUESTION_TYPE_ORDER,
    GenerationError,
    generate_test_set,
)
from evalora.ingest.loaders import DocumentLoadError
from evalora.llm.base import LLMError
from evalora.llm.factory import build_llm_client
from evalora.report.builder import build_html_report, compare_to_baseline
from evalora.run.estimate import estimate_run
from evalora.run.runner import TestSetError, hash_test_set, load_test_set, run_test_set
from evalora.schemas import CaseResult
from evalora.store.db import RunRecord, RunStore, StoreError, get_git_sha
from evalora.validate.judge_eval import (
    DEFAULT_RESAMPLES as VALIDATION_RESAMPLES,
)
from evalora.validate.judge_eval import (
    DEFAULT_SEED as VALIDATION_SEED,
)
from evalora.validate.judge_eval import (
    JudgeValidationSummary,
    MetricWithCI,
    run_validation,
)
from evalora.validate.labels import (
    LabelExportError,
    LabelFileError,
    build_label_sheet,
    read_labels,
    write_label_sheet,
    write_labels,
)
from evalora.validate.ragtruth import (
    RagTruthError,
    build_ragtruth_sample,
    load_ragtruth,
    write_manifest,
)


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


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _summary_table(
    run_name: str,
    summary: RetrievalSummary,
    stored: int,
    total: int,
) -> Table:
    table = Table(title=f"Run: {run_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Cases completed", f"{stored}/{total}")
    table.add_row("Answerable cases", str(summary.answerable_cases))
    table.add_row("Unanswerable cases", str(summary.unanswerable_cases))
    table.add_row("Hit rate", _format_rate(summary.hit_rate))
    table.add_row("MRR", _format_rate(summary.mrr))
    table.add_row("Context recall", _format_rate(summary.context_recall))
    table.add_row(
        "Unanswerable with retrieved context",
        f"{summary.unanswerable_with_retrieval}/{summary.unanswerable_cases}",
    )
    table.add_row("Pipeline errors", str(summary.pipeline_errors))
    return table


@app.command()
def run(
    ctx: typer.Context,
    run_name: Annotated[str, typer.Option("--run-name", help="Name for this run.")],
    config_path: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the evalora config file."),
    ] = Path("evalora.yaml"),
    tests: Annotated[
        Path,
        typer.Option("--tests", "-t", help="Path to the JSONL test set."),
    ] = Path("test_set.jsonl"),
    no_judge: Annotated[
        bool,
        typer.Option("--no-judge", help="Skip LLM judging and store retrieval-only results."),
    ] = False,
    max_cases: Annotated[
        int | None,
        typer.Option("--max-cases", min=1, help="Run at most this many pending cases."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print a call and cost estimate for the pending cases and exit.",
        ),
    ] = False,
    max_budget_usd: Annotated[
        float | None,
        typer.Option(
            "--max-budget-usd",
            min=0.0,
            help=(
                "Abort when projected judge spend exceeds this budget. "
                "Needs a model with verified pricing."
            ),
        ),
    ] = None,
) -> None:
    """Run a test set against the configured pipeline."""
    state = ctx.obj if isinstance(ctx.obj, AppState) else AppState()
    console = Console()
    try:
        config = load_config(config_path)
        cases = load_test_set(tests)
        store = RunStore()
        if dry_run:
            existing = store.get_run(run_name)
            completed = store.completed_case_ids(run_name) if existing is not None else set()
            pending = [case for case in cases if case.case_id not in completed]
            if max_cases is not None:
                pending = pending[:max_cases]
            estimate = estimate_run(
                pending,
                provider=config.judge.provider,
                model=config.judge.model,
                judge_enabled=not no_judge,
            )
            price_note = (
                f"{config.judge.model}"
                if estimate.price_known
                else f"no verified price for '{config.judge.model}', reported as $0"
            )
            typer.echo(
                f"Dry run for '{run_name}': {estimate.pending_cases} pending of {len(cases)} cases"
            )
            typer.echo(f"  pipeline calls: {estimate.pipeline_calls}")
            typer.echo(f"  judge calls: ~{estimate.judge_calls}")
            typer.echo(
                f"  estimated tokens: ~{estimate.estimated_prompt_tokens} prompt, "
                f"~{estimate.estimated_completion_tokens} completion"
            )
            typer.echo(f"  approximate cost: ${estimate.estimated_cost_usd:.6f} ({price_note})")
            if max_budget_usd is not None and estimate.price_known and not no_judge:
                if estimate.estimated_cost_usd > max_budget_usd:
                    typer.echo(
                        f"  projected spend exceeds --max-budget-usd "
                        f"${max_budget_usd:.2f}; the run would abort"
                    )
                else:
                    typer.echo(f"  within budget ${max_budget_usd:.2f}")
            return
        judge_client = (
            None
            if no_judge
            else build_llm_client(model=config.judge.model, no_cache=state.no_cache)
        )
        snapshot = redacted_config_snapshot(config)
        snapshot["judge_runtime"] = {
            "enabled": not no_judge,
            "model": config.judge.model,
            "prompt_versions": dict(JUDGE_PROMPT_VERSIONS),
        }
        store.get_or_create_run(
            run_name,
            config_snapshot=snapshot,
            test_set_hash=hash_test_set(tests),
            case_count=len(cases),
            git_sha=get_git_sha(),
        )
        completed = store.completed_case_ids(run_name)
        pending = [case for case in cases if case.case_id not in completed]
        if max_cases is not None:
            pending = pending[:max_cases]
        budget_enforceable = False
        if max_budget_usd is not None and not no_judge:
            estimate = estimate_run(
                pending,
                provider=config.judge.provider,
                model=config.judge.model,
                judge_enabled=True,
            )
            if not estimate.price_known:
                typer.echo(
                    f"warning: no verified price for '{config.judge.model}'; "
                    "--max-budget-usd cannot be enforced",
                    err=True,
                )
            elif estimate.estimated_cost_usd > max_budget_usd:
                raise TestSetError(
                    f"projected spend ${estimate.estimated_cost_usd:.6f} exceeds "
                    f"--max-budget-usd ${max_budget_usd:.2f}; aborting before any "
                    "calls (reduce --max-cases or use --no-judge to cut cost)"
                )
            else:
                budget_enforceable = True
        if pending:
            spent = 0.0

            async def processor(result: CaseResult) -> CaseResult:
                if judge_client is None:
                    return result
                return await judge_case(result, judge_client)

            def stop_check() -> bool:
                return budget_enforceable and spent > (max_budget_usd or 0.0)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(f"Running {run_name}", total=len(pending))

                def on_result(result: CaseResult) -> None:
                    nonlocal spent
                    spent += result.usage.cost_usd
                    store.save_result(run_name, result)
                    progress.advance(task_id)

                executed = asyncio.run(
                    run_test_set(
                        pending,
                        config.pipeline,
                        completed_case_ids=set(),
                        on_result=on_result,
                        processor=processor,
                        stop_check=stop_check,
                    )
                )
            skipped = len(pending) - len(executed)
            if skipped:
                typer.echo(
                    f"warning: budget reached, {skipped} case(s) skipped; "
                    "rerun with a higher --max-budget-usd to resume",
                    err=True,
                )
        else:
            typer.echo("All cases already completed.")
        stored = store.load_results(run_name)
    except (ConfigError, TestSetError, StoreError, LLMError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    console.print(_summary_table(run_name, aggregate_retrieval(stored), len(stored), len(cases)))
    console.print(f"Stored results: {store.path}")
    if not no_judge:
        console.print(f"Build the full report with: evalora report --run {run_name}")


def _ci_confidence(run: RunRecord) -> float:
    gate = run.config_snapshot.get("gate")
    if isinstance(gate, dict):
        value = gate.get("ci_confidence", 0.95)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.95


@app.command()
def report(
    run_name: Annotated[str, typer.Option("--run", help="Run name to report.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the self-contained HTML report."),
    ] = Path("report.html"),
    baseline_name: Annotated[
        str | None,
        typer.Option("--baseline", help="Baseline run name for a diff section."),
    ] = None,
    judge_validation_path: Annotated[
        Path | None,
        typer.Option(
            "--judge-validation",
            help="Judge validation JSON to embed (defaults to judge_validation.json if present).",
        ),
    ] = None,
) -> None:
    """Build a self-contained HTML report for a stored run."""
    try:
        store = RunStore()
        run = store.get_run(run_name)
        if run is None:
            raise StoreError(f"run '{run_name}' not found")
        results = store.load_results(run_name)
        confidence = _ci_confidence(run)
        aggregate = aggregate_run(
            results,
            confidence=confidence,
            resamples=DEFAULT_RESAMPLES,
            seed=DEFAULT_SEED,
        )
        comparison = None
        if baseline_name is not None:
            resolved_baseline = _resolve_baseline(store, baseline_name)
            if resolved_baseline is None:
                raise StoreError(f"baseline '{baseline_name}' not found")
            baseline_run = store.get_run(resolved_baseline)
            if baseline_run is None:
                raise StoreError(f"baseline run '{resolved_baseline}' not found")
            baseline_results = store.load_results(resolved_baseline)
            baseline_aggregate = aggregate_run(
                baseline_results,
                confidence=confidence,
                resamples=DEFAULT_RESAMPLES,
                seed=DEFAULT_SEED,
            )
            comparison = compare_to_baseline(
                current=aggregate,
                baseline=baseline_aggregate,
                current_results=results,
                baseline_results=baseline_results,
                baseline_run_name=resolved_baseline,
            )
        validation_path = judge_validation_path
        if validation_path is None and Path("judge_validation.json").is_file():
            validation_path = Path("judge_validation.json")
        judge_validation = None
        if validation_path is not None:
            try:
                judge_validation = JudgeValidationSummary.model_validate_json(
                    validation_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                raise StoreError(
                    f"could not read judge validation summary {validation_path}: {exc}"
                ) from exc
        build_html_report(
            run=run,
            results=results,
            aggregate=aggregate,
            out=out,
            baseline=comparison,
            judge_validation=judge_validation,
        )
    except StoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {out}")


def _format_metric(metric: MetricWithCI) -> str:
    if metric.value is None:
        return "n/a"
    if metric.ci_low is not None and metric.ci_high is not None:
        return f"{metric.value:.3f} [{metric.ci_low:.3f}, {metric.ci_high:.3f}]"
    return f"{metric.value:.3f}"


@app.command()
def export_labels(
    run_name: Annotated[str, typer.Option("--run", help="Run name to export claims from.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the local labelling HTML."),
    ] = Path("labels.html"),
    max_items: Annotated[
        int,
        typer.Option("--max-items", min=1, help="Maximum number of claim items to export."),
    ] = 150,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Seed used when sampling claim items."),
    ] = 42,
) -> None:
    """Export a local HTML sheet for hand-labelling judge claims."""
    try:
        store = RunStore()
        if store.get_run(run_name) is None:
            raise StoreError(f"run '{run_name}' not found")
        results = store.load_results(run_name)
        sheet = build_label_sheet(
            results,
            run_name=run_name,
            max_items=max_items,
            seed=seed,
        )
        write_label_sheet(sheet, out)
    except (StoreError, LabelExportError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote {len(sheet.items)} of {sheet.available_count} claim items to {out}")
    if sheet.sampled:
        typer.echo(f"Sampled with seed {sheet.seed}.")
    typer.echo(
        "Label each claim, download labels.jsonl, then run "
        "evalora validate-judge --labels labels.jsonl"
    )


@app.command()
def validate_judge(
    ctx: typer.Context,
    labels: Annotated[
        Path,
        typer.Option("--labels", "-l", help="JSONL file of labelled items."),
    ] = Path("labels.jsonl"),
    model: Annotated[
        str,
        typer.Option("--model", help="Gemini model used as the judge."),
    ] = "gemini-3.5-flash-lite",
    prompt_version: Annotated[
        str,
        typer.Option("--prompt-version", help="Claim verification prompt version (v1 or v2)."),
    ] = "v1",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the validation JSON."),
    ] = Path("judge_validation.json"),
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="Dataset name recorded in the summary."),
    ] = "manual",
    seed: Annotated[
        int,
        typer.Option("--seed", help="Bootstrap seed."),
    ] = VALIDATION_SEED,
    resamples: Annotated[
        int,
        typer.Option("--resamples", min=1, help="Bootstrap resamples."),
    ] = VALIDATION_RESAMPLES,
) -> None:
    """Measure judge agreement against hand labels."""
    state = ctx.obj if isinstance(ctx.obj, AppState) else AppState()
    verification_prompt = CLAIM_VERIFICATION_PROMPTS.get(prompt_version)
    if verification_prompt is None:
        typer.echo(
            f"error: unknown prompt version '{prompt_version}'; "
            f"choose one of {', '.join(CLAIM_VERIFICATION_PROMPTS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        items = read_labels(labels)
        sample_seed: int | None = None
        sample_size: int | None = None
        for item in items:
            item_seed = item.metadata.get("sample_seed")
            item_size = item.metadata.get("sample_size")
            if isinstance(item_seed, int) and isinstance(item_size, int):
                sample_seed = item_seed
                sample_size = item_size
                break
        dataset_name = (
            "RAGTruth"
            if dataset == "manual" and all(item.source == "RAGTruth" for item in items)
            else dataset
        )
        client = build_llm_client(model=model, no_cache=state.no_cache)
        typer.echo(f"Validating {len(items)} items with {model} ({prompt_version})...")
        summary = asyncio.run(
            run_validation(
                items,
                client,
                model=model,
                prompt_version=prompt_version,
                dataset=dataset_name,
                verification_prompt=verification_prompt,
                sample_seed=sample_seed,
                sample_size=sample_size if sample_size is not None else len(items),
                resamples=resamples,
                seed=seed,
            )
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (LabelFileError, LLMError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    console = Console()
    table = Table(title="Judge validation")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Items", str(summary.n))
    table.add_row("Precision", _format_metric(summary.precision))
    table.add_row("Recall", _format_metric(summary.recall))
    table.add_row("F1", _format_metric(summary.f1))
    table.add_row("Accuracy", _format_metric(summary.accuracy))
    table.add_row("Cohen's kappa", _format_metric(summary.kappa))
    confusion = summary.confusion
    table.add_row(
        "TP / FP / FN / TN",
        f"{confusion.tp} / {confusion.fp} / {confusion.fn} / {confusion.tn}",
    )
    console.print(table)
    console.print(f"Wrote {out}")


@app.command()
def ragtruth_sample(
    responses_path: Annotated[
        Path,
        typer.Option("--responses", help="Path to RAGTruth response.jsonl."),
    ],
    sources_path: Annotated[
        Path,
        typer.Option("--sources", help="Path to RAGTruth source_info.jsonl."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the sampled label JSONL."),
    ] = Path("ragtruth_sample.jsonl"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="Where to write the sample manifest."),
    ] = Path("ragtruth_sample_manifest.json"),
    sample_size: Annotated[
        int,
        typer.Option("--sample-size", min=1, help="Number of responses to sample."),
    ] = 250,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Sampling seed."),
    ] = 42,
    split: Annotated[
        str,
        typer.Option("--split", help="RAGTruth split to sample, or 'all'."),
    ] = "test",
) -> None:
    """Sample RAGTruth responses into a labelled JSONL file."""
    split_value = None if split.lower() in {"all", "none", ""} else split
    try:
        responses, sources = load_ragtruth(responses_path, sources_path)
        items, manifest = build_ragtruth_sample(
            responses,
            sources,
            sample_size=sample_size,
            seed=seed,
            split=split_value,
        )
        write_labels(out, items)
        write_manifest(manifest, manifest_path)
    except (RagTruthError, LabelFileError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Wrote {manifest.sample_size} RAGTruth items to {out} "
        f"(seed {manifest.sample_seed}, split {manifest.split})"
    )
    typer.echo(
        f"  hallucination={manifest.hallucination_items}, "
        f"supported={manifest.supported_items}, eligible={manifest.eligible_count}"
    )
    typer.echo(f"  license={manifest.license}; manifest={manifest_path}")


baseline_app = typer.Typer(help="Manage baseline runs for the quality gate.")
app.add_typer(baseline_app, name="baseline")


@baseline_app.command("set")
def baseline_set(
    run_name: Annotated[str, typer.Option("--run", help="Run to register as a baseline.")],
    name: Annotated[
        str,
        typer.Option("--as", help="Baseline alias to create or replace."),
    ] = "main",
) -> None:
    """Point a named baseline at a stored run."""
    try:
        store = RunStore()
        if store.get_run(run_name) is None:
            raise StoreError(f"run '{run_name}' not found")
        store.set_baseline(name, run_name)
    except StoreError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Baseline '{name}' now points at run '{run_name}'")


@baseline_app.command("export")
def baseline_export(
    run_name: Annotated[str, typer.Option("--run", help="Run to export as a baseline.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Where to write the baseline JSON."),
    ] = Path("baseline.json"),
) -> None:
    """Export a run as a portable baseline JSON artifact."""
    try:
        store = RunStore()
        export = export_baseline(store, run_name)
        write_baseline_file(export, out)
    except (StoreError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Wrote baseline for run '{run_name}' ({len(export.results)} results) to {out}")


@baseline_app.command("import")
def baseline_import(
    file: Annotated[Path, typer.Option("--file", "-f", help="Baseline JSON to import.")],
    name: Annotated[
        str,
        typer.Option("--as", help="Baseline alias to point at the imported run."),
    ] = "main",
) -> None:
    """Import a baseline JSON artifact and register it as a named baseline."""
    try:
        export = read_baseline_file(file)
        store = RunStore()
        record = import_baseline(store, export, baseline_name=name)
    except (BaselineFileError, StoreError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Imported run '{record.run_name}' ({len(export.results)} results) as baseline '{name}'"
    )


def _default_baseline_name(run: RunRecord) -> str:
    gate = run.config_snapshot.get("gate")
    if isinstance(gate, dict):
        baseline = gate.get("baseline")
        if isinstance(baseline, str) and baseline:
            return baseline
    return "main"


def _gate_settings(
    run: RunRecord, config_path: Path | None
) -> tuple[dict[str, ThresholdConfig], float]:
    """Gate thresholds and CI confidence from an explicit config or the snapshot."""
    if config_path is not None:
        config = load_config(config_path)
        return dict(config.gate.thresholds), config.gate.ci_confidence
    try:
        raw = run.config_snapshot.get("gate")
        gate = GateConfig.model_validate(raw) if raw is not None else GateConfig()
    except ValidationError:
        gate = GateConfig()
    return dict(gate.thresholds), gate.ci_confidence


def _resolve_baseline(store: RunStore, name: str) -> str | None:
    """Resolve a baseline alias first, then a direct run name."""
    alias = store.get_baseline(name)
    if alias is not None:
        return alias
    if store.get_run(name) is not None:
        return name
    return None


@app.command()
def gate(
    run_name: Annotated[str, typer.Option("--run", help="Run to evaluate against the baseline.")],
    baseline_name: Annotated[
        str | None,
        typer.Option(
            "--baseline",
            help="Baseline alias or run name; defaults to the config's gate.baseline.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Optional config for gate thresholds; defaults to the run snapshot.",
        ),
    ] = None,
    markdown_out: Annotated[
        Path | None,
        typer.Option("--markdown", help="Write a PR-comment markdown summary to this file."),
    ] = None,
    resamples: Annotated[
        int,
        typer.Option("--resamples", min=1, help="Bootstrap resamples for the diff CI."),
    ] = GATE_RESAMPLES,
    seed: Annotated[int, typer.Option("--seed", help="Bootstrap seed.")] = GATE_SEED,
) -> None:
    """Fail (exit 1) when quality regressed against the baseline.

    Tool errors exit with code 2; a regression exits with code 1.
    """
    try:
        store = RunStore()
        run = store.get_run(run_name)
        if run is None:
            raise StoreError(f"run '{run_name}' not found")
        name = baseline_name or _default_baseline_name(run)
        resolved = _resolve_baseline(store, name)
        if resolved is None:
            raise StoreError(
                f"baseline '{name}' not found; create it with "
                "`evalora baseline set --run <name> --as " + name + "`"
            )
        current_results = store.load_results(run_name)
        baseline_results = store.load_results(resolved)
        thresholds, confidence = _gate_settings(run, config_path)
        report = compare_runs(
            current_results,
            baseline_results,
            run_name=run_name,
            baseline_run_name=resolved,
            thresholds=thresholds,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
    except (StoreError, ConfigError, GateError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    console = Console()
    icon = "passed" if report.passed else "FAILED"
    table = Table(title=f"Quality gate: {run_name} vs {report.baseline_run_name} ({icon})")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Diff", justify="right")
    table.add_column("CI", justify="right")
    table.add_column("Verdict")
    for metric in report.metrics:
        baseline = "n/a" if metric.baseline is None else f"{metric.baseline:.3f}"
        current = "n/a" if metric.current is None else f"{metric.current:.3f}"
        diff = "n/a" if metric.diff is None else f"{metric.diff:+.3f}"
        span = (
            "n/a"
            if metric.ci_low is None or metric.ci_high is None
            else f"[{metric.ci_low:.3f}, {metric.ci_high:.3f}]"
        )
        verdict = "fail" if metric.regressed else "pass"
        table.add_row(metric.metric, baseline, current, diff, span, verdict)
    console.print(table)
    if report.new_failures_by_category:
        console.print("New failures by category:")
        for category, count in sorted(
            report.new_failures_by_category.items(), key=lambda item: item[0].value
        ):
            console.print(f"  {category.value}: {count}")
    for regressed in report.regressed_cases:
        console.print(
            f"  regressed: {regressed.case_id} ({regressed.question_type}) "
            f"now `{regressed.current_category.value if regressed.current_category else 'none'}`"
        )

    if markdown_out is not None:
        try:
            markdown_out.parent.mkdir(parents=True, exist_ok=True)
            markdown_out.write_text(render_markdown(report), encoding="utf-8")
        except OSError as exc:
            typer.echo(f"error: could not write markdown summary: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        typer.echo(f"Wrote markdown summary to {markdown_out}")

    if not report.passed:
        raise typer.Exit(code=1)
