"""Build one self-contained HTML report with no external dependencies."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel

from evalora.evaluate.aggregate import MetricSummary, RunAggregate
from evalora.schemas import CaseResult, FailureCategory, QuestionType
from evalora.store.db import RunRecord

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class MetricDelta(BaseModel):
    name: str
    current: float | None
    baseline: float | None
    delta: float | None


class BaselineComparison(BaseModel):
    baseline_run_name: str
    metrics: list[MetricDelta]
    flipped_cases: list[CaseResult]


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _metric_delta(
    name: str,
    current: MetricSummary,
    baseline: MetricSummary,
) -> MetricDelta:
    delta = None
    if current.mean is not None and baseline.mean is not None:
        delta = current.mean - baseline.mean
    return MetricDelta(
        name=name,
        current=current.mean,
        baseline=baseline.mean,
        delta=delta,
    )


def compare_to_baseline(
    *,
    current: RunAggregate,
    baseline: RunAggregate,
    current_results: Sequence[CaseResult],
    baseline_results: Sequence[CaseResult],
    baseline_run_name: str,
) -> BaselineComparison:
    metrics = [
        _metric_delta("Faithfulness", current.faithfulness, baseline.faithfulness),
        _metric_delta("Correctness", current.correctness, baseline.correctness),
        _metric_delta("Hit rate", current.hit_rate, baseline.hit_rate),
        _metric_delta("MRR", current.mrr, baseline.mrr),
        _metric_delta("Context recall", current.context_recall, baseline.context_recall),
    ]
    baseline_by_id = {result.case.case_id: result for result in baseline_results}
    flipped = [
        result
        for result in current_results
        if result.failure_category is not None
        and (baseline_result := baseline_by_id.get(result.case.case_id)) is not None
        and baseline_result.failure_category is None
    ]
    return BaselineComparison(
        baseline_run_name=baseline_run_name,
        metrics=metrics,
        flipped_cases=sorted(flipped, key=lambda result: result.case.case_id),
    )


def build_html_report(
    *,
    run: RunRecord,
    results: Sequence[CaseResult],
    aggregate: RunAggregate,
    out: Path,
    baseline: BaselineComparison | None = None,
) -> Path:
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    environment.filters["fmt"] = _fmt
    environment.filters["fmt_delta"] = _fmt_delta
    template = environment.get_template("report.html.j2")

    metric_cards: list[tuple[str, MetricSummary]] = [
        ("Faithfulness", aggregate.faithfulness),
        ("Correctness", aggregate.correctness),
        ("Hit rate", aggregate.hit_rate),
        ("MRR", aggregate.mrr),
        ("Context recall", aggregate.context_recall),
    ]
    type_rows = [
        (question_type, aggregate.by_type[question_type]) for question_type in QuestionType
    ]
    failure_rows = [(category, aggregate.failure_counts[category]) for category in FailureCategory]
    html = template.render(
        run=run,
        results=results,
        aggregate=aggregate,
        baseline=baseline,
        metric_cards=metric_cards,
        type_rows=type_rows,
        failure_rows=failure_rows,
        question_types=list(QuestionType),
        failure_categories=list(FailureCategory),
        config_json=json.dumps(run.config_snapshot, indent=2, sort_keys=True),
        ci_label=f"{int(aggregate.ci_confidence * 100)}% CI",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
