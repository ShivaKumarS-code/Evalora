"""Markdown PR-comment body for a gate report."""

from __future__ import annotations

import math

from evalora.gate.compare import GateReport, MetricComparison

_QUESTION_LIMIT = 60


def _delta(diff: float | None) -> str:
    if diff is None:
        return "n/a"
    if math.isclose(diff, 0.0, abs_tol=1e-9):
        return "±0.000"
    if diff < 0:
        return f"↓ {-diff:.3f}"
    return f"↑ {diff:.3f}"


def _verdict(metric: MetricComparison) -> str:
    return "❌ fail" if metric.regressed else "✅ pass"


def _shorten(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= _QUESTION_LIMIT:
        return text
    return text[: _QUESTION_LIMIT - 1].rstrip() + "…"


def _metric_row(metric: MetricComparison) -> str:
    baseline = "n/a" if metric.baseline is None else f"{metric.baseline:.3f}"
    current = "n/a" if metric.current is None else f"{metric.current:.3f}"
    span = ""
    if metric.ci_low is not None and metric.ci_high is not None:
        span = f" (CI {metric.ci_low:.3f} to {metric.ci_high:.3f})"
    return (
        f"| {metric.metric} | {baseline} | {current}{span} "
        f"| {_delta(metric.diff)} | {_verdict(metric)} |"
    )


def render_markdown(report: GateReport) -> str:
    """Render a gate report as the body of a pull-request comment."""
    icon = "✅ passed" if report.passed else "❌ failed"
    lines = [
        f"## Evalora quality gate: {icon}",
        "",
        f"`{report.run_name}` vs baseline `{report.baseline_run_name}` "
        f"· {report.paired_cases} paired cases "
        f"· {report.ci_confidence:.0%} CI "
        f"({report.bootstrap_resamples} resamples, seed {report.seed})",
        "",
        "| Metric | Baseline | Current | Δ | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(_metric_row(metric) for metric in report.metrics)

    if report.new_failures_by_category:
        lines += ["", "### New failures by category", ""]
        lines += ["| Category | New failures |", "| --- | --- |"]
        for category, count in sorted(
            report.new_failures_by_category.items(), key=lambda item: item[0].value
        ):
            lines.append(f"| `{category.value}` | {count} |")

    if report.regressed_cases:
        lines += ["", "### Top regressed cases", ""]
        lines += [
            "| Case | Type | Question | Faithfulness | Failure |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in report.regressed_cases:
            if case.baseline_faithfulness is not None and case.current_faithfulness is not None:
                faithfulness = f"{case.baseline_faithfulness:.2f} → {case.current_faithfulness:.2f}"
            else:
                faithfulness = "n/a"
            failure = f"`{case.current_category.value}`" if case.current_category else "none"
            lines.append(
                f"| `{case.case_id}` | {case.question_type} "
                f"| {_shorten(case.question)} | {faithfulness} | {failure} |"
            )

    if report.new_case_count or report.missing_case_count:
        lines += [
            "",
            f"Cases only in the current run: {report.new_case_count}; "
            f"missing versus the baseline: {report.missing_case_count}.",
        ]

    return "\n".join(lines) + "\n"
