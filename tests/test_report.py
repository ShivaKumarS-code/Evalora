from __future__ import annotations

from pathlib import Path

from evalora.evaluate.aggregate import aggregate_run
from evalora.report.builder import build_html_report, compare_to_baseline
from evalora.schemas import (
    ClaimVerdict,
    CorrectnessLabel,
    FailureCategory,
    QuestionType,
    RetrievedContext,
)
from tests.factories import make_result, make_run_record


def _sample_results() -> list:
    results = []
    for index, category in enumerate(FailureCategory):
        hallucination = category == FailureCategory.HALLUCINATION
        wrong = category in {
            FailureCategory.RETRIEVAL_FAILURE,
            FailureCategory.GENERATION_FAILURE,
            FailureCategory.HALLUCINATION,
        }
        results.append(
            make_result(
                f"case-{index}",
                QuestionType.SINGLE_HOP,
                answer=f"answer for {category.value}",
                contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994.")],
                hit=category != FailureCategory.RETRIEVAL_FAILURE,
                faithfulness=0.5 if hallucination else 1.0,
                claim_verdicts=[
                    ClaimVerdict(
                        claim=(
                            "Unsupported claim text" if hallucination else "Supported claim text"
                        ),
                        supported=not hallucination,
                        evidence_context_index=None if hallucination else 0,
                        rationale="judge rationale",
                    )
                ],
                correct=not wrong,
                correctness_label=(
                    CorrectnessLabel.INCORRECT if wrong else CorrectnessLabel.CORRECT
                ),
                abstained=category == FailureCategory.WRONG_ABSTENTION,
                failure_category=category,
            )
        )
    results.append(
        make_result(
            "case-html",
            QuestionType.SINGLE_HOP,
            answer="<script>alert(1)</script>",
            failure_category=None,
            question="What is <b>unsafe</b>?",
        )
    )
    return results


def test_report_is_self_contained_and_covers_every_failure(tmp_path: Path) -> None:
    results = _sample_results()
    run = make_run_record("report-run", results=results)
    aggregate = aggregate_run(results, resamples=200, seed=7)
    out = tmp_path / "report.html"

    built = build_html_report(
        run=run,
        results=results,
        aggregate=aggregate,
        out=out,
    )

    html = built.read_text(encoding="utf-8")
    assert built == out
    assert "<style>" in html
    assert "<script>" in html
    assert "<script src=" not in html
    assert "<link " not in html
    assert "cdn" not in html.lower()
    for category in FailureCategory:
        assert category.value in html
    assert "Unsupported claim text" in html
    assert 'class="claim unsupported"' in html
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Config snapshot" in html
    assert "Faithfulness" in html


def test_report_includes_baseline_diff_and_flipped_cases(tmp_path: Path) -> None:
    baseline_results = [
        make_result("case-pass", failure_category=None),
        make_result("case-flip", failure_category=None),
    ]
    current_results = [
        make_result("case-pass", failure_category=None),
        make_result("case-flip", failure_category=FailureCategory.HALLUCINATION),
    ]
    baseline_aggregate = aggregate_run(baseline_results, resamples=100, seed=7)
    current_aggregate = aggregate_run(current_results, resamples=100, seed=7)
    comparison = compare_to_baseline(
        current=current_aggregate,
        baseline=baseline_aggregate,
        current_results=current_results,
        baseline_results=baseline_results,
        baseline_run_name="baseline-run",
    )

    run = make_run_record("current-run", results=current_results)
    out = tmp_path / "report-with-baseline.html"
    built = build_html_report(
        run=run,
        results=current_results,
        aggregate=current_aggregate,
        out=out,
        baseline=comparison,
    )

    html = built.read_text(encoding="utf-8")
    assert [case.case.case_id for case in comparison.flipped_cases] == ["case-flip"]
    assert "Baseline diff" in html
    assert "baseline-run" in html
    assert "case-flip" in html
