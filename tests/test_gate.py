"""Unit tests for the gate comparison, markdown summary, and baseline files."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalora.config import ThresholdConfig
from evalora.gate.baseline import (
    BASELINE_FORMAT_VERSION,
    BaselineFileError,
    export_baseline,
    import_baseline,
    read_baseline_file,
    write_baseline_file,
)
from evalora.gate.compare import GateError, compare_runs
from evalora.gate.markdown import render_markdown
from evalora.schemas import FailureCategory, QuestionType
from evalora.store.db import RunStore, StoreError
from tests.factories import make_result


def _thresholds(**overrides: float) -> dict[str, ThresholdConfig]:
    return {"faithfulness": ThresholdConfig(max_drop=overrides.get("faithfulness", 0.05))}


def _healthy(case_id: str, faithfulness: float = 1.0) -> object:
    return make_result(
        case_id,
        QuestionType.SINGLE_HOP,
        faithfulness=faithfulness,
        correct=True,
        hit=True,
        reciprocal_rank=1.0,
        context_recall=1.0,
    )


def _degraded(case_id: str, faithfulness: float, category: FailureCategory | None) -> object:
    return make_result(
        case_id,
        QuestionType.SINGLE_HOP,
        faithfulness=faithfulness,
        correct=faithfulness >= 0.99,
        hit=faithfulness >= 0.99,
        reciprocal_rank=1.0 if faithfulness >= 0.99 else 0.0,
        context_recall=1.0 if faithfulness >= 0.99 else 0.0,
        failure_category=category,
    )


def _compare(
    current: list[object],
    baseline: list[object],
    *,
    thresholds: dict[str, ThresholdConfig] | None = None,
    resamples: int = 200,
    seed: int = 7,
) -> object:
    return compare_runs(
        current,  # type: ignore[arg-type]
        baseline,  # type: ignore[arg-type]
        run_name="current",
        baseline_run_name="baseline",
        thresholds=thresholds if thresholds is not None else _thresholds(),
        confidence=0.95,
        resamples=resamples,
        seed=seed,
    )


def test_identical_runs_pass() -> None:
    baseline = [_healthy(f"case-{index}") for index in range(4)]
    report = _compare(list(baseline), list(baseline))

    assert report.passed is True
    assert report.paired_cases == 4
    assert report.new_failures_by_category == {}
    assert report.regressed_cases == []
    for metric in report.metrics:
        if metric.paired_cases:
            assert metric.diff == pytest.approx(0.0)
            assert metric.regressed is False


def test_controlled_drop_beyond_threshold_fails() -> None:
    baseline = [_healthy(f"case-{index}") for index in range(8)]
    current = [_degraded(f"case-{index}", 0.5, None) for index in range(8)]

    report = _compare(current, baseline)

    assert report.passed is False
    faithfulness = next(m for m in report.metrics if m.metric == "faithfulness")
    assert faithfulness.regressed is True
    assert faithfulness.diff == pytest.approx(-0.5)
    assert faithfulness.ci_high is not None and faithfulness.ci_high < 0.0
    assert any("exceeds max_drop" in reason for reason in faithfulness.reasons)


def test_drop_inside_the_noise_band_passes() -> None:
    # Mean drop 0.05 exceeds max_drop 0.01, but the paired bootstrap CI
    # straddles zero, so the gate must not fail on noise.
    baseline = [_healthy("case-a", faithfulness=1.0), _healthy("case-b", faithfulness=0.1)]
    current = [_degraded("case-a", 0.0, None), _healthy("case-b", faithfulness=1.0)]

    report = _compare(
        current, baseline, thresholds={"faithfulness": ThresholdConfig(max_drop=0.01)}
    )

    assert report.passed is True
    faithfulness = next(m for m in report.metrics if m.metric == "faithfulness")
    assert faithfulness.regressed is False
    assert faithfulness.diff == pytest.approx(-0.05)
    assert faithfulness.ci_low is not None and faithfulness.ci_low < 0.0
    assert faithfulness.ci_high is not None and faithfulness.ci_high > 0.0


def test_absolute_floor_fails_directly() -> None:
    baseline = [_healthy(f"case-{index}") for index in range(4)]
    current = [_healthy(f"case-{index}", faithfulness=0.4) for index in range(4)]

    report = _compare(
        current,
        baseline,
        thresholds={"faithfulness": ThresholdConfig(max_drop=0.05, min_absolute=0.5)},
    )

    assert report.passed is False
    faithfulness = next(m for m in report.metrics if m.metric == "faithfulness")
    assert faithfulness.regressed is True
    assert any("absolute floor" in reason for reason in faithfulness.reasons)


def test_no_shared_cases_compares_unpaired_samples() -> None:
    baseline = [_healthy(f"base-{index}") for index in range(3)]
    current = [_healthy(f"cur-{index}", faithfulness=0.4) for index in range(3)]

    report = _compare(current, baseline)

    assert report.passed is False
    assert report.paired_cases == 0
    assert report.new_case_count == 3
    assert report.missing_case_count == 3
    faithfulness = next(m for m in report.metrics if m.metric == "faithfulness")
    assert faithfulness.diff == pytest.approx(-0.6)
    assert any("unpaired" in reason for reason in faithfulness.reasons)


def test_new_failures_are_counted_by_category() -> None:
    baseline = [_healthy("case-a"), _healthy("case-b"), _healthy("case-c")]
    current = [
        _degraded("case-a", 0.0, FailureCategory.HALLUCINATION),
        _degraded("case-b", 0.0, FailureCategory.RETRIEVAL_FAILURE),
        _degraded("case-c", 0.0, FailureCategory.HALLUCINATION),  # counted again
    ]

    report = _compare(current, baseline)

    assert report.new_failures_by_category == {
        FailureCategory.HALLUCINATION: 2,
        FailureCategory.RETRIEVAL_FAILURE: 1,
    }


def test_recovered_or_stable_failures_are_not_new_failures() -> None:
    baseline = [
        _degraded("case-a", 0.0, FailureCategory.HALLUCINATION),
        _degraded("case-b", 0.0, FailureCategory.RETRIEVAL_FAILURE),
    ]
    current = [
        _healthy("case-a"),
        _degraded("case-b", 0.0, FailureCategory.RETRIEVAL_FAILURE),
    ]

    report = _compare(current, baseline)

    assert report.new_failures_by_category == {}


def test_regressed_cases_rank_new_failures_first_and_cap_the_list() -> None:
    baseline = [_healthy(f"case-{index}", faithfulness=0.9) for index in range(7)]
    current = [
        _degraded("case-0", 0.8, FailureCategory.HALLUCINATION),  # new failure
        _degraded("case-1", 0.8, FailureCategory.WRONG_ABSTENTION),  # new failure
        _healthy("case-2", faithfulness=0.5),  # drop, no failure
        _healthy("case-3", faithfulness=0.6),  # drop, no failure
        _healthy("case-4", faithfulness=0.7),  # drop, no failure
        _healthy("case-5", faithfulness=0.85),  # drop, no failure
        _healthy("case-6", faithfulness=0.9),  # unchanged, excluded
    ]

    report = _compare(current, baseline)

    assert len(report.regressed_cases) == 5
    assert [case.case_id for case in report.regressed_cases[:2]] == ["case-0", "case-1"]
    assert all(case.newly_failing for case in report.regressed_cases[:2])
    drops = [
        case.faithfulness_drop
        for case in report.regressed_cases[2:]
        if case.faithfulness_drop is not None
    ]
    assert drops == sorted(drops, reverse=True)
    assert report.regressed_cases[-1].case_id == "case-4"


def test_unknown_metric_raises() -> None:
    with pytest.raises(GateError, match="unknown gate metric"):
        _compare(
            [_healthy("case-1")],
            [_healthy("case-1")],
            thresholds={"latency": ThresholdConfig(max_drop=0.1)},
        )


def test_empty_thresholds_raise() -> None:
    with pytest.raises(GateError, match="no metrics configured"):
        _compare([_healthy("case-1")], [_healthy("case-1")], thresholds={})


def test_current_run_without_any_data_raises() -> None:
    baseline = [_healthy("case-1")]
    current = [make_result("case-1", QuestionType.UNANSWERABLE, applicable=False)]

    with pytest.raises(GateError, match="no data for any gated metric"):
        _compare(current, baseline)


def test_metric_without_data_is_skipped_not_failed() -> None:
    baseline = [
        make_result("case-1", QuestionType.SINGLE_HOP, correct=True, hit=True),
        make_result("case-2", QuestionType.SINGLE_HOP, correct=True, hit=True),
    ]
    # Faithfulness absent on both sides; correctness identical, so the gate
    # must skip faithfulness with a note instead of failing.
    current = [
        make_result("case-1", QuestionType.SINGLE_HOP, correct=True, hit=True),
        make_result("case-2", QuestionType.SINGLE_HOP, correct=True, hit=True),
    ]

    report = _compare(
        current,
        baseline,
        thresholds={
            "faithfulness": ThresholdConfig(max_drop=0.05),
            "correctness": ThresholdConfig(max_drop=0.05),
        },
    )

    assert report.passed is True
    faithfulness = next(m for m in report.metrics if m.metric == "faithfulness")
    assert faithfulness.paired_cases == 0
    assert faithfulness.regressed is False
    assert any("not enough data" in reason for reason in faithfulness.reasons)


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    baseline = [_healthy(f"case-{index}") for index in range(6)]
    current = [_degraded(f"case-{index}", 0.7, None) for index in range(6)]

    first = _compare(list(current), list(baseline), seed=11)
    second = _compare(list(current), list(baseline), seed=11)
    third = _compare(list(current), list(baseline), seed=12)

    assert first.model_dump() == second.model_dump()
    assert third.model_dump() != first.model_dump()


def test_markdown_pass_report() -> None:
    baseline = [_healthy("case-1")]
    report = _compare([_healthy("case-1")], baseline)
    markdown = render_markdown(report)

    assert "## Evalora quality gate: ✅ passed" in markdown
    assert "| Metric | Baseline | Current | Δ | Verdict |" in markdown
    assert "✅ pass" in markdown
    assert "New failures by category" not in markdown


def test_markdown_fail_report_includes_failures_and_cases() -> None:
    baseline = [_healthy("case-a", faithfulness=0.9), _healthy("case-b")]
    current = [
        _degraded("case-a", 0.0, FailureCategory.HALLUCINATION),
        _healthy("case-b", faithfulness=0.5),
    ]
    report = _compare(current, baseline)
    assert report.passed is False
    markdown = render_markdown(report)

    assert "## Evalora quality gate: ❌ failed" in markdown
    assert "❌ fail" in markdown
    assert "### New failures by category" in markdown
    assert "`hallucination`" in markdown
    assert "### Top regressed cases" in markdown
    assert "`case-a`" in markdown
    assert "↓" in markdown


def test_baseline_export_import_roundtrip(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.get_or_create_run(
        "baseline-run",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="hash-baseline",
        case_count=2,
    )
    store.save_result("baseline-run", make_result("case-1", QuestionType.SINGLE_HOP))
    store.save_result("baseline-run", make_result("case-2", QuestionType.SINGLE_HOP))

    export = export_baseline(store, "baseline-run")
    path = write_baseline_file(export, tmp_path / "baseline.json")

    loaded = read_baseline_file(path)
    assert loaded.format_version == BASELINE_FORMAT_VERSION
    assert loaded.run_name == "baseline-run"
    assert [result.case.case_id for result in loaded.results] == ["case-1", "case-2"]


def test_baseline_import_into_fresh_store_registers_alias(tmp_path: Path) -> None:
    source = RunStore(tmp_path / "source.db")
    source.get_or_create_run(
        "baseline-run",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="hash-baseline",
        case_count=1,
    )
    result = make_result("case-1", QuestionType.SINGLE_HOP)
    source.save_result("baseline-run", result)
    export = export_baseline(source, "baseline-run")

    fresh = RunStore(tmp_path / "fresh.db")
    record = import_baseline(fresh, export, baseline_name="main")

    assert record.run_name == "baseline-run"
    assert fresh.get_baseline("main") == "baseline-run"
    loaded = fresh.load_results("baseline-run")
    assert [item.case.case_id for item in loaded] == ["case-1"]
    assert loaded[0].faithfulness == result.faithfulness

    # Importing twice is idempotent.
    import_baseline(fresh, export, baseline_name="main")
    assert len(fresh.load_results("baseline-run")) == 1


def test_baseline_import_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = RunStore(tmp_path / "source.db")
    source.get_or_create_run(
        "baseline-run",
        config_snapshot={},
        test_set_hash="hash-a",
        case_count=1,
    )
    export = export_baseline(source, "baseline-run")

    clash = RunStore(tmp_path / "clash.db")
    clash.get_or_create_run(
        "baseline-run",
        config_snapshot={},
        test_set_hash="hash-b",
        case_count=1,
    )

    with pytest.raises(StoreError, match="different test set"):
        import_baseline(clash, export, baseline_name="main")


def test_baseline_export_missing_run_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")

    with pytest.raises(StoreError, match="not found"):
        export_baseline(store, "nope")


def test_baseline_file_validation(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(BaselineFileError, match="not found"):
        read_baseline_file(missing)

    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(BaselineFileError, match="Invalid baseline file"):
        read_baseline_file(path)
