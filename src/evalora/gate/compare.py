"""Baseline comparison gate with paired bootstrap confidence intervals.

The gate is deterministic and never calls an LLM. It compares the current run
to a baseline run per configured metric, pairs results by ``case_id`` where
possible, and fails only when the drop exceeds the configured threshold and
the bootstrap confidence interval of the paired difference excludes zero.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from evalora.config import ThresholdConfig
from evalora.schemas import CaseResult, FailureCategory

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_TOP_REGRESSED = 5


class GateError(Exception):
    """Raised when a gate comparison cannot be built."""


class MetricName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_RECALL = "context_recall"
    CORRECTNESS = "correctness"


def faithfulness_value(result: CaseResult) -> float | None:
    return result.faithfulness


def correctness_value(result: CaseResult) -> float | None:
    if result.correct is None:
        return None
    return 1.0 if result.correct else 0.0


def context_recall_value(result: CaseResult) -> float | None:
    if not result.retrieval.applicable:
        return None
    return result.retrieval.context_recall


MetricExtractor = Callable[[CaseResult], float | None]

_EXTRACTORS: dict[str, MetricExtractor] = {
    MetricName.FAITHFULNESS: faithfulness_value,
    MetricName.CONTEXT_RECALL: context_recall_value,
    MetricName.CORRECTNESS: correctness_value,
}


class MetricComparison(BaseModel):
    metric: str
    baseline: float | None = None
    current: float | None = None
    diff: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    paired_cases: int = 0
    max_drop: float | None = None
    min_absolute: float | None = None
    regressed: bool = False
    reasons: list[str] = Field(default_factory=list)


class RegressedCase(BaseModel):
    case_id: str
    question_type: str
    question: str
    baseline_category: FailureCategory | None
    current_category: FailureCategory | None
    baseline_faithfulness: float | None
    current_faithfulness: float | None
    faithfulness_drop: float | None
    newly_failing: bool


class GateReport(BaseModel):
    run_name: str
    baseline_run_name: str
    passed: bool
    metrics: list[MetricComparison] = Field(default_factory=list)
    new_failures_by_category: dict[FailureCategory, int] = Field(default_factory=dict)
    regressed_cases: list[RegressedCase] = Field(default_factory=list)
    paired_cases: int = 0
    new_case_count: int = 0
    missing_case_count: int = 0
    ci_confidence: float
    bootstrap_resamples: int
    seed: int


def _validate_settings(
    thresholds: Mapping[str, ThresholdConfig],
    confidence: float,
    resamples: int,
) -> None:
    if not thresholds:
        raise GateError("no metrics configured for the gate")
    unknown = [name for name in thresholds if name not in _EXTRACTORS]
    if unknown:
        known = ", ".join(sorted(_EXTRACTORS))
        raise GateError(f"unknown gate metric(s): {', '.join(unknown)}; supported metrics: {known}")
    if not 0.0 < confidence < 1.0:
        raise GateError("ci_confidence must be between 0 and 1")
    if resamples < 1:
        raise GateError("resamples must be at least 1")


def _paired_diff_ci(
    diffs: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Mean paired difference and its bootstrap confidence interval."""
    if len(diffs) == 1:
        value = diffs[0]
        return value, value, value
    array = np.asarray(diffs, dtype=float)
    mean = float(array.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    bootstrap_means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        mean,
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha)),
    )


def _independent_diff_ci(
    baseline_values: Sequence[float],
    current_values: Sequence[float],
    *,
    confidence: float,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Difference of means with an independent two-sample bootstrap.

    Used only when no case ids overlap, so pairing is impossible.
    """
    baseline_array = np.asarray(baseline_values, dtype=float)
    current_array = np.asarray(current_values, dtype=float)
    mean_diff = float(current_array.mean() - baseline_array.mean())
    rng = np.random.default_rng(seed)
    baseline_indices = rng.integers(0, len(baseline_values), size=(resamples, len(baseline_values)))
    current_indices = rng.integers(0, len(current_values), size=(resamples, len(current_values)))
    baseline_means = baseline_array[baseline_indices].mean(axis=1)
    current_means = current_array[current_indices].mean(axis=1)
    diff_means = current_means - baseline_means
    alpha = (1.0 - confidence) / 2.0
    return (
        mean_diff,
        float(np.quantile(diff_means, alpha)),
        float(np.quantile(diff_means, 1.0 - alpha)),
    )


def _compare_metric(
    metric: str,
    baseline_map: Mapping[str, CaseResult],
    current_map: Mapping[str, CaseResult],
    *,
    threshold: ThresholdConfig,
    confidence: float,
    resamples: int,
    seed: int,
) -> MetricComparison:
    extractor = _EXTRACTORS[metric]
    paired_ids = sorted(set(baseline_map) & set(current_map))
    pairs: list[tuple[float, float]] = []
    for case_id in paired_ids:
        baseline_value = extractor(baseline_map[case_id])
        current_value = extractor(current_map[case_id])
        if baseline_value is None or current_value is None:
            continue
        pairs.append((baseline_value, current_value))

    comparison = MetricComparison(
        metric=metric,
        max_drop=threshold.max_drop,
        min_absolute=threshold.min_absolute,
        paired_cases=len(pairs),
    )

    baseline_only = [
        extractor(baseline_map[case_id]) for case_id in set(baseline_map) - set(current_map)
    ]
    current_only = [
        extractor(current_map[case_id]) for case_id in set(current_map) - set(baseline_map)
    ]
    if pairs:
        diffs = [current - baseline for baseline, current in pairs]
        mean, ci_low, ci_high = _paired_diff_ci(
            diffs,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
        comparison.baseline = float(np.mean([baseline for baseline, _ in pairs]))
        comparison.current = float(np.mean([current for _, current in pairs]))
        comparison.diff = mean
        comparison.ci_low = ci_low
        comparison.ci_high = ci_high
    elif baseline_only or current_only:
        baseline_values = [value for value in baseline_only if value is not None]
        current_values = [value for value in current_only if value is not None]
        if baseline_values and current_values:
            mean, ci_low, ci_high = _independent_diff_ci(
                baseline_values,
                current_values,
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
            comparison.baseline = float(np.mean(baseline_values))
            comparison.current = float(np.mean(current_values))
            comparison.diff = mean
            comparison.ci_low = ci_low
            comparison.ci_high = ci_high
            comparison.reasons.append("no shared case ids; compared unpaired samples")
        else:
            comparison.reasons.append("not enough data on both sides to compare")
    else:
        comparison.reasons.append("not enough data on both sides to compare")

    if (
        comparison.current is not None
        and threshold.min_absolute is not None
        and comparison.current < threshold.min_absolute
    ):
        comparison.regressed = True
        comparison.reasons.append(
            f"current value {comparison.current:.3f} is below the "
            f"absolute floor {threshold.min_absolute:.3f}"
        )

    if (
        comparison.diff is not None
        and comparison.ci_high is not None
        and threshold.max_drop is not None
        and comparison.diff < -threshold.max_drop
        and comparison.ci_high < 0.0
    ):
        comparison.regressed = True
        comparison.reasons.append(
            f"drop {-comparison.diff:.3f} exceeds max_drop "
            f"{threshold.max_drop:.3f} and the "
            f"{confidence:.0%} CI [{comparison.ci_low:.3f}, "
            f"{comparison.ci_high:.3f}] excludes zero"
        )

    return comparison


def _new_failures_by_category(
    baseline_map: Mapping[str, CaseResult],
    current_map: Mapping[str, CaseResult],
) -> dict[FailureCategory, int]:
    counts: dict[FailureCategory, int] = {}
    for case_id in set(baseline_map) & set(current_map):
        baseline_category = baseline_map[case_id].failure_category
        current_category = current_map[case_id].failure_category
        if baseline_category is None and current_category is not None:
            counts[current_category] = counts.get(current_category, 0) + 1
    return counts


def _regressed_cases(
    baseline_map: Mapping[str, CaseResult],
    current_map: Mapping[str, CaseResult],
    *,
    top: int,
) -> list[RegressedCase]:
    """Newly failing cases first, then by largest faithfulness drop."""
    candidates: list[RegressedCase] = []
    for case_id in set(baseline_map) & set(current_map):
        baseline = baseline_map[case_id]
        current = current_map[case_id]
        newly_failing = baseline.failure_category is None and current.failure_category is not None
        baseline_faithfulness = baseline.faithfulness
        current_faithfulness = current.faithfulness
        drop = None
        if baseline_faithfulness is not None and current_faithfulness is not None:
            drop = baseline_faithfulness - current_faithfulness
        if not newly_failing and not (drop is not None and drop > 0.0):
            continue
        candidates.append(
            RegressedCase(
                case_id=case_id,
                question_type=current.case.type.value,
                question=current.case.question,
                baseline_category=baseline.failure_category,
                current_category=current.failure_category,
                baseline_faithfulness=baseline_faithfulness,
                current_faithfulness=current_faithfulness,
                faithfulness_drop=drop,
                newly_failing=newly_failing,
            )
        )

    def sort_key(item: RegressedCase) -> tuple[int, float, str]:
        severity = 0 if item.newly_failing else 1
        drop = item.faithfulness_drop if item.faithfulness_drop is not None else 0.0
        return (severity, -drop, item.case_id)

    return sorted(candidates, key=sort_key)[:top]


def compare_runs(
    current_results: Sequence[CaseResult],
    baseline_results: Sequence[CaseResult],
    *,
    run_name: str,
    baseline_run_name: str,
    thresholds: Mapping[str, ThresholdConfig],
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    top_regressed: int = DEFAULT_TOP_REGRESSED,
) -> GateReport:
    """Compare a run against a baseline and decide whether quality regressed.

    The gate fails a metric only when the mean drop exceeds the configured
    ``max_drop`` and the bootstrap confidence interval of the paired
    difference excludes zero, so noise inside the CI does not fail the build.
    Absolute floors fail directly when the current value falls below them.
    """
    if top_regressed < 1:
        raise GateError("top_regressed must be at least 1")
    _validate_settings(thresholds, confidence, resamples)

    baseline_map = {result.case.case_id: result for result in baseline_results}
    current_map = {result.case.case_id: result for result in current_results}
    shared_ids = set(baseline_map) & set(current_map)
    new_case_count = len(set(current_map) - set(baseline_map))
    missing_case_count = len(set(baseline_map) - set(current_map))

    metrics = [
        _compare_metric(
            metric,
            baseline_map,
            current_map,
            threshold=threshold,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
        for metric, threshold in thresholds.items()
    ]

    current_has_any_data = any(comparison.current is not None for comparison in metrics)
    if not current_has_any_data:
        raise GateError(
            "the current run has no data for any gated metric (was it run with --no-judge?)"
        )

    regressed = any(comparison.regressed for comparison in metrics)

    return GateReport(
        run_name=run_name,
        baseline_run_name=baseline_run_name,
        passed=not regressed,
        metrics=metrics,
        new_failures_by_category=_new_failures_by_category(baseline_map, current_map),
        regressed_cases=_regressed_cases(baseline_map, current_map, top=top_regressed),
        paired_cases=len(shared_ids),
        new_case_count=new_case_count,
        missing_case_count=missing_case_count,
        ci_confidence=confidence,
        bootstrap_resamples=resamples,
        seed=seed,
    )


def metric_value(result: CaseResult, metric: str) -> float | None:
    """Per-case value for a gated metric, or None when not applicable."""
    extractor = _EXTRACTORS.get(metric)
    if extractor is None:
        raise GateError(f"unknown gate metric '{metric}'")
    return extractor(result)
