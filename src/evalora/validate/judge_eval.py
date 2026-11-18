"""Judge validation metrics and end-to-end validation runs."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from pydantic import BaseModel, Field

from evalora.evaluate.faithfulness import judge_faithfulness, verify_claim
from evalora.evaluate.prompts import CLAIM_VERIFICATION_PROMPT, JudgePrompt
from evalora.llm.base import LLMClient, LLMError
from evalora.llm.costs import TokenUsage, UsageTotals, aggregate_usage
from evalora.validate.labels import LabelFileError, LabelItem, LabelValue

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 42
METRIC_NAMES = ("precision", "recall", "f1", "accuracy", "kappa")


class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int


class MetricWithCI(BaseModel):
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class JudgeMetrics(BaseModel):
    n: int
    confusion: ConfusionMatrix
    precision: MetricWithCI
    recall: MetricWithCI
    f1: MetricWithCI
    accuracy: MetricWithCI
    kappa: MetricWithCI


class ValidationItemResult(BaseModel):
    item_id: str
    true_label: LabelValue
    predicted_label: LabelValue
    correct: bool


class JudgeValidationSummary(JudgeMetrics):
    model: str
    prompt_version: str
    dataset: str
    sample_seed: int | None = None
    sample_size: int | None = None
    ci_confidence: float
    bootstrap_resamples: int
    bootstrap_seed: int
    token_usage: UsageTotals = Field(default_factory=UsageTotals)
    created_at: str
    items: list[ValidationItemResult] = Field(default_factory=list)


def compute_confusion(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> ConfusionMatrix:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = fp = tn = fn = 0
    for true_value, predicted_value in zip(y_true, y_pred, strict=True):
        if true_value and predicted_value:
            tp += 1
        elif predicted_value:
            fp += 1
        elif true_value:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _kappa(y_true: Sequence[bool], y_pred: Sequence[bool]) -> float | None:
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return None
    observed = (
        sum(
            1
            for true_value, predicted_value in zip(y_true, y_pred, strict=True)
            if true_value == predicted_value
        )
        / n
    )
    true_positive_rate = sum(1 for value in y_true if value) / n
    predicted_positive_rate = sum(1 for value in y_pred if value) / n
    expected = true_positive_rate * predicted_positive_rate + (1.0 - true_positive_rate) * (
        1.0 - predicted_positive_rate
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _point_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> dict[str, float | None]:
    confusion = compute_confusion(y_true, y_pred)
    n = confusion.tp + confusion.fp + confusion.tn + confusion.fn
    precision = (
        confusion.tp / (confusion.tp + confusion.fp) if confusion.tp + confusion.fp > 0 else None
    )
    recall = (
        confusion.tp / (confusion.tp + confusion.fn) if confusion.tp + confusion.fn > 0 else None
    )
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    accuracy = (confusion.tp + confusion.tn) / n if n else None
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "kappa": _kappa(y_true, y_pred),
    }


def _metric_with_ci(
    value: float | None,
    samples: Sequence[float],
    confidence: float,
) -> MetricWithCI:
    if value is None or not samples:
        return MetricWithCI(value=value)
    alpha = (1.0 - confidence) / 2.0
    return MetricWithCI(
        value=value,
        ci_low=float(np.quantile(samples, alpha)),
        ci_high=float(np.quantile(samples, 1.0 - alpha)),
    )


def compute_metrics(
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> JudgeMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")

    point = _point_metrics(y_true, y_pred)
    n = len(y_true)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    if n:
        rng = random.Random(seed)
        for _ in range(resamples):
            indices = [rng.randrange(n) for _ in range(n)]
            resampled_true = [y_true[index] for index in indices]
            resampled_pred = [y_pred[index] for index in indices]
            values = _point_metrics(resampled_true, resampled_pred)
            for name in METRIC_NAMES:
                value = values[name]
                if value is not None:
                    samples[name].append(value)

    return JudgeMetrics(
        n=n,
        confusion=compute_confusion(y_true, y_pred),
        precision=_metric_with_ci(point["precision"], samples["precision"], confidence),
        recall=_metric_with_ci(point["recall"], samples["recall"], confidence),
        f1=_metric_with_ci(point["f1"], samples["f1"], confidence),
        accuracy=_metric_with_ci(point["accuracy"], samples["accuracy"], confidence),
        kappa=_metric_with_ci(point["kappa"], samples["kappa"], confidence),
    )


async def predict_item(
    client: LLMClient,
    item: LabelItem,
    *,
    verification_prompt: JudgePrompt = CLAIM_VERIFICATION_PROMPT,
) -> tuple[bool, list[TokenUsage]]:
    if item.label is None:
        raise LabelFileError(f"item '{item.item_id}' has no label")
    if item.claim is not None:
        verdict, usages = await verify_claim(
            client,
            item.claim,
            item.contexts,
            prompt=verification_prompt,
        )
        return not verdict.supported, usages
    if item.answer is not None:
        result = await judge_faithfulness(
            client,
            item.answer,
            item.contexts,
            verification_prompt=verification_prompt,
        )
        predicted_unsupported = any(not verdict.supported for verdict in result.claim_verdicts)
        return predicted_unsupported, result.usages
    raise LabelFileError(f"item '{item.item_id}' has neither a claim nor an answer")


async def run_validation(
    items: Sequence[LabelItem],
    client: LLMClient,
    *,
    model: str,
    prompt_version: str,
    dataset: str,
    verification_prompt: JudgePrompt = CLAIM_VERIFICATION_PROMPT,
    sample_seed: int | None = None,
    sample_size: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> JudgeValidationSummary:
    if not items:
        raise LabelFileError("No label items to validate")

    y_true: list[bool] = []
    y_pred: list[bool] = []
    results: list[ValidationItemResult] = []
    usages: list[TokenUsage] = []

    async def predict(index: int, item: LabelItem) -> tuple[int, bool, list[TokenUsage]]:
        try:
            predicted, item_usages = await predict_item(
                client,
                item,
                verification_prompt=verification_prompt,
            )
        except LLMError as exc:
            raise LLMError(f"validation failed for item '{item.item_id}': {exc}") from exc
        return index, predicted, item_usages

    predictions = await asyncio.gather(*(predict(index, item) for index, item in enumerate(items)))
    for index, predicted_unsupported, item_usages in sorted(predictions):
        item = items[index]
        if item.label is None:
            raise LabelFileError(f"item '{item.item_id}' has no label")
        true_unsupported = item.label == LabelValue.UNSUPPORTED
        y_true.append(true_unsupported)
        y_pred.append(predicted_unsupported)
        results.append(
            ValidationItemResult(
                item_id=item.item_id,
                true_label=item.label,
                predicted_label=(
                    LabelValue.UNSUPPORTED if predicted_unsupported else LabelValue.SUPPORTED
                ),
                correct=predicted_unsupported == true_unsupported,
            )
        )
        usages.extend(item_usages)

    metrics = compute_metrics(
        y_true,
        y_pred,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )
    return JudgeValidationSummary(
        n=metrics.n,
        confusion=metrics.confusion,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        accuracy=metrics.accuracy,
        kappa=metrics.kappa,
        model=model,
        prompt_version=prompt_version,
        dataset=dataset,
        sample_seed=sample_seed,
        sample_size=sample_size,
        ci_confidence=confidence,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        token_usage=aggregate_usage(usages),
        created_at=datetime.now(UTC).isoformat(),
        items=results,
    )
