from __future__ import annotations

import pytest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from evalora.evaluate.prompts import CLAIM_VERIFICATION_PROMPT
from evalora.llm.base import LLMClient
from evalora.llm.fake import FakeProvider
from evalora.schemas import RetrievedContext
from evalora.validate.judge_eval import (
    compute_confusion,
    compute_metrics,
    predict_item,
    run_validation,
)
from evalora.validate.labels import LabelFileError, LabelItem, LabelValue


def test_confusion_matrix_hand_computed() -> None:
    confusion = compute_confusion(
        [True, True, False, False],
        [True, False, True, False],
    )

    assert confusion.tp == 1
    assert confusion.fn == 1
    assert confusion.fp == 1
    assert confusion.tn == 1


def test_metrics_match_sklearn_reference() -> None:
    y_true = [True, True, False, False, True, False]
    y_pred = [True, False, True, False, True, True]

    metrics = compute_metrics(y_true, y_pred, resamples=100, seed=7)

    assert metrics.precision.value == pytest.approx(
        precision_score(y_true, y_pred, pos_label=True, zero_division=0)
    )
    assert metrics.recall.value == pytest.approx(
        recall_score(y_true, y_pred, pos_label=True, zero_division=0)
    )
    assert metrics.f1.value == pytest.approx(
        f1_score(y_true, y_pred, pos_label=True, zero_division=0)
    )
    assert metrics.accuracy.value == pytest.approx(accuracy_score(y_true, y_pred))
    assert metrics.kappa.value == pytest.approx(
        cohen_kappa_score(y_true, y_pred, labels=[False, True])
    )
    assert metrics.confusion.tp == 2
    assert metrics.confusion.fp == 2
    assert metrics.confusion.fn == 1
    assert metrics.confusion.tn == 1


def test_metrics_match_sklearn_on_asymmetric_data() -> None:
    y_true = [True, True, True, False]
    y_pred = [True, True, False, False]

    metrics = compute_metrics(y_true, y_pred, resamples=50, seed=11)

    assert metrics.precision.value == pytest.approx(1.0)
    assert metrics.recall.value == pytest.approx(2.0 / 3.0)
    assert metrics.f1.value == pytest.approx(0.8)
    assert metrics.accuracy.value == pytest.approx(0.75)
    assert metrics.kappa.value == pytest.approx(
        cohen_kappa_score(y_true, y_pred, labels=[False, True])
    )


def test_undefined_metrics_are_null() -> None:
    metrics = compute_metrics([False, False], [False, False], resamples=20, seed=7)

    assert metrics.precision.value is None
    assert metrics.recall.value is None
    assert metrics.f1.value is None
    assert metrics.accuracy.value == 1.0
    assert metrics.kappa.value is None


def test_bootstrap_is_deterministic_and_contains_point_values() -> None:
    y_true = [True, True, False, False, True, False]
    y_pred = [True, False, True, False, True, True]

    first = compute_metrics(y_true, y_pred, resamples=100, seed=7)
    second = compute_metrics(y_true, y_pred, resamples=100, seed=7)

    assert first.model_dump() == second.model_dump()
    assert first.f1.ci_low is not None
    assert first.f1.ci_high is not None
    assert first.f1.ci_low <= first.f1.value <= first.f1.ci_high


def test_invalid_settings_raise() -> None:
    with pytest.raises(ValueError):
        compute_metrics([], [], confidence=1.0)
    with pytest.raises(ValueError):
        compute_metrics([], [], resamples=0)
    with pytest.raises(ValueError):
        compute_metrics([True], [])


def _claim_item(
    item_id: str,
    claim: str,
    label: LabelValue,
) -> LabelItem:
    return LabelItem(
        item_id=item_id,
        question="When was TechMart founded?",
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland.")],
        answer="TechMart was founded in 1994.",
        claim=claim,
        label=label,
    )


async def test_run_validation_on_claim_items() -> None:
    items = [
        _claim_item("item-1", "TechMart was founded in 1994.", LabelValue.SUPPORTED),
        _claim_item("item-2", "TechMart has 500 stores.", LabelValue.UNSUPPORTED),
    ]
    provider = FakeProvider(
        [
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
            '{"supported": false, "evidence_context_index": null, "rationale": "absent"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    summary = await run_validation(
        items,
        client,
        model="fake",
        prompt_version="v1",
        dataset="manual",
        verification_prompt=CLAIM_VERIFICATION_PROMPT,
        resamples=50,
        seed=7,
    )

    assert summary.n == 2
    assert summary.confusion.tp == 1
    assert summary.confusion.tn == 1
    assert summary.precision.value == 1.0
    assert summary.recall.value == 1.0
    assert summary.f1.value == 1.0
    assert summary.accuracy.value == 1.0
    assert all(item.correct for item in summary.items)
    assert summary.token_usage.calls == 2
    assert summary.ci_confidence == 0.95
    assert summary.bootstrap_resamples == 50
    assert provider.call_count == 2


async def test_predict_item_supports_response_level_items() -> None:
    item = LabelItem(
        item_id="answer-1",
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994 in Portland.")],
        answer="TechMart was founded in 1994.",
        label=LabelValue.SUPPORTED,
    )
    provider = FakeProvider(
        [
            '{"claims": ["TechMart was founded in 1994."]}',
            '{"supported": true, "evidence_context_index": 0, "rationale": "stated"}',
        ]
    )
    client = LLMClient(provider, max_retries=0)

    predicted_unsupported, usages = await predict_item(
        client,
        item,
        verification_prompt=CLAIM_VERIFICATION_PROMPT,
    )

    assert predicted_unsupported is False
    assert len(usages) == 2


async def test_unlabelled_item_is_rejected() -> None:
    item = LabelItem(item_id="item-1", contexts=[], answer="answer")
    client = LLMClient(FakeProvider([]), max_retries=0)

    with pytest.raises(LabelFileError, match="no label"):
        await predict_item(client, item)
