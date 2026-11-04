from __future__ import annotations

from collections.abc import Sequence

import pytest

from evalora.generate.filters import (
    Candidate,
    DropReason,
    dedupe_candidates,
    expected_answerable,
    filter_answerability,
    normalize_text,
)
from evalora.llm.base import LLMClient, RateLimitError
from evalora.llm.fake import FakeProvider
from evalora.schemas import Chunk, QuestionType
from evalora.schemas import TestCase as EvalTestCase


def _candidate(
    question: str,
    *,
    question_type: QuestionType = QuestionType.SINGLE_HOP,
    context_ids: Sequence[str] = ("c1",),
) -> Candidate:
    ids = list(context_ids)
    unanswerable = question_type == QuestionType.UNANSWERABLE
    case = EvalTestCase(
        case_id="case",
        type=question_type,
        question=question,
        gold_answer=None if unanswerable else "answer",
        gold_chunk_ids=[] if unanswerable else ids,
        source_model="fake",
        answerability_verdict="unchecked",
        metadata={},
    )
    return Candidate(case=case, context_chunk_ids=ids)


def _chunk(chunk_id: str = "c1", text: str = "The sky is blue.") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc", ordinal=0, text=text)


def test_normalize_text_ignores_case_and_punctuation() -> None:
    assert normalize_text("What IS the sky?") == "what is the sky"
    assert normalize_text("what   is the sky") == "what is the sky"


def test_exact_duplicates_are_dropped() -> None:
    result = dedupe_candidates([_candidate("What is the sky?"), _candidate("what is   the sky?")])

    assert len(result.kept) == 1
    assert result.dropped == {DropReason.EXACT_DUPLICATE: 1}


def test_near_duplicates_are_dropped() -> None:
    result = dedupe_candidates(
        [
            _candidate("What is the capital of France?"),
            _candidate("What is the capital city of France?"),
        ],
        threshold=0.8,
    )

    assert len(result.kept) == 1
    assert result.dropped == {DropReason.NEAR_DUPLICATE: 1}


def test_distinct_questions_are_kept() -> None:
    result = dedupe_candidates(
        [_candidate("Who founded Alpha?"), _candidate("When did Alpha go public?")]
    )

    assert len(result.kept) == 2
    assert result.dropped == {}


def test_dedupe_threshold_is_validated() -> None:
    with pytest.raises(ValueError):
        dedupe_candidates([], threshold=1.5)


def test_expected_answerable() -> None:
    assert expected_answerable(QuestionType.SINGLE_HOP) is True
    assert expected_answerable(QuestionType.MULTI_HOP) is True
    assert expected_answerable(QuestionType.ADVERSARIAL) is True
    assert expected_answerable(QuestionType.UNANSWERABLE) is False


async def test_answerable_case_passes_and_records_verdict() -> None:
    provider = FakeProvider(['{"answerable": true, "reason": "stated in context"}'])
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [_candidate("What color is the sky?")],
        client=client,
        chunks_by_id={"c1": _chunk()},
    )

    assert len(result.kept) == 1
    assert result.kept[0].case.answerability_verdict == "answerable"
    assert result.kept[0].case.metadata["answerability_reason"] == "stated in context"
    assert len(result.usages) == 1
    assert result.dropped == {}


async def test_unanswerable_case_passes_when_judge_says_unanswerable() -> None:
    provider = FakeProvider(['{"answerable": false, "reason": "not in context"}'])
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [
            _candidate(
                "What is the population of Mars?",
                question_type=QuestionType.UNANSWERABLE,
            )
        ],
        client=client,
        chunks_by_id={"c1": _chunk()},
    )

    assert len(result.kept) == 1
    assert result.kept[0].case.answerability_verdict == "unanswerable"
    assert result.dropped == {}


async def test_answerability_disagreement_is_dropped() -> None:
    provider = FakeProvider(['{"answerable": false, "reason": "not in context"}'])
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [_candidate("What color is the sky?")],
        client=client,
        chunks_by_id={"c1": _chunk()},
    )

    assert result.kept == []
    assert result.dropped == {DropReason.ANSWERABILITY_DISAGREEMENT: 1}


async def test_unanswerable_disagreement_is_dropped() -> None:
    provider = FakeProvider(['{"answerable": true, "reason": "apparently answerable"}'])
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [
            _candidate(
                "What is the population of Mars?",
                question_type=QuestionType.UNANSWERABLE,
            )
        ],
        client=client,
        chunks_by_id={"c1": _chunk()},
    )

    assert result.kept == []
    assert result.dropped == {DropReason.ANSWERABILITY_DISAGREEMENT: 1}


async def test_missing_context_is_dropped_without_an_llm_call() -> None:
    provider = FakeProvider(default='{"answerable": true, "reason": "unused"}')
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [_candidate("What color is the sky?")],
        client=client,
        chunks_by_id={},
    )

    assert result.kept == []
    assert result.dropped == {DropReason.INSUFFICIENT_CONTEXT: 1}
    assert provider.call_count == 0


async def test_llm_failure_is_dropped_as_generation_failed() -> None:
    provider = FakeProvider([RateLimitError("429")])
    client = LLMClient(provider, max_retries=0)

    result = await filter_answerability(
        [_candidate("What color is the sky?")],
        client=client,
        chunks_by_id={"c1": _chunk()},
    )

    assert result.kept == []
    assert result.dropped == {DropReason.GENERATION_FAILED: 1}
