"""Deterministic retrieval metrics. No LLM calls are made here."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from evalora.schemas import (
    CaseResult,
    QuestionType,
    RetrievalScores,
    RetrievedContext,
    TestCase,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


class ContextMatch(BaseModel):
    context_index: int
    gold_chunk_id: str
    score: float


class RetrievalSummary(BaseModel):
    cases: int
    answerable_cases: int
    unanswerable_cases: int
    hit_rate: float | None
    mrr: float | None
    context_recall: float | None
    unanswerable_with_retrieval: int
    pipeline_errors: int


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def text_match_score(gold_text: str, context_text: str) -> float:
    gold_tokens = _tokens(gold_text)
    context_tokens = _tokens(context_text)
    if not gold_tokens or not context_tokens:
        return 0.0

    normalized_gold = " ".join(gold_tokens)
    normalized_context = " ".join(context_tokens)
    if normalized_gold in normalized_context or normalized_context in normalized_gold:
        return 1.0

    gold_set = set(gold_tokens)
    context_set = set(context_tokens)
    return len(gold_set & context_set) / len(gold_set)


def match_contexts(
    contexts: Sequence[RetrievedContext],
    gold_chunk_ids: Sequence[str],
    gold_chunk_texts: Mapping[str, str],
    *,
    threshold: float,
) -> list[ContextMatch]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    gold_ids = list(dict.fromkeys(gold_chunk_ids))
    gold_id_set = set(gold_ids)
    matches: list[ContextMatch] = []

    for context_index, context in enumerate(contexts):
        if context.id is not None:
            if context.id in gold_id_set:
                matches.append(
                    ContextMatch(
                        context_index=context_index,
                        gold_chunk_id=context.id,
                        score=1.0,
                    )
                )
            continue

        best_id: str | None = None
        best_score = 0.0
        for gold_id in gold_ids:
            gold_text = gold_chunk_texts.get(gold_id)
            if gold_text is None:
                continue
            score = text_match_score(gold_text, context.text)
            if score > best_score:
                best_id = gold_id
                best_score = score
        if best_id is not None and best_score >= threshold:
            matches.append(
                ContextMatch(
                    context_index=context_index,
                    gold_chunk_id=best_id,
                    score=best_score,
                )
            )
    return matches


def compute_retrieval(
    case: TestCase,
    contexts: Sequence[RetrievedContext],
    gold_chunk_texts: Mapping[str, str],
    *,
    threshold: float,
) -> RetrievalScores:
    gold_ids = list(dict.fromkeys(case.gold_chunk_ids))
    if case.type == QuestionType.UNANSWERABLE or not gold_ids:
        return RetrievalScores(
            applicable=False,
            hit=False,
            reciprocal_rank=0.0,
            context_recall=0.0,
        )

    matches = match_contexts(
        contexts,
        gold_ids,
        gold_chunk_texts,
        threshold=threshold,
    )
    matched_ids = {match.gold_chunk_id for match in matches}
    first_index = min((match.context_index for match in matches), default=None)
    return RetrievalScores(
        applicable=True,
        hit=bool(matches),
        reciprocal_rank=0.0 if first_index is None else 1.0 / (first_index + 1),
        context_recall=len(matched_ids) / len(gold_ids),
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_retrieval(results: Sequence[CaseResult]) -> RetrievalSummary:
    applicable = [result for result in results if result.retrieval.applicable]
    unanswerable = [result for result in results if result.case.type == QuestionType.UNANSWERABLE]
    return RetrievalSummary(
        cases=len(results),
        answerable_cases=len(applicable),
        unanswerable_cases=len(unanswerable),
        hit_rate=_mean([1.0 if result.retrieval.hit else 0.0 for result in applicable]),
        mrr=_mean([result.retrieval.reciprocal_rank for result in applicable]),
        context_recall=_mean([result.retrieval.context_recall for result in applicable]),
        unanswerable_with_retrieval=sum(1 for result in unanswerable if result.response.contexts),
        pipeline_errors=sum(1 for result in results if result.response.error is not None),
    )
