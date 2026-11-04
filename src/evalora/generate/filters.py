"""Dedupe and answerability filters for generated test cases."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from evalora.generate.prompts import ANSWERABILITY_PROMPT
from evalora.llm.base import LLMClient, LLMError, Message, Role
from evalora.llm.costs import TokenUsage
from evalora.llm.json_mode import complete_json_with_usage
from evalora.schemas import Chunk, QuestionType, TestCase

DEDUPE_THRESHOLD = 0.9
_WORD_RE = re.compile(r"[a-z0-9]+")


class DropReason(StrEnum):
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    ANSWERABILITY_DISAGREEMENT = "answerability_disagreement"
    GENERATION_FAILED = "generation_failed"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class Candidate(BaseModel):
    case: TestCase
    context_chunk_ids: list[str]


class AnswerabilityVerdict(BaseModel):
    answerable: bool
    reason: str


class DedupeResult(BaseModel):
    kept: list[Candidate]
    dropped: dict[DropReason, int] = Field(default_factory=dict)


class AnswerabilityFilterResult(BaseModel):
    kept: list[Candidate]
    dropped: dict[DropReason, int] = Field(default_factory=dict)
    usages: list[TokenUsage] = Field(default_factory=list)


def normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def dedupe_candidates(
    candidates: Sequence[Candidate],
    *,
    threshold: float = DEDUPE_THRESHOLD,
) -> DedupeResult:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    kept: list[Candidate] = []
    seen: list[tuple[str, set[str]]] = []
    dropped: dict[DropReason, int] = {}
    for candidate in candidates:
        normalized = normalize_text(candidate.case.question)
        tokens = set(normalized.split())
        reason: DropReason | None = None
        for previous_text, previous_tokens in seen:
            if normalized == previous_text:
                reason = DropReason.EXACT_DUPLICATE
                break
            if _token_jaccard(tokens, previous_tokens) >= threshold:
                reason = DropReason.NEAR_DUPLICATE
                break
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        kept.append(candidate)
        seen.append((normalized, tokens))
    return DedupeResult(kept=kept, dropped=dropped)


def expected_answerable(question_type: QuestionType) -> bool:
    return question_type != QuestionType.UNANSWERABLE


def _increment(dropped: dict[DropReason, int], reason: DropReason) -> None:
    dropped[reason] = dropped.get(reason, 0) + 1


async def filter_answerability(
    candidates: Sequence[Candidate],
    *,
    client: LLMClient,
    chunks_by_id: Mapping[str, Chunk],
) -> AnswerabilityFilterResult:
    kept: list[Candidate] = []
    dropped: dict[DropReason, int] = {}
    usages: list[TokenUsage] = []
    for candidate in candidates:
        contexts: list[str] = []
        missing_context = False
        for chunk_id in candidate.context_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                missing_context = True
                break
            contexts.append(chunk.text)
        if missing_context:
            _increment(dropped, DropReason.INSUFFICIENT_CONTEXT)
            continue
        numbered_context = "\n\n".join(
            f"[{index}] {text}" for index, text in enumerate(contexts, start=1)
        )
        messages = [
            Message(role=Role.SYSTEM, content=ANSWERABILITY_PROMPT.system),
            Message(
                role=Role.USER,
                content=ANSWERABILITY_PROMPT.user.format(
                    question=candidate.case.question,
                    context=numbered_context,
                ),
            ),
        ]
        try:
            verdict, responses = await complete_json_with_usage(
                client,
                messages,
                AnswerabilityVerdict,
                temperature=0.0,
                max_tokens=256,
            )
        except LLMError:
            _increment(dropped, DropReason.GENERATION_FAILED)
            continue
        usages.extend(response.usage for response in responses)
        expected = expected_answerable(candidate.case.type)
        if verdict.answerable != expected:
            _increment(dropped, DropReason.ANSWERABILITY_DISAGREEMENT)
            continue
        metadata = dict(candidate.case.metadata)
        metadata["answerability_reason"] = verdict.reason
        label = "answerable" if expected else "unanswerable"
        updated_case = candidate.case.model_copy(
            update={"answerability_verdict": label, "metadata": metadata}
        )
        kept.append(candidate.model_copy(update={"case": updated_case}))
    return AnswerabilityFilterResult(kept=kept, dropped=dropped, usages=usages)
