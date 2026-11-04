"""Generate a filtered, reproducible test set from document chunks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evalora.config import EvaloraConfig
from evalora.generate.filters import (
    Candidate,
    DropReason,
    dedupe_candidates,
    filter_answerability,
)
from evalora.generate.prompts import (
    ADVERSARIAL_PROMPT,
    MULTI_HOP_PROMPT,
    PROMPT_VERSIONS,
    SINGLE_HOP_PROMPT,
    UNANSWERABLE_PROMPT,
    PromptTemplate,
)
from evalora.ingest.chunker import chunk_documents
from evalora.ingest.loaders import load_documents
from evalora.llm.base import LLMClient, LLMError, Message, Role
from evalora.llm.costs import TokenUsage, UsageTotals, aggregate_usage
from evalora.llm.json_mode import complete_json_with_usage
from evalora.schemas import Chunk, QuestionType, TestCase

QUESTION_TYPE_ORDER: tuple[QuestionType, ...] = (
    QuestionType.SINGLE_HOP,
    QuestionType.MULTI_HOP,
    QuestionType.UNANSWERABLE,
    QuestionType.ADVERSARIAL,
)

_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "not",
        "but",
        "you",
        "your",
        "its",
        "into",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "will",
        "would",
        "could",
        "should",
        "about",
        "which",
        "when",
        "where",
        "what",
        "who",
        "how",
        "why",
        "all",
        "any",
        "can",
        "may",
        "use",
        "using",
        "also",
        "more",
        "most",
        "some",
        "such",
        "only",
        "other",
        "over",
        "under",
        "between",
        "within",
        "without",
    }
)
_WORD_RE = re.compile(r"[a-z0-9]+")


class GenerationError(Exception):
    """Raised when a test set cannot be generated."""


class SingleHopDraft(BaseModel):
    question: str
    gold_answer: str
    supporting_quote: str


class MultiHopDraft(BaseModel):
    question: str
    gold_answer: str
    supporting_quotes: list[str]


class UnanswerableDraft(BaseModel):
    question: str
    rationale: str


class AdversarialDraft(BaseModel):
    question: str
    gold_answer: str
    supporting_quote: str
    near_miss_detail: str


class GenerationReport(BaseModel):
    seed: int
    documents: int
    chunks: int
    requested: dict[QuestionType, int]
    kept: dict[QuestionType, int]
    dropped: dict[DropReason, int]
    token_usage: UsageTotals
    prompt_versions: dict[str, str]


class GenerationResult(BaseModel):
    cases: list[TestCase]
    report: GenerationReport


def allocate_requests(
    cases_per_type: Mapping[QuestionType, int],
    limit: int | None,
) -> dict[QuestionType, int]:
    requested = {
        question_type: int(cases_per_type.get(question_type, 0))
        for question_type in QUESTION_TYPE_ORDER
    }
    if any(count < 0 for count in requested.values()):
        raise ValueError("case counts must be zero or greater")
    if limit is None:
        return requested
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if limit >= sum(requested.values()):
        return requested
    allocation = dict.fromkeys(QUESTION_TYPE_ORDER, 0)
    remaining = limit
    while remaining > 0:
        progressed = False
        for question_type in QUESTION_TYPE_ORDER:
            if remaining == 0:
                break
            if allocation[question_type] < requested[question_type]:
                allocation[question_type] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation


async def generate_test_set(
    config: EvaloraConfig,
    client: LLMClient,
    *,
    out: Path,
    limit: int | None = None,
) -> GenerationResult:
    documents = load_documents(config.docs.paths)
    chunks = chunk_documents(
        documents,
        chunk_size=config.docs.chunk_size,
        chunk_overlap=config.docs.chunk_overlap,
    )
    if not chunks:
        raise GenerationError("No chunks were produced from the configured documents")
    result = await generate_cases(
        chunks=chunks,
        client=client,
        cases_per_type=config.generation.cases_per_type,
        seed=config.generation.seed,
        limit=limit,
        document_count=len(documents),
    )
    _write_test_set(result.cases, out)
    _write_report(result.report, out.with_name("generation_report.json"))
    return result


async def generate_cases(
    *,
    chunks: Sequence[Chunk],
    client: LLMClient,
    cases_per_type: Mapping[QuestionType, int],
    seed: int,
    limit: int | None = None,
    document_count: int = 0,
) -> GenerationResult:
    ordered_chunks = sorted(chunks, key=lambda chunk: (chunk.doc_id, chunk.ordinal, chunk.chunk_id))
    if not ordered_chunks:
        raise GenerationError("No chunks available for generation")
    chunks_by_id = {chunk.chunk_id: chunk for chunk in ordered_chunks}
    requested = allocate_requests(cases_per_type, limit)
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    usages: list[TokenUsage] = []
    dropped: Counter[DropReason] = Counter()

    for question_type in QUESTION_TYPE_ORDER:
        wanted = requested[question_type]
        if wanted == 0:
            continue
        if question_type in (QuestionType.SINGLE_HOP, QuestionType.UNANSWERABLE):
            targets = list(ordered_chunks)
            rng.shuffle(targets)
            selected_targets = targets[:wanted]
            if len(selected_targets) < wanted:
                dropped[DropReason.INSUFFICIENT_CONTEXT] += wanted - len(selected_targets)
            for target in selected_targets:
                try:
                    if question_type == QuestionType.SINGLE_HOP:
                        candidate, used = await _single_hop(client, target)
                    else:
                        neighbors = _neighbor_chunks(ordered_chunks, target, 2)
                        candidate, used = await _unanswerable(client, target, neighbors)
                except LLMError:
                    dropped[DropReason.GENERATION_FAILED] += 1
                    continue
                candidates.append(candidate)
                usages.extend(used)
        else:
            pairs = _related_pairs(ordered_chunks)
            rng.shuffle(pairs)
            selected_pairs = pairs[:wanted]
            if len(selected_pairs) < wanted:
                dropped[DropReason.INSUFFICIENT_CONTEXT] += wanted - len(selected_pairs)
            for first, second in selected_pairs:
                try:
                    if question_type == QuestionType.MULTI_HOP:
                        candidate, used = await _multi_hop(client, first, second)
                    else:
                        candidate, used = await _adversarial(client, first, second)
                except LLMError:
                    dropped[DropReason.GENERATION_FAILED] += 1
                    continue
                candidates.append(candidate)
                usages.extend(used)

    deduped = dedupe_candidates(candidates)
    dropped.update(deduped.dropped)
    answerability = await filter_answerability(
        deduped.kept,
        client=client,
        chunks_by_id=chunks_by_id,
    )
    dropped.update(answerability.dropped)
    usages.extend(answerability.usages)

    kept_cases = [candidate.case for candidate in answerability.kept]
    kept_counts = Counter(case.type for case in kept_cases)
    report = GenerationReport(
        seed=seed,
        documents=document_count,
        chunks=len(ordered_chunks),
        requested=dict(requested),
        kept={
            question_type: kept_counts.get(question_type, 0)
            for question_type in QUESTION_TYPE_ORDER
        },
        dropped=dict(dropped),
        token_usage=aggregate_usage(usages),
        prompt_versions=dict(PROMPT_VERSIONS),
    )
    return GenerationResult(cases=kept_cases, report=report)


async def _single_hop(client: LLMClient, chunk: Chunk) -> tuple[Candidate, list[TokenUsage]]:
    messages = _prompt_messages(SINGLE_HOP_PROMPT, context=chunk.text)
    draft, responses = await complete_json_with_usage(
        client,
        messages,
        SingleHopDraft,
        temperature=0.0,
        max_tokens=512,
    )
    case = _build_case(
        question_type=QuestionType.SINGLE_HOP,
        question=draft.question,
        gold_answer=draft.gold_answer,
        gold_chunk_ids=[chunk.chunk_id],
        source_model=client.provider.model,
        metadata={
            "prompt": SINGLE_HOP_PROMPT.name,
            "prompt_version": SINGLE_HOP_PROMPT.version,
            "source_chunk_ids": [chunk.chunk_id],
            "source_doc_ids": [chunk.doc_id],
            "source_pages": _pages([chunk]),
            "supporting_quote": draft.supporting_quote,
            "generation_strategy": "single_chunk",
        },
    )
    return Candidate(case=case, context_chunk_ids=[chunk.chunk_id]), [
        response.usage for response in responses
    ]


async def _multi_hop(
    client: LLMClient,
    first: Chunk,
    second: Chunk,
) -> tuple[Candidate, list[TokenUsage]]:
    messages = _prompt_messages(
        MULTI_HOP_PROMPT,
        context_a=first.text,
        context_b=second.text,
    )
    draft, responses = await complete_json_with_usage(
        client,
        messages,
        MultiHopDraft,
        temperature=0.0,
        max_tokens=512,
    )
    gold_chunk_ids = sorted([first.chunk_id, second.chunk_id])
    case = _build_case(
        question_type=QuestionType.MULTI_HOP,
        question=draft.question,
        gold_answer=draft.gold_answer,
        gold_chunk_ids=gold_chunk_ids,
        source_model=client.provider.model,
        metadata={
            "prompt": MULTI_HOP_PROMPT.name,
            "prompt_version": MULTI_HOP_PROMPT.version,
            "source_chunk_ids": gold_chunk_ids,
            "source_doc_ids": _doc_ids([first, second]),
            "source_pages": _pages([first, second]),
            "supporting_quotes": draft.supporting_quotes,
            "generation_strategy": "related_chunks",
        },
    )
    return Candidate(case=case, context_chunk_ids=gold_chunk_ids), [
        response.usage for response in responses
    ]


async def _unanswerable(
    client: LLMClient,
    target: Chunk,
    neighbors: Sequence[Chunk],
) -> tuple[Candidate, list[TokenUsage]]:
    context_chunks = [target, *neighbors]
    messages = _prompt_messages(
        UNANSWERABLE_PROMPT,
        context=_context_block(context_chunks),
    )
    draft, responses = await complete_json_with_usage(
        client,
        messages,
        UnanswerableDraft,
        temperature=0.0,
        max_tokens=512,
    )
    case = _build_case(
        question_type=QuestionType.UNANSWERABLE,
        question=draft.question,
        gold_answer=None,
        gold_chunk_ids=[],
        source_model=client.provider.model,
        metadata={
            "prompt": UNANSWERABLE_PROMPT.name,
            "prompt_version": UNANSWERABLE_PROMPT.version,
            "source_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
            "source_doc_ids": _doc_ids(context_chunks),
            "source_pages": _pages(context_chunks),
            "rationale": draft.rationale,
            "generation_strategy": "source_plus_neighbors",
        },
    )
    return Candidate(
        case=case,
        context_chunk_ids=[chunk.chunk_id for chunk in context_chunks],
    ), [response.usage for response in responses]


async def _adversarial(
    client: LLMClient,
    target: Chunk,
    near_miss: Chunk,
) -> tuple[Candidate, list[TokenUsage]]:
    messages = _prompt_messages(
        ADVERSARIAL_PROMPT,
        target=target.text,
        near_miss=near_miss.text,
    )
    draft, responses = await complete_json_with_usage(
        client,
        messages,
        AdversarialDraft,
        temperature=0.0,
        max_tokens=512,
    )
    case = _build_case(
        question_type=QuestionType.ADVERSARIAL,
        question=draft.question,
        gold_answer=draft.gold_answer,
        gold_chunk_ids=[target.chunk_id],
        source_model=client.provider.model,
        metadata={
            "prompt": ADVERSARIAL_PROMPT.name,
            "prompt_version": ADVERSARIAL_PROMPT.version,
            "source_chunk_ids": [target.chunk_id],
            "source_doc_ids": [target.doc_id],
            "source_pages": _pages([target]),
            "supporting_quote": draft.supporting_quote,
            "near_miss_chunk_id": near_miss.chunk_id,
            "near_miss_detail": draft.near_miss_detail,
            "generation_strategy": "near_miss_pair",
        },
    )
    return Candidate(case=case, context_chunk_ids=[target.chunk_id]), [
        response.usage for response in responses
    ]


def make_case_id(
    question_type: QuestionType,
    question: str,
    gold_answer: str | None,
    gold_chunk_ids: Sequence[str],
) -> str:
    normalized_question = " ".join(question.lower().split())
    payload = "|".join(
        [
            question_type.value,
            normalized_question,
            gold_answer or "",
            ",".join(gold_chunk_ids),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _prompt_messages(prompt: PromptTemplate, **values: str) -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content=prompt.system),
        Message(role=Role.USER, content=prompt.user.format(**values)),
    ]


def _build_case(
    *,
    question_type: QuestionType,
    question: str,
    gold_answer: str | None,
    gold_chunk_ids: list[str],
    source_model: str,
    metadata: dict[str, Any],
) -> TestCase:
    return TestCase(
        case_id=make_case_id(question_type, question, gold_answer, gold_chunk_ids),
        type=question_type,
        question=question,
        gold_answer=gold_answer,
        gold_chunk_ids=gold_chunk_ids,
        source_model=source_model,
        answerability_verdict="unchecked",
        metadata=metadata,
    )


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _nearest_neighbor(chunks: Sequence[Chunk], target: Chunk) -> Chunk | None:
    same_doc = [
        chunk
        for chunk in chunks
        if chunk.doc_id == target.doc_id and chunk.chunk_id != target.chunk_id
    ]
    if same_doc:
        return min(
            same_doc,
            key=lambda chunk: (
                abs(chunk.ordinal - target.ordinal),
                chunk.ordinal,
                chunk.chunk_id,
            ),
        )
    target_keywords = _keywords(target.text)
    related = [
        chunk
        for chunk in chunks
        if chunk.chunk_id != target.chunk_id and target_keywords & _keywords(chunk.text)
    ]
    if not related:
        return None
    return min(
        related,
        key=lambda chunk: (
            -len(target_keywords & _keywords(chunk.text)),
            chunk.doc_id,
            chunk.ordinal,
            chunk.chunk_id,
        ),
    )


def _related_pairs(chunks: Sequence[Chunk]) -> list[tuple[Chunk, Chunk]]:
    pairs: dict[tuple[str, str], tuple[Chunk, Chunk]] = {}
    for chunk in chunks:
        neighbor = _nearest_neighbor(chunks, chunk)
        if neighbor is None:
            continue
        first_id, second_id = sorted((chunk.chunk_id, neighbor.chunk_id))
        key = (first_id, second_id)
        if key in pairs:
            continue
        first, second = (
            (chunk, neighbor) if chunk.chunk_id < neighbor.chunk_id else (neighbor, chunk)
        )
        pairs[key] = (first, second)
    return list(pairs.values())


def _neighbor_chunks(chunks: Sequence[Chunk], target: Chunk, count: int) -> list[Chunk]:
    selected: list[Chunk] = []
    same_doc = [
        chunk
        for chunk in chunks
        if chunk.doc_id == target.doc_id and chunk.chunk_id != target.chunk_id
    ]
    same_doc.sort(
        key=lambda chunk: (
            abs(chunk.ordinal - target.ordinal),
            chunk.ordinal,
            chunk.chunk_id,
        )
    )
    selected.extend(same_doc[:count])
    if len(selected) >= count:
        return selected
    chosen_ids = {chunk.chunk_id for chunk in selected} | {target.chunk_id}
    target_keywords = _keywords(target.text)
    related = [
        chunk
        for chunk in chunks
        if chunk.chunk_id not in chosen_ids and target_keywords & _keywords(chunk.text)
    ]
    related.sort(
        key=lambda chunk: (
            -len(target_keywords & _keywords(chunk.text)),
            chunk.doc_id,
            chunk.ordinal,
            chunk.chunk_id,
        )
    )
    selected.extend(related[: count - len(selected)])
    return selected


def _pages(chunks: Sequence[Chunk]) -> list[int]:
    return sorted({chunk.page for chunk in chunks if chunk.page is not None})


def _doc_ids(chunks: Sequence[Chunk]) -> list[str]:
    return sorted({chunk.doc_id for chunk in chunks})


def _context_block(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(
        f"[Passage {index}]\n{chunk.text}" for index, chunk in enumerate(chunks, start=1)
    )


def _write_test_set(cases: Sequence[TestCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            payload = json.dumps(case.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            handle.write(payload + "\n")


def _write_report(report: GenerationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report.model_dump_json(indent=2) + "\n")
