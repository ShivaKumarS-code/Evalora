"""Domain models shared across ingestion, running, evaluation, and reporting."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    page: int | None = None


class QuestionType(StrEnum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"
    ADVERSARIAL = "adversarial"


class TestCase(BaseModel):
    case_id: str
    type: QuestionType
    question: str
    gold_answer: str | None
    gold_chunk_ids: list[str]
    source_model: str
    answerability_verdict: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(BaseModel):
    id: str | None = None
    text: str


class PipelineResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext]
    latency_ms: float
    error: str | None = None


class RetrievalScores(BaseModel):
    hit: bool
    reciprocal_rank: float
    context_recall: float


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    evidence_context_index: int | None = None
    rationale: str


class FailureCategory(StrEnum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    GENERATION_FAILURE = "generation_failure"
    HALLUCINATION = "hallucination"
    FAILED_ABSTENTION = "failed_abstention"
    WRONG_ABSTENTION = "wrong_abstention"
    PIPELINE_ERROR = "pipeline_error"


class CaseResult(BaseModel):
    case: TestCase
    response: PipelineResponse
    retrieval: RetrievalScores
    faithfulness: float | None
    claim_verdicts: list[ClaimVerdict]
    correct: bool | None
    abstained: bool
    failure_category: FailureCategory | None
