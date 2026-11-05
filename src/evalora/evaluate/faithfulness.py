"""Faithfulness judging: claim extraction and per-claim verification."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from evalora.evaluate.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT
from evalora.llm.base import LLMClient, Message, Role
from evalora.llm.costs import TokenUsage
from evalora.llm.json_mode import complete_json_with_usage
from evalora.schemas import ClaimVerdict, RetrievedContext


class ClaimExtraction(BaseModel):
    claims: list[str]


class ClaimVerification(BaseModel):
    supported: bool
    evidence_context_index: int | None = None
    rationale: str


class FaithfulnessResult(BaseModel):
    faithfulness: float | None
    claim_verdicts: list[ClaimVerdict]
    usages: list[TokenUsage] = Field(default_factory=list)


def _context_block(contexts: Sequence[RetrievedContext]) -> str:
    return "\n\n".join(f"[{index}] {context.text}" for index, context in enumerate(contexts))


async def extract_claims(client: LLMClient, answer: str) -> tuple[list[str], list[TokenUsage]]:
    messages = [
        Message(role=Role.SYSTEM, content=CLAIM_EXTRACTION_PROMPT.system),
        Message(
            role=Role.USER,
            content=CLAIM_EXTRACTION_PROMPT.user.format(answer=answer),
        ),
    ]
    extraction, responses = await complete_json_with_usage(
        client,
        messages,
        ClaimExtraction,
        temperature=0.0,
        max_tokens=512,
    )
    claims = [claim.strip() for claim in extraction.claims if claim.strip()]
    return claims, [response.usage for response in responses if not response.cached]


async def verify_claim(
    client: LLMClient,
    claim: str,
    contexts: Sequence[RetrievedContext],
) -> tuple[ClaimVerdict, list[TokenUsage]]:
    messages = [
        Message(role=Role.SYSTEM, content=CLAIM_VERIFICATION_PROMPT.system),
        Message(
            role=Role.USER,
            content=CLAIM_VERIFICATION_PROMPT.user.format(
                claim=claim,
                contexts=_context_block(contexts),
            ),
        ),
    ]
    verification, responses = await complete_json_with_usage(
        client,
        messages,
        ClaimVerification,
        temperature=0.0,
        max_tokens=256,
    )
    evidence = verification.evidence_context_index
    if evidence is not None and not (0 <= evidence < len(contexts)):
        evidence = None
    verdict = ClaimVerdict(
        claim=claim,
        supported=verification.supported,
        evidence_context_index=evidence,
        rationale=verification.rationale,
    )
    return verdict, [response.usage for response in responses if not response.cached]


async def judge_faithfulness(
    client: LLMClient,
    answer: str,
    contexts: Sequence[RetrievedContext],
) -> FaithfulnessResult:
    if not answer.strip():
        return FaithfulnessResult(faithfulness=None, claim_verdicts=[])

    claims, usages = await extract_claims(client, answer)
    if not claims:
        return FaithfulnessResult(
            faithfulness=None,
            claim_verdicts=[],
            usages=usages,
        )

    verdicts: list[ClaimVerdict] = []
    for claim in claims:
        verdict, verification_usages = await verify_claim(client, claim, contexts)
        verdicts.append(verdict)
        usages.extend(verification_usages)

    supported = sum(1 for verdict in verdicts if verdict.supported)
    return FaithfulnessResult(
        faithfulness=supported / len(verdicts),
        claim_verdicts=verdicts,
        usages=usages,
    )
