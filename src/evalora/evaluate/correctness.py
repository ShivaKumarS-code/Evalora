"""Correctness judging against gold answers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from evalora.evaluate.prompts import CORRECTNESS_PROMPT
from evalora.llm.base import LLMClient, Message, Role
from evalora.llm.costs import TokenUsage
from evalora.llm.json_mode import complete_json_with_usage
from evalora.schemas import CorrectnessLabel


class CorrectnessVerdict(BaseModel):
    label: CorrectnessLabel
    reason: str


class CorrectnessResult(BaseModel):
    label: CorrectnessLabel
    correct: bool
    reason: str
    usages: list[TokenUsage] = Field(default_factory=list)


async def judge_correctness(
    client: LLMClient,
    *,
    question: str,
    gold_answer: str,
    answer: str,
) -> CorrectnessResult:
    messages = [
        Message(role=Role.SYSTEM, content=CORRECTNESS_PROMPT.system),
        Message(
            role=Role.USER,
            content=CORRECTNESS_PROMPT.user.format(
                question=question,
                gold_answer=gold_answer,
                answer=answer,
            ),
        ),
    ]
    verdict, responses = await complete_json_with_usage(
        client,
        messages,
        CorrectnessVerdict,
        temperature=0.0,
        max_tokens=256,
    )
    return CorrectnessResult(
        label=verdict.label,
        correct=verdict.label == CorrectnessLabel.CORRECT,
        reason=verdict.reason,
        usages=[response.usage for response in responses if not response.cached],
    )
