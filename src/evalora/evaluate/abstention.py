"""Abstention detection: deterministic patterns with an optional LLM fallback."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from evalora.evaluate.prompts import ABSTENTION_PROMPT
from evalora.llm.base import LLMClient, Message, Role
from evalora.llm.costs import TokenUsage
from evalora.llm.json_mode import complete_json_with_usage

_ABSTENTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bi (?:do not|don't) know\b",
        r"\bnot (?:enough|sufficient) (?:information|context)\b",
        r"\b(?:cannot|can't|unable to) (?:answer|determine|find|confirm|say)\b",
        r"\b(?:the )?(?:provided )?context (?:does not|doesn't) "
        r"(?:contain|mention|include|say)\b",
        r"\bnot (?:stated|mentioned|specified|available) in the "
        r"(?:provided )?context\b",
        r"\bno (?:information|answer|relevant information)\b",
        r"\bi (?:cannot|can't) (?:find|determine)\b",
        r"\bi(?:'m| am) not sure\b",
        r"\b(?:do not|don't) have enough (?:information|context)\b",
    )
)
_AMBIGUOUS_MARKERS = frozenset(
    {"maybe", "possibly", "perhaps", "unclear", "uncertain", "unsure", "depends"}
)


class AbstentionVerdict(BaseModel):
    abstained: bool
    reason: str


class AbstentionResult(BaseModel):
    abstained: bool
    method: Literal["rule", "llm", "default"]
    reason: str | None = None
    usages: list[TokenUsage] = Field(default_factory=list)


def detect_abstention_rule_based(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return True
    return any(pattern.search(text) for pattern in _ABSTENTION_PATTERNS)


def _is_ambiguous(answer: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", answer.lower()))
    return bool(tokens & _AMBIGUOUS_MARKERS)


async def detect_abstention(
    client: LLMClient,
    *,
    question: str,
    answer: str,
    use_llm_fallback: bool = True,
) -> AbstentionResult:
    if not answer.strip():
        return AbstentionResult(abstained=True, method="rule", reason="empty answer")
    if detect_abstention_rule_based(answer):
        return AbstentionResult(
            abstained=True,
            method="rule",
            reason="matched a refusal pattern",
        )
    if use_llm_fallback and _is_ambiguous(answer):
        messages = [
            Message(role=Role.SYSTEM, content=ABSTENTION_PROMPT.system),
            Message(
                role=Role.USER,
                content=ABSTENTION_PROMPT.user.format(
                    question=question,
                    answer=answer,
                ),
            ),
        ]
        verdict, responses = await complete_json_with_usage(
            client,
            messages,
            AbstentionVerdict,
            temperature=0.0,
            max_tokens=256,
        )
        return AbstentionResult(
            abstained=verdict.abstained,
            method="llm",
            reason=verdict.reason,
            usages=[response.usage for response in responses if not response.cached],
        )
    return AbstentionResult(abstained=False, method="default")
