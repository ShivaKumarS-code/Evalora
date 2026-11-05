"""Versioned prompts for faithfulness, correctness and abstention judges."""

from __future__ import annotations

from dataclasses import dataclass

JUDGE_PROMPT_VERSION = "v1"


@dataclass(frozen=True)
class JudgePrompt:
    name: str
    version: str
    system: str
    user: str


CLAIM_EXTRACTION_PROMPT = JudgePrompt(
    name="claim_extraction",
    version=JUDGE_PROMPT_VERSION,
    system=(
        "You extract atomic factual claims from answers. "
        "Return only JSON matching the requested schema."
    ),
    user=(
        "Split the answer into atomic, self-contained factual claims. Resolve pronouns so "
        "each claim stands alone. Ignore hedges, greetings, opinions and refusals. If there "
        "are no factual claims, return an empty list.\n\n"
        "Answer:\n{answer}"
    ),
)

CLAIM_VERIFICATION_PROMPT = JudgePrompt(
    name="claim_verification",
    version=JUDGE_PROMPT_VERSION,
    system=(
        "You verify claims using retrieved context only. "
        "Return only JSON matching the requested schema."
    ),
    user=(
        "Decide whether the claim is supported by the provided context. Use only the "
        "context, not outside knowledge. Return supported, the 0-based index of the single "
        "context that best supports the claim (or null), and a short rationale.\n\n"
        "Claim:\n{claim}\n\nContext:\n{contexts}"
    ),
)

CORRECTNESS_PROMPT = JudgePrompt(
    name="correctness",
    version=JUDGE_PROMPT_VERSION,
    system=(
        "You grade answers against a gold answer. Return only JSON matching the requested schema."
    ),
    user=(
        "Compare the candidate answer with the gold answer for the question. Label the "
        "answer as correct, partially_correct or incorrect. A correct answer may be "
        "phrased differently. A partially correct answer includes some correct content but "
        "misses or contradicts an important part. Return the label and a short reason.\n\n"
        "Question:\n{question}\n\nGold answer:\n{gold_answer}\n\nCandidate answer:\n{answer}"
    ),
)

ABSTENTION_PROMPT = JudgePrompt(
    name="abstention",
    version=JUDGE_PROMPT_VERSION,
    system=(
        "You classify whether an answer abstains. Return only JSON matching the requested schema."
    ),
    user=(
        "Decide whether the answer abstains from answering the question, for example by "
        "refusing, saying it does not know, or saying the information is missing. An answer "
        "that makes a substantive attempt is not an abstention. Return abstained and a short "
        "reason.\n\n"
        "Question:\n{question}\n\nAnswer:\n{answer}"
    ),
)

JUDGE_PROMPT_VERSIONS: dict[str, str] = {
    prompt.name: prompt.version
    for prompt in (
        CLAIM_EXTRACTION_PROMPT,
        CLAIM_VERIFICATION_PROMPT,
        CORRECTNESS_PROMPT,
        ABSTENTION_PROMPT,
    )
}
