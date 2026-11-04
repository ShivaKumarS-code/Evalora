"""Versioned prompts used for test generation and filtering."""

from __future__ import annotations

from dataclasses import dataclass

PROMPT_VERSION = "v1"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str


GENERATION_SYSTEM = (
    "You create evaluation data for retrieval systems. "
    "Return only JSON matching the requested schema."
)
JUDGE_SYSTEM = (
    "You judge whether questions are answerable from retrieved context. "
    "Return only JSON matching the requested schema."
)

SINGLE_HOP_PROMPT = PromptTemplate(
    name="single_hop",
    version=PROMPT_VERSION,
    system=GENERATION_SYSTEM,
    user=(
        "Write one factual question that can be answered using only the passage below.\n"
        "The question must be self-contained and specific, and the answer must be a short "
        "phrase. Prefer an important fact over a trivial detail.\n"
        "Also return the exact sentence or clause from the passage that supports the answer "
        "and a short gold answer.\n\n"
        "Passage:\n{context}"
    ),
)

MULTI_HOP_PROMPT = PromptTemplate(
    name="multi_hop",
    version=PROMPT_VERSION,
    system=GENERATION_SYSTEM,
    user=(
        "Write one factual question that requires facts from both passages to answer. "
        "The question must not be answerable from either passage alone. "
        "Return a short gold answer and one exact supporting quote from each passage.\n\n"
        "Passage A:\n{context_a}\n\nPassage B:\n{context_b}"
    ),
)

UNANSWERABLE_PROMPT = PromptTemplate(
    name="unanswerable",
    version=PROMPT_VERSION,
    system=GENERATION_SYSTEM,
    user=(
        "Below is a source passage and neighboring passages. Write one plausible, specific "
        "question on the same topic whose answer is NOT stated or implied anywhere in the "
        "provided context. Do not ask about anything the context answers. Return the question "
        "and a short note explaining why the context cannot answer it.\n\n"
        "Context:\n{context}"
    ),
)

ADVERSARIAL_PROMPT = PromptTemplate(
    name="adversarial",
    version=PROMPT_VERSION,
    system=GENERATION_SYSTEM,
    user=(
        "Two passages on the same topic contain similar but different details. Write one "
        "question whose correct answer is supported only by the target passage, while the "
        "near-miss passage contains a similar-looking but incorrect detail. The question "
        "should tempt a retriever that grabs the near-miss passage. Return the question, a "
        "short gold answer, the exact supporting quote from the target passage, and the "
        "near-miss detail.\n\n"
        "Target passage:\n{target}\n\nNear-miss passage:\n{near_miss}"
    ),
)

ANSWERABILITY_PROMPT = PromptTemplate(
    name="answerability",
    version=PROMPT_VERSION,
    system=JUDGE_SYSTEM,
    user=(
        "Decide whether the question can be answered using only the provided context. "
        "Answer true only if the context states or directly implies the answer. Answer false "
        "if the answer requires outside knowledge or is absent.\n\n"
        "Question: {question}\n\nContext:\n{context}"
    ),
)

PROMPT_VERSIONS: dict[str, str] = {
    prompt.name: prompt.version
    for prompt in (
        SINGLE_HOP_PROMPT,
        MULTI_HOP_PROMPT,
        UNANSWERABLE_PROMPT,
        ADVERSARIAL_PROMPT,
        ANSWERABILITY_PROMPT,
    )
}
