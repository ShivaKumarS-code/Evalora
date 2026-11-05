from __future__ import annotations

import pytest

from evalora.evaluate.correctness import judge_correctness
from evalora.llm.base import LLMClient, StructuredOutputError
from evalora.llm.fake import FakeProvider
from evalora.schemas import CorrectnessLabel


@pytest.mark.parametrize(
    ("raw_label", "expected_label", "expected_correct"),
    [
        ("correct", CorrectnessLabel.CORRECT, True),
        ("partially_correct", CorrectnessLabel.PARTIALLY_CORRECT, False),
        ("incorrect", CorrectnessLabel.INCORRECT, False),
    ],
)
async def test_correctness_labels(
    raw_label: str,
    expected_label: CorrectnessLabel,
    expected_correct: bool,
) -> None:
    provider = FakeProvider([f'{{"label": "{raw_label}", "reason": "because"}}'])
    client = LLMClient(provider, max_retries=0)

    result = await judge_correctness(
        client,
        question="When was TechMart founded?",
        gold_answer="1994",
        answer="TechMart was founded in 1994.",
    )

    assert result.label == expected_label
    assert result.correct is expected_correct
    assert result.reason == "because"
    assert len(result.usages) == 1


async def test_persistent_json_failure_raises() -> None:
    provider = FakeProvider(["bad", "still bad", "nope"])
    client = LLMClient(provider, max_retries=0)

    with pytest.raises(StructuredOutputError) as info:
        await judge_correctness(
            client,
            question="When?",
            gold_answer="1994",
            answer="1994",
        )

    assert info.value.attempts == 3
    assert provider.call_count == 3
