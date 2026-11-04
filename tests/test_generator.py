from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel

from evalora.config import EvaloraConfig
from evalora.generate.filters import AnswerabilityVerdict, DropReason
from evalora.generate.generator import (
    QUESTION_TYPE_ORDER,
    AdversarialDraft,
    GenerationReport,
    MultiHopDraft,
    SingleHopDraft,
    UnanswerableDraft,
    allocate_requests,
    generate_test_set,
    make_case_id,
)
from evalora.llm.base import LLMClient, Message
from evalora.llm.cache import ResponseCache
from evalora.llm.fake import FakeProvider
from evalora.schemas import QuestionType
from evalora.schemas import TestCase as EvalTestCase

ALPHA_TEXT = (
    "Alpha Corporation was founded in 1998 in Denver. "
    "Its flagship product, AlphaSync, launched in 2004. "
    "The company reported revenue of 42 million dollars in 2023. "
    "Alpha's CEO is Morgan Lee. "
    "The research team publishes on distributed systems.\n\n"
    "AlphaSync stores customer records in a regional data center. "
    "The support team operates from Austin. "
    "Alpha acquired Beta Labs in 2019 for 7 million dollars. "
    "Beta Labs built the Fathom search engine. "
    "The combined company has 300 employees."
)
BETA_TEXT = (
    "Beta Labs was founded in 2001 in Seattle. "
    "Its main product, Fathom, indexes technical manuals. "
    "Beta Labs was acquired by Alpha Corporation in 2019. "
    "The acquisition price was 7 million dollars. "
    "Beta's founder is Priya Nair.\n\n"
    "Fathom version 3 shipped in 2018 with a new ranking model. "
    "The ranking model uses sparse lexical signals. "
    "Beta Labs employed 45 people before the acquisition. "
    "Its office moved to Denver in 2020. "
    "The Fathom API supports batch indexing."
)


def _handler(
    messages: Sequence[Message],
    schema: type[BaseModel] | None,
) -> str:
    prompt = messages[-1].content
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    if schema is SingleHopDraft:
        return json.dumps(
            {
                "question": f"single {digest}?",
                "gold_answer": f"answer {digest}",
                "supporting_quote": f"quote {digest}",
            }
        )
    if schema is MultiHopDraft:
        return json.dumps(
            {
                "question": f"multi {digest}?",
                "gold_answer": f"answer {digest}",
                "supporting_quotes": [f"quote a {digest}", f"quote b {digest}"],
            }
        )
    if schema is UnanswerableDraft:
        return json.dumps(
            {
                "question": f"unanswerable {digest}?",
                "rationale": "not present in the context",
            }
        )
    if schema is AdversarialDraft:
        return json.dumps(
            {
                "question": f"adversarial {digest}?",
                "gold_answer": f"answer {digest}",
                "supporting_quote": f"quote {digest}",
                "near_miss_detail": "similar but different detail",
            }
        )
    if schema is AnswerabilityVerdict:
        return json.dumps(
            {
                "answerable": "unanswerable" not in prompt,
                "reason": "test verdict",
            }
        )
    raise AssertionError(f"unexpected schema: {schema}")


def _make_config(
    tmp_path: Path,
    *,
    cases_per_type: dict[QuestionType, int] | None = None,
    text: str | None = None,
    chunk_size: int = 200,
) -> EvaloraConfig:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    if text is None:
        (docs / "alpha.md").write_text(ALPHA_TEXT, encoding="utf-8")
        (docs / "beta.md").write_text(BETA_TEXT, encoding="utf-8")
    else:
        (docs / "alpha.md").write_text(text, encoding="utf-8")
    counts = cases_per_type or {question_type: 1 for question_type in QUESTION_TYPE_ORDER}
    return EvaloraConfig.model_validate(
        {
            "docs": {
                "paths": [str(docs)],
                "chunk_size": chunk_size,
                "chunk_overlap": 30,
            },
            "generation": {
                "provider": "gemini",
                "model": "gemini-2.0-flash",
                "cases_per_type": {
                    question_type.value: count for question_type, count in counts.items()
                },
                "seed": 7,
            },
            "pipeline": {"endpoint": "http://localhost:8000/query"},
        }
    )


def test_allocate_requests_round_robin() -> None:
    counts = {
        QuestionType.SINGLE_HOP: 30,
        QuestionType.MULTI_HOP: 15,
        QuestionType.UNANSWERABLE: 15,
        QuestionType.ADVERSARIAL: 15,
    }

    assert allocate_requests(counts, None) == counts
    assert allocate_requests(counts, 5) == {
        QuestionType.SINGLE_HOP: 2,
        QuestionType.MULTI_HOP: 1,
        QuestionType.UNANSWERABLE: 1,
        QuestionType.ADVERSARIAL: 1,
    }
    assert allocate_requests(counts, 100) == counts
    with pytest.raises(ValueError):
        allocate_requests(counts, 0)


def test_make_case_id_is_stable_and_content_sensitive() -> None:
    first = make_case_id(QuestionType.SINGLE_HOP, "Who founded Alpha?", "Morgan", ["c1"])
    same = make_case_id(QuestionType.SINGLE_HOP, "who founded alpha?", "Morgan", ["c1"])
    different = make_case_id(QuestionType.SINGLE_HOP, "Who founded Alpha?", "Priya", ["c1"])

    assert first == same
    assert first != different


async def test_generate_end_to_end_writes_valid_jsonl_and_report(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    provider = FakeProvider(handler=_handler)
    client = LLMClient(provider)
    out = tmp_path / "test_set.jsonl"

    result = await generate_test_set(config, client, out=out)

    assert {case.type for case in result.cases} == set(QUESTION_TYPE_ORDER)
    assert len(result.cases) == 4
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    parsed = [EvalTestCase.model_validate_json(line) for line in lines]
    assert {case.type for case in parsed} == set(QUESTION_TYPE_ORDER)
    assert all("prompt_version" in case.metadata for case in parsed)
    assert all(case.answerability_verdict in {"answerable", "unanswerable"} for case in parsed)

    report_path = tmp_path / "generation_report.json"
    report = GenerationReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    assert report.requested == {question_type: 1 for question_type in QUESTION_TYPE_ORDER}
    assert report.kept == {question_type: 1 for question_type in QUESTION_TYPE_ORDER}
    assert report.dropped == {}
    assert report.chunks > 1
    assert report.token_usage.calls == 8
    assert report.prompt_versions["single_hop"] == "v1"
    assert provider.call_count == 8


async def test_same_seed_and_warm_cache_reproduces_identical_files(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    cache = ResponseCache(tmp_path / "cache.db")
    first_out = tmp_path / "first.jsonl"
    first_provider = FakeProvider(handler=_handler)
    first = await generate_test_set(
        config,
        LLMClient(first_provider, cache=cache),
        out=first_out,
    )
    first_bytes = first_out.read_bytes()
    first_report_bytes = (tmp_path / "generation_report.json").read_bytes()

    second_out = tmp_path / "second.jsonl"
    second_provider = FakeProvider(handler=_handler)
    second = await generate_test_set(
        config,
        LLMClient(second_provider, cache=cache),
        out=second_out,
    )

    assert second_provider.call_count == 0
    assert second_out.read_bytes() == first_bytes
    assert (tmp_path / "generation_report.json").read_bytes() == first_report_bytes
    assert second.report == first.report
    assert second.cases == first.cases


async def test_single_chunk_corpus_drops_pair_question_types(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        text="Alpha was founded in 1998 in Denver.",
        chunk_size=500,
    )
    provider = FakeProvider(handler=_handler)

    result = await generate_test_set(
        config,
        LLMClient(provider),
        out=tmp_path / "test_set.jsonl",
    )

    assert result.report.kept[QuestionType.SINGLE_HOP] == 1
    assert result.report.kept[QuestionType.UNANSWERABLE] == 1
    assert result.report.kept[QuestionType.MULTI_HOP] == 0
    assert result.report.kept[QuestionType.ADVERSARIAL] == 0
    assert result.report.dropped[DropReason.INSUFFICIENT_CONTEXT] == 2
