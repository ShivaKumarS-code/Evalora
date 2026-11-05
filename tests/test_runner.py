from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx

from evalora.config import PipelineConfig
from evalora.run.runner import (
    TestSetError as RunnerTestSetError,
)
from evalora.run.runner import (
    hash_test_set,
    load_test_set,
    run_test_set,
)
from evalora.schemas import CaseResult, FailureCategory, QuestionType
from evalora.schemas import TestCase as EvalTestCase

URL = "https://pipeline.test/query"
GOLD_TEXT = "TechMart was founded in 1994 in Portland, Oregon."


def _pipeline(**overrides: object) -> PipelineConfig:
    data: dict[str, object] = {
        "endpoint": URL,
        "timeout_s": 1.0,
        "concurrency": 2,
        "match_threshold": 0.8,
        "headers": {},
    }
    data.update(overrides)
    return PipelineConfig.model_validate(data)


def _case(case_id: str = "case-1") -> EvalTestCase:
    return EvalTestCase(
        case_id=case_id,
        type=QuestionType.SINGLE_HOP,
        question="When was TechMart founded?",
        gold_answer="1994",
        gold_chunk_ids=["c1"],
        source_model="fake",
        answerability_verdict="answerable",
        metadata={"gold_chunk_texts": {"c1": GOLD_TEXT}},
    )


@respx.mock
async def test_success_records_retrieval_scores() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "1994",
                "contexts": [{"id": "c1", "text": GOLD_TEXT}],
            },
        )
    )

    results = await run_test_set([_case()], _pipeline(), max_retries=0)

    assert route.call_count == 1
    result = results[0]
    assert result.response.error is None
    assert result.response.latency_ms >= 0
    assert result.retrieval.hit is True
    assert result.retrieval.reciprocal_rank == 1.0
    assert result.retrieval.context_recall == 1.0
    assert result.failure_category is None


@respx.mock
async def test_timeout_is_recorded_without_crashing() -> None:
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))

    results = await run_test_set([_case()], _pipeline(), max_retries=0)

    result = results[0]
    assert result.response.error is not None
    assert "timeout" in result.response.error
    assert result.failure_category == FailureCategory.PIPELINE_ERROR
    assert result.retrieval.hit is False


@respx.mock
async def test_server_error_is_recorded_without_crashing() -> None:
    respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))

    results = await run_test_set([_case()], _pipeline(), max_retries=0)

    assert results[0].response.error is not None
    assert "HTTP 500" in results[0].response.error


@respx.mock
async def test_malformed_contract_is_recorded_without_crashing() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"answer": "1994"}))

    results = await run_test_set([_case()], _pipeline(), max_retries=0)

    assert results[0].response.error is not None
    assert "contexts" in results[0].response.error


@respx.mock
async def test_invalid_json_is_recorded_without_crashing() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, text="not json"))

    results = await run_test_set([_case()], _pipeline(), max_retries=0)

    assert results[0].response.error == "response body was not valid JSON"


@respx.mock
async def test_transient_failure_is_retried_then_succeeds() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(500, text="temporary"),
            httpx.Response(
                200,
                json={"answer": "1994", "contexts": [{"id": "c1", "text": GOLD_TEXT}]},
            ),
        ]
    )

    results = await run_test_set(
        [_case()],
        _pipeline(),
        max_retries=1,
        retry_base_delay=0.5,
        sleep=fake_sleep,
    )

    assert route.call_count == 2
    assert sleeps == [pytest.approx(0.5)]
    assert results[0].response.error is None


async def test_concurrency_limit_is_respected() -> None:
    transport = TrackingTransport(delay=0.01)
    cases = [_case(f"case-{index}") for index in range(6)]

    results = await run_test_set(
        cases,
        _pipeline(concurrency=2),
        transport=transport,
        max_retries=0,
    )

    assert len(results) == 6
    assert transport.calls == 6
    assert transport.max_active <= 2


async def test_completed_cases_are_skipped() -> None:
    transport = TrackingTransport(delay=0.0)

    results = await run_test_set(
        [_case("case-1"), _case("case-2")],
        _pipeline(),
        transport=transport,
        completed_case_ids={"case-1"},
        max_retries=0,
    )

    assert transport.calls == 1
    assert [result.case.case_id for result in results] == ["case-2"]


async def test_processor_is_applied_before_the_callback() -> None:
    transport = TrackingTransport(delay=0.0)
    seen: list[CaseResult] = []

    async def processor(result: CaseResult) -> CaseResult:
        return result.model_copy(update={"faithfulness": 1.0})

    results = await run_test_set(
        [_case()],
        _pipeline(),
        transport=transport,
        processor=processor,
        on_result=seen.append,
        max_retries=0,
    )

    assert results[0].faithfulness == 1.0
    assert seen[0].faithfulness == 1.0


def test_load_test_set_validates_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "test_set.jsonl"
    path.write_text(
        _case("case-1").model_dump_json() + "\n" + _case("case-2").model_dump_json() + "\n",
        encoding="utf-8",
    )

    cases = load_test_set(path)

    assert [case.case_id for case in cases] == ["case-1", "case-2"]
    assert len(hash_test_set(path)) == 64


def test_load_test_set_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "test_set.jsonl"
    case = _case("case-1").model_dump_json()
    path.write_text(case + "\n" + case + "\n", encoding="utf-8")

    with pytest.raises(RunnerTestSetError, match="duplicate case_id"):
        load_test_set(path)


def test_load_test_set_rejects_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "test_set.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(RunnerTestSetError, match="invalid test case"):
        load_test_set(path)


def test_load_test_set_rejects_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "test_set.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(RunnerTestSetError, match="empty"):
        load_test_set(path)


def test_hash_test_set_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "test_set.jsonl"
    path.write_text(_case("case-1").model_dump_json() + "\n", encoding="utf-8")
    first = hash_test_set(path)
    path.write_text(_case("case-2").model_dump_json() + "\n", encoding="utf-8")

    assert hash_test_set(path) != first


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, delay: float) -> None:
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls += 1
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return httpx.Response(
                200,
                json={"answer": "ok", "contexts": []},
                request=request,
            )
        finally:
            self.active -= 1
