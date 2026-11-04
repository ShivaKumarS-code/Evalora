"""Async pipeline runner, test-set loading, and per-case error capture."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from evalora.config import PipelineConfig
from evalora.evaluate.retrieval import compute_retrieval
from evalora.run.adapter import ContractError, parse_pipeline_response
from evalora.schemas import (
    CaseResult,
    FailureCategory,
    PipelineResponse,
    TestCase,
)

SleepFn = Callable[[float], Awaitable[None]]


class TestSetError(Exception):
    """Raised when a test set file is missing, empty, or invalid."""


def load_test_set(path: str | Path) -> list[TestCase]:
    test_path = Path(path)
    if not test_path.is_file():
        raise TestSetError(f"Test set not found: {test_path}")
    cases: list[TestCase] = []
    seen_ids: set[str] = set()
    with test_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = TestCase.model_validate_json(line)
            except ValidationError as exc:
                raise TestSetError(f"{test_path}:{line_number}: invalid test case: {exc}") from exc
            if case.case_id in seen_ids:
                raise TestSetError(f"{test_path}:{line_number}: duplicate case_id '{case.case_id}'")
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise TestSetError(f"Test set is empty: {test_path}")
    return cases


def hash_test_set(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gold_chunk_texts(case: TestCase) -> dict[str, str]:
    raw = case.metadata.get("gold_chunk_texts")
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if isinstance(value, str)}


def _error_result(
    case: TestCase,
    message: str,
    latency_ms: float,
    pipeline: PipelineConfig,
    gold_chunk_texts: dict[str, str],
) -> CaseResult:
    response = PipelineResponse(
        answer="",
        contexts=[],
        latency_ms=latency_ms,
        error=message,
    )
    retrieval = compute_retrieval(
        case,
        response.contexts,
        gold_chunk_texts,
        threshold=pipeline.match_threshold,
    )
    return CaseResult(
        case=case,
        response=response,
        retrieval=retrieval,
        faithfulness=None,
        claim_verdicts=[],
        correct=None,
        abstained=False,
        failure_category=FailureCategory.PIPELINE_ERROR,
    )


async def run_case(
    client: httpx.AsyncClient,
    case: TestCase,
    pipeline: PipelineConfig,
    *,
    max_retries: int = 2,
    retry_base_delay: float = 0.5,
    sleep: SleepFn = asyncio.sleep,
) -> CaseResult:
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    gold_chunk_texts = _gold_chunk_texts(case)
    attempts = max_retries + 1

    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = await client.post(
                str(pipeline.endpoint),
                json={"question": case.question},
                headers=pipeline.headers,
            )
        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - started) * 1000
            message = f"timeout after {pipeline.timeout_s:g}s"
            if attempt < max_retries:
                await sleep(retry_base_delay * 2**attempt)
                continue
            return _error_result(case, message, elapsed_ms, pipeline, gold_chunk_texts)
        except httpx.TransportError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            message = f"transport error: {exc}"
            if attempt < max_retries:
                await sleep(retry_base_delay * 2**attempt)
                continue
            return _error_result(case, message, elapsed_ms, pipeline, gold_chunk_texts)

        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.status_code == 429 or response.status_code >= 500:
            message = f"HTTP {response.status_code}: {response.text[:300]}"
            if attempt < max_retries:
                await sleep(retry_base_delay * 2**attempt)
                continue
            return _error_result(case, message, elapsed_ms, pipeline, gold_chunk_texts)
        if response.status_code >= 400:
            message = f"HTTP {response.status_code}: {response.text[:300]}"
            return _error_result(case, message, elapsed_ms, pipeline, gold_chunk_texts)

        try:
            payload: Any = response.json()
        except ValueError:
            return _error_result(
                case,
                "response body was not valid JSON",
                elapsed_ms,
                pipeline,
                gold_chunk_texts,
            )
        try:
            pipeline_response = parse_pipeline_response(
                payload,
                pipeline.response_mapping,
                elapsed_ms,
            )
        except ContractError as exc:
            return _error_result(case, str(exc), elapsed_ms, pipeline, gold_chunk_texts)

        retrieval = compute_retrieval(
            case,
            pipeline_response.contexts,
            gold_chunk_texts,
            threshold=pipeline.match_threshold,
        )
        return CaseResult(
            case=case,
            response=pipeline_response,
            retrieval=retrieval,
            faithfulness=None,
            claim_verdicts=[],
            correct=None,
            abstained=False,
            failure_category=None,
        )

    raise AssertionError("retry loop exited without returning or raising")


async def run_test_set(
    cases: Sequence[TestCase],
    pipeline: PipelineConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    completed_case_ids: set[str] | None = None,
    on_result: Callable[[CaseResult], None] | None = None,
    max_retries: int = 2,
    retry_base_delay: float = 0.5,
    sleep: SleepFn = asyncio.sleep,
) -> list[CaseResult]:
    completed = completed_case_ids or set()
    pending = [case for case in cases if case.case_id not in completed]
    if not pending:
        return []

    semaphore = asyncio.Semaphore(pipeline.concurrency)
    timeout = httpx.Timeout(pipeline.timeout_s)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:

        async def run_one(case: TestCase) -> CaseResult:
            async with semaphore:
                return await run_case(
                    client,
                    case,
                    pipeline,
                    max_retries=max_retries,
                    retry_base_delay=retry_base_delay,
                    sleep=sleep,
                )

        tasks = [asyncio.create_task(run_one(case)) for case in pending]
        results_by_id: dict[str, CaseResult] = {}
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            results_by_id[result.case.case_id] = result
            if on_result is not None:
                on_result(result)

    return [results_by_id[case.case_id] for case in pending]
