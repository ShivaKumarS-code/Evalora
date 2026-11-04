from __future__ import annotations

from pathlib import Path

import pytest

from evalora.schemas import (
    CaseResult,
    PipelineResponse,
    QuestionType,
    RetrievalScores,
)
from evalora.schemas import (
    TestCase as EvalTestCase,
)
from evalora.store.db import (
    RunMismatchError,
    RunNotFoundError,
    RunStore,
    default_db_path,
)


def _case(case_id: str) -> EvalTestCase:
    return EvalTestCase(
        case_id=case_id,
        type=QuestionType.SINGLE_HOP,
        question="When was TechMart founded?",
        gold_answer="1994",
        gold_chunk_ids=["c1"],
        source_model="fake",
        answerability_verdict="answerable",
        metadata={},
    )


def _result(case_id: str) -> CaseResult:
    return CaseResult(
        case=_case(case_id),
        response=PipelineResponse(
            answer="1994",
            contexts=[],
            latency_ms=1.0,
        ),
        retrieval=RetrievalScores(
            hit=False,
            reciprocal_rank=0.0,
            context_recall=0.0,
        ),
        faithfulness=None,
        claim_verdicts=[],
        correct=None,
        abstained=False,
        failure_category=None,
    )


def test_create_run_and_roundtrip_results(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    record = store.get_or_create_run(
        "baseline",
        config_snapshot={"pipeline": {"endpoint": "http://example.test/query"}},
        test_set_hash="hash-1",
        case_count=2,
        git_sha="abc123",
    )

    assert record.run_name == "baseline"
    assert record.completed_count == 0
    assert record.test_set_hash == "hash-1"
    assert record.git_sha == "abc123"

    store.save_result("baseline", _result("case-1"))
    store.save_result("baseline", _result("case-2"))

    assert store.completed_case_ids("baseline") == {"case-1", "case-2"}
    loaded = store.load_results("baseline")
    assert [result.case.case_id for result in loaded] == ["case-1", "case-2"]
    updated = store.get_run("baseline")
    assert updated is not None
    assert updated.completed_count == 2


def test_get_or_create_reuses_existing_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    first = store.get_or_create_run(
        "run",
        config_snapshot={},
        test_set_hash="hash-1",
        case_count=1,
    )
    second = store.get_or_create_run(
        "run",
        config_snapshot={},
        test_set_hash="hash-1",
        case_count=1,
    )

    assert first.created_at == second.created_at


def test_get_or_create_rejects_different_test_set(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.get_or_create_run(
        "run",
        config_snapshot={},
        test_set_hash="hash-1",
        case_count=1,
    )

    with pytest.raises(RunMismatchError, match="different test set"):
        store.get_or_create_run(
            "run",
            config_snapshot={},
            test_set_hash="hash-2",
            case_count=1,
        )


def test_save_result_requires_a_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")

    with pytest.raises(RunNotFoundError):
        store.save_result("missing", _result("case-1"))


def test_baseline_set_and_get(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.get_or_create_run(
        "run",
        config_snapshot={},
        test_set_hash="hash-1",
        case_count=1,
    )

    store.set_baseline("main", "run", metadata={"source": "test"})

    assert store.get_baseline("main") == "run"
    assert store.get_baseline("missing") is None


def test_baseline_requires_an_existing_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")

    with pytest.raises(RunNotFoundError):
        store.set_baseline("main", "missing")


def test_default_db_path_can_be_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "custom.db"
    monkeypatch.setenv("EVALORA_DB_PATH", str(target))

    assert default_db_path() == target
