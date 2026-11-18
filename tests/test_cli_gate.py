"""CLI tests for the gate, baseline management, and run cost controls."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalora.cli import app
from evalora.config import DEFAULT_CONFIG_YAML
from evalora.run.estimate import estimate_run
from evalora.schemas import QuestionType
from evalora.store.db import RunStore
from tests.factories import make_result

runner = CliRunner()


def _healthy_result(case_id: str) -> object:
    return make_result(
        case_id,
        QuestionType.SINGLE_HOP,
        faithfulness=1.0,
        correct=True,
        hit=True,
        reciprocal_rank=1.0,
        context_recall=1.0,
    )


def _degraded_result(case_id: str) -> object:
    return make_result(
        case_id,
        QuestionType.SINGLE_HOP,
        faithfulness=0.4,
        correct=False,
        hit=False,
        reciprocal_rank=0.0,
        context_recall=0.0,
    )


def _seed_run(
    db_path: Path,
    run_name: str,
    results: list[object],
    *,
    test_set_hash: str = "hash-gate",
) -> None:
    store = RunStore(db_path)
    store.get_or_create_run(
        run_name,
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash=test_set_hash,
        case_count=len(results),
    )
    for result in results:
        store.save_result(run_name, result)  # type: ignore[arg-type]


def _write_config(
    tmp_path: Path,
    *,
    endpoint: str = "https://pipeline.test/query",
    judge_model: str = "gemini-3.5-flash-lite",
    faithfulness_max_drop: str = "0.05",
) -> Path:
    config_path = tmp_path / "evalora.yaml"
    content = (
        DEFAULT_CONFIG_YAML.replace(
            "endpoint: http://localhost:8000/query", f"endpoint: {endpoint}"
        )
        .replace("model: gemini-3.5-flash-lite", f"model: {judge_model}")
        .replace(
            "faithfulness: {max_drop: 0.05}",
            f"faithfulness: {{max_drop: {faithfulness_max_drop}}}",
        )
    )
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _write_test_set(tmp_path: Path, count: int = 3) -> Path:
    from evalora.schemas import TestCase as EvalTestCase

    cases = [
        EvalTestCase(
            case_id=f"case-{index}",
            type=QuestionType.SINGLE_HOP,
            question=f"Question {index}?",
            gold_answer=f"answer {index}",
            gold_chunk_ids=["c1"],
            source_model="fake",
            answerability_verdict="answerable",
            metadata={"gold_chunk_texts": {"c1": "TechMart was founded in 1994."}},
        )
        for index in range(count)
    ]
    path = tmp_path / "test_set.jsonl"
    path.write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")
    return path


def test_gate_passes_identical_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "runs.db"
    results = [_healthy_result("case-1"), _healthy_result("case-2")]
    _seed_run(db_path, "baseline-run", list(results))
    _seed_run(db_path, "current-run", list(results))
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))

    result = runner.invoke(
        app,
        ["gate", "--run", "current-run", "--baseline", "baseline-run", "--resamples", "50"],
    )

    assert result.exit_code == 0, result.output
    assert "passed" in result.output


def test_gate_fails_on_regression_and_writes_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "runs.db"
    _seed_run(db_path, "baseline-run", [_healthy_result("case-1")])
    _seed_run(db_path, "current-run", [_degraded_result("case-1")])
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    markdown_path = tmp_path / "gate.md"

    result = runner.invoke(
        app,
        [
            "gate",
            "--run",
            "current-run",
            "--baseline",
            "baseline-run",
            "--markdown",
            str(markdown_path),
            "--resamples",
            "50",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "FAILED" in result.output
    assert "fail" in result.output
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Evalora quality gate: ❌ failed" in markdown
    assert "↓" in markdown


def test_gate_missing_run_is_a_tool_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))

    result = runner.invoke(app, ["gate", "--run", "nope"])

    assert result.exit_code == 2
    assert "run 'nope' not found" in result.output


def test_gate_missing_baseline_is_a_tool_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "runs.db"
    _seed_run(db_path, "current-run", [_healthy_result("case-1")])
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))

    result = runner.invoke(app, ["gate", "--run", "current-run"])

    assert result.exit_code == 2
    assert "baseline 'main' not found" in result.output
    assert "evalora baseline set" in result.output


def test_gate_uses_config_thresholds_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "runs.db"
    _seed_run(db_path, "baseline-run", [_healthy_result("case-1")])
    near_perfect = make_result(
        "case-1",
        QuestionType.SINGLE_HOP,
        faithfulness=0.99,
        correct=True,
        hit=True,
        reciprocal_rank=1.0,
        context_recall=1.0,
    )
    _seed_run(db_path, "current-run", [near_perfect])
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    strict_config = _write_config(tmp_path, faithfulness_max_drop="0.0")

    with_defaults = runner.invoke(
        app,
        ["gate", "--run", "current-run", "--baseline", "baseline-run", "--resamples", "50"],
    )
    with_strict = runner.invoke(
        app,
        [
            "gate",
            "--run",
            "current-run",
            "--baseline",
            "baseline-run",
            "--config",
            str(strict_config),
            "--resamples",
            "50",
        ],
    )

    assert with_defaults.exit_code == 0, with_defaults.output
    assert with_strict.exit_code == 1, with_strict.output


def test_baseline_set_export_import_roundtrip_across_databases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_db = tmp_path / "source.db"
    results = [_healthy_result("case-1"), _healthy_result("case-2")]
    _seed_run(source_db, "good-run", list(results))
    monkeypatch.setenv("EVALORA_DB_PATH", str(source_db))

    set_result = runner.invoke(app, ["baseline", "set", "--run", "good-run", "--as", "main"])
    assert set_result.exit_code == 0, set_result.output
    assert "Baseline 'main' now points at run 'good-run'" in set_result.output

    baseline_path = tmp_path / "baseline.json"
    export_result = runner.invoke(
        app, ["baseline", "export", "--run", "good-run", "--out", str(baseline_path)]
    )
    assert export_result.exit_code == 0, export_result.output
    assert baseline_path.is_file()

    # A different CI machine: fresh database, only the current run exists.
    fresh_db = tmp_path / "fresh.db"
    _seed_run(fresh_db, "current-run", [_degraded_result("case-1")])
    monkeypatch.setenv("EVALORA_DB_PATH", str(fresh_db))

    import_result = runner.invoke(
        app, ["baseline", "import", "--file", str(baseline_path), "--as", "main"]
    )
    assert import_result.exit_code == 0, import_result.output
    assert "Imported run 'good-run'" in import_result.output

    gate_result = runner.invoke(app, ["gate", "--run", "current-run", "--resamples", "50"])
    assert gate_result.exit_code == 1, gate_result.output
    assert "vs good-run" in gate_result.output


def test_baseline_set_rejects_unknown_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))

    result = runner.invoke(app, ["baseline", "set", "--run", "nope", "--as", "main"])

    assert result.exit_code == 1
    assert "run 'nope' not found" in result.output


def test_run_dry_run_estimates_without_creating_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))
    config_path = _write_config(tmp_path)
    test_set_path = _write_test_set(tmp_path, count=3)

    result = runner.invoke(
        app,
        [
            "run",
            "--run-name",
            "dry-run",
            "--config",
            str(config_path),
            "--tests",
            str(test_set_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run for 'dry-run': 3 pending of 3 cases" in result.output
    assert "judge calls: ~15" in result.output
    assert "no verified price for 'gemini-3.5-flash-lite'" in result.output
    store = RunStore(tmp_path / "runs.db")
    assert store.get_run("dry-run") is None


def test_run_dry_run_respects_max_cases_and_budget_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))
    config_path = _write_config(tmp_path)
    test_set_path = _write_test_set(tmp_path, count=5)

    result = runner.invoke(
        app,
        [
            "run",
            "--run-name",
            "dry-run",
            "--config",
            str(config_path),
            "--tests",
            str(test_set_path),
            "--dry-run",
            "--max-cases",
            "2",
            "--max-budget-usd",
            "10.00",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2 pending of 5 cases" in result.output


def test_run_aborts_when_projection_exceeds_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))
    config_path = _write_config(tmp_path, judge_model="gemini-2.0-flash")
    test_set_path = _write_test_set(tmp_path, count=5)

    result = runner.invoke(
        app,
        [
            "run",
            "--run-name",
            "budget-run",
            "--config",
            str(config_path),
            "--tests",
            str(test_set_path),
            "--max-budget-usd",
            "0.0001",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "exceeds --max-budget-usd $0.00" in result.output
    store = RunStore(tmp_path / "runs.db")
    # Nothing ran and nothing was stored.
    assert store.get_run("budget-run") is None or not store.load_results("budget-run")


def test_estimate_run_math() -> None:
    from evalora.schemas import TestCase as EvalTestCase

    def _case(case_id: str, *, answerable: bool) -> EvalTestCase:
        return EvalTestCase(
            case_id=case_id,
            type=QuestionType.SINGLE_HOP if answerable else QuestionType.UNANSWERABLE,
            question="q",
            gold_answer="1994" if answerable else None,
            gold_chunk_ids=["c1"] if answerable else [],
            source_model="fake",
            answerability_verdict="answerable" if answerable else "unanswerable",
            metadata={},
        )

    cases = [_case("a", answerable=True), _case("b", answerable=True), _case("c", answerable=False)]

    judged = estimate_run(cases, provider="gemini", model="gemini-2.0-flash", judge_enabled=True)
    # 2 answerable cases: 1 extraction + 3 verifications + 0.25 abstention
    # fallback + 1 correctness = 5.25 each; 1 unanswerable: 4.25.
    assert judged.judge_calls == 14  # int(2 * 5.25 + 4.25)
    assert judged.pipeline_calls == 3
    assert judged.price_known is True
    assert judged.estimated_cost_usd > 0.0

    unjudged = estimate_run(cases, provider="gemini", model="gemini-2.0-flash", judge_enabled=False)
    assert unjudged.judge_calls == 0
    assert unjudged.estimated_cost_usd == 0.0

    unknown_price = estimate_run(
        cases, provider="gemini", model="gemini-3.5-flash-lite", judge_enabled=True
    )
    assert unknown_price.price_known is False
    assert unknown_price.estimated_cost_usd == 0.0
