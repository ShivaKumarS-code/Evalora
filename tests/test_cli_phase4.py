from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalora.cli import app
from evalora.llm.base import LLMClient
from evalora.llm.costs import UsageTotals
from evalora.llm.fake import FakeProvider
from evalora.schemas import ClaimVerdict, FailureCategory, QuestionType
from evalora.store.db import RunStore
from evalora.validate.judge_eval import JudgeValidationSummary, compute_metrics
from evalora.validate.labels import LabelItem, LabelValue, write_labels
from tests.factories import make_result

runner = CliRunner()


def _seed_run(path: Path) -> None:
    store = RunStore(path)
    store.get_or_create_run(
        "run-1",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="hash-1",
        case_count=1,
    )
    result = make_result(
        "case-1",
        QuestionType.SINGLE_HOP,
        failure_category=FailureCategory.HALLUCINATION,
        claim_verdicts=[
            ClaimVerdict(
                claim="TechMart was founded in 1994.",
                supported=True,
                evidence_context_index=0,
                rationale="stated",
            )
        ],
    )
    store.save_result("run-1", result)


def _label_items() -> list[LabelItem]:
    return [
        LabelItem(
            item_id="item-1",
            contexts=[],
            answer="answer",
            claim="TechMart was founded in 1994.",
            label=LabelValue.SUPPORTED,
        ),
        LabelItem(
            item_id="item-2",
            contexts=[],
            answer="answer",
            claim="TechMart has 500 stores.",
            label=LabelValue.UNSUPPORTED,
        ),
    ]


def _write_ragtruth_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    responses = [
        {
            "id": "1",
            "source_id": "s1",
            "labels": [],
            "split": "test",
            "quality": "good",
            "response": "Answer one",
        },
        {
            "id": "2",
            "source_id": "s1",
            "labels": [{"start": 0, "end": 3, "text": "Bad"}],
            "split": "test",
            "quality": "good",
            "response": "Bad answer",
        },
    ]
    sources = [
        {
            "source_id": "s1",
            "task_type": "Summary",
            "source_info": "Source text.",
        }
    ]
    responses_path = tmp_path / "response.jsonl"
    sources_path = tmp_path / "source_info.jsonl"
    responses_path.write_text(
        "\n".join(json.dumps(item) for item in responses) + "\n", encoding="utf-8"
    )
    sources_path.write_text(
        "\n".join(json.dumps(item) for item in sources) + "\n", encoding="utf-8"
    )
    return responses_path, sources_path


def test_export_labels_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "runs.db"
    _seed_run(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    out = tmp_path / "labels.html"

    result = runner.invoke(
        app,
        ["export-labels", "--run", "run-1", "--out", str(out), "--max-items", "10"],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "TechMart was founded in 1994." in out.read_text(encoding="utf-8")


def test_ragtruth_sample_command(tmp_path: Path) -> None:
    responses_path, sources_path = _write_ragtruth_fixtures(tmp_path)
    out = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "ragtruth-sample",
            "--responses",
            str(responses_path),
            "--sources",
            str(sources_path),
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--sample-size",
            "2",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert manifest.is_file()
    assert json.loads(manifest.read_text(encoding="utf-8"))["license"] == "MIT"


def test_validate_judge_command_with_fake_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, _label_items())
    out = tmp_path / "judge_validation.json"
    client = LLMClient(
        FakeProvider(
            [
                '{"supported": true, "evidence_context_index": 0, "rationale": "ok"}',
                '{"supported": false, "evidence_context_index": null, "rationale": "absent"}',
            ]
        ),
        max_retries=0,
    )
    monkeypatch.setattr("evalora.cli.build_llm_client", lambda **kwargs: client)

    result = runner.invoke(
        app,
        [
            "validate-judge",
            "--labels",
            str(labels_path),
            "--out",
            str(out),
            "--model",
            "fake",
            "--resamples",
            "50",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Precision" in result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n"] == 2
    assert payload["confusion"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
    assert payload["model"] == "fake"


def test_validate_judge_rejects_unknown_prompt(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, _label_items())

    result = runner.invoke(
        app,
        [
            "validate-judge",
            "--labels",
            str(labels_path),
            "--prompt-version",
            "v9",
        ],
    )

    assert result.exit_code == 1
    assert "unknown prompt version" in result.output


def test_report_command_embeds_judge_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "runs.db"
    _seed_run(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    metrics = compute_metrics([True, False], [True, True], resamples=20, seed=7)
    summary = JudgeValidationSummary(
        **metrics.model_dump(),
        model="fake",
        prompt_version="v1",
        dataset="RAGTruth",
        sample_seed=42,
        sample_size=2,
        ci_confidence=0.95,
        bootstrap_resamples=20,
        bootstrap_seed=7,
        token_usage=UsageTotals(),
        created_at="2026-01-01T00:00:00+00:00",
        items=[],
    )
    validation_path = tmp_path / "judge_validation.json"
    validation_path.write_text(summary.model_dump_json(), encoding="utf-8")
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            "--run",
            "run-1",
            "--out",
            str(out),
            "--judge-validation",
            str(validation_path),
        ],
    )

    assert result.exit_code == 0, result.output
    html = out.read_text(encoding="utf-8")
    assert "Judge validation" in html
    assert "RAGTruth" in html
