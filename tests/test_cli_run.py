from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from evalora.cli import app
from evalora.config import DEFAULT_CONFIG_YAML
from evalora.schemas import QuestionType
from evalora.schemas import TestCase as EvalTestCase

runner = CliRunner()
URL = "https://pipeline.test/query"


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "evalora.yaml"
    config_path.write_text(
        DEFAULT_CONFIG_YAML.replace(
            "endpoint: http://localhost:8000/query",
            f"endpoint: {URL}",
        ),
        encoding="utf-8",
    )
    case = EvalTestCase(
        case_id="cli-case",
        type=QuestionType.SINGLE_HOP,
        question="When was TechMart founded?",
        gold_answer="1994",
        gold_chunk_ids=["c1"],
        source_model="fake",
        answerability_verdict="answerable",
        metadata={"gold_chunk_texts": {"c1": "TechMart was founded in 1994 in Portland, Oregon."}},
    )
    test_set_path = tmp_path / "test_set.jsonl"
    test_set_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    return config_path, test_set_path


@respx.mock
def test_run_command_prints_summary_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))
    config_path, test_set_path = _write_inputs(tmp_path)
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "1994",
                "contexts": [
                    {
                        "id": "c1",
                        "text": "TechMart was founded in 1994 in Portland, Oregon.",
                    }
                ],
            },
        )
    )
    args = [
        "run",
        "--run-name",
        "cli-run",
        "--config",
        str(config_path),
        "--tests",
        str(test_set_path),
        "--no-judge",
    ]

    first = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert "Hit rate" in first.output
    assert "1.000" in first.output
    assert route.call_count == 1

    second = runner.invoke(app, args)

    assert second.exit_code == 0, second.output
    assert "All cases already completed." in second.output
    assert route.call_count == 1
