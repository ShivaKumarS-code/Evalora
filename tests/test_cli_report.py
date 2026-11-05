from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalora.cli import app
from evalora.schemas import FailureCategory
from evalora.store.db import RunStore
from tests.factories import make_result

runner = CliRunner()


def _seed_store(path: Path) -> None:
    store = RunStore(path)
    store.get_or_create_run(
        "run-1",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="hash-1",
        case_count=1,
    )
    store.save_result(
        "run-1",
        make_result("case-1", failure_category=FailureCategory.HALLUCINATION),
    )
    store.get_or_create_run(
        "baseline",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="hash-0",
        case_count=1,
    )
    store.save_result("baseline", make_result("case-1", failure_category=None))


def test_report_command_writes_html_with_baseline_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "runs.db"
    _seed_store(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            "--run",
            "run-1",
            "--out",
            str(out),
            "--baseline",
            "baseline",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.is_file()
    html = out.read_text(encoding="utf-8")
    assert "Baseline diff" in html
    assert "case-1" in html


def test_report_command_rejects_unknown_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALORA_DB_PATH", str(tmp_path / "runs.db"))

    result = runner.invoke(
        app,
        ["report", "--run", "missing", "--out", str(tmp_path / "report.html")],
    )

    assert result.exit_code == 1
    assert "not found" in result.output
