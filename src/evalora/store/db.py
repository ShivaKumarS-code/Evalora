"""SQLite storage for runs, results and baselines. No ORM."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evalora.schemas import CaseResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    test_set_hash TEXT NOT NULL,
    git_sha TEXT,
    case_count INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS results (
    run_name TEXT NOT NULL,
    case_id TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_name, case_id),
    FOREIGN KEY (run_name) REFERENCES runs(run_name)
);

CREATE TABLE IF NOT EXISTS baselines (
    baseline_name TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY (run_name) REFERENCES runs(run_name)
);
"""


class StoreError(Exception):
    """Base class for run-store failures."""


class RunNotFoundError(StoreError):
    pass


class RunMismatchError(StoreError):
    pass


class RunRecord(BaseModel):
    run_name: str
    created_at: datetime
    updated_at: datetime
    config_snapshot: dict[str, Any]
    test_set_hash: str
    git_sha: str | None
    case_count: int
    completed_count: int


def default_db_path() -> Path:
    override = os.environ.get("EVALORA_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path(".evalora") / "runs.db"


def get_git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


class RunStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get_run(self, run_name: str) -> RunRecord | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_name = ?",
                (run_name,),
            ).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def get_or_create_run(
        self,
        run_name: str,
        *,
        config_snapshot: dict[str, Any],
        test_set_hash: str,
        case_count: int,
        git_sha: str | None = None,
    ) -> RunRecord:
        existing = self.get_run(run_name)
        if existing is not None:
            if existing.test_set_hash != test_set_hash:
                raise RunMismatchError(
                    f"run '{run_name}' was created with a different test set "
                    f"(existing hash {existing.test_set_hash[:12]}, "
                    f"new hash {test_set_hash[:12]}); use a new run name"
                )
            return existing

        now = datetime.now(UTC).isoformat()
        with self._session() as conn:
            conn.execute(
                "INSERT INTO runs ("
                "run_name, created_at, updated_at, config_json, test_set_hash, "
                "git_sha, case_count, completed_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    run_name,
                    now,
                    now,
                    json.dumps(config_snapshot, sort_keys=True),
                    test_set_hash,
                    git_sha,
                    case_count,
                ),
            )
        created = self.get_run(run_name)
        if created is None:
            raise StoreError(f"failed to create run '{run_name}'")
        return created

    def save_result(self, run_name: str, result: CaseResult) -> None:
        now = datetime.now(UTC).isoformat()
        with self._session() as conn:
            run_row = conn.execute(
                "SELECT 1 FROM runs WHERE run_name = ?",
                (run_name,),
            ).fetchone()
            if run_row is None:
                raise RunNotFoundError(f"run '{run_name}' does not exist")
            conn.execute(
                "INSERT OR REPLACE INTO results ("
                "run_name, case_id, result_json, created_at"
                ") VALUES (?, ?, ?, ?)",
                (
                    run_name,
                    result.case.case_id,
                    result.model_dump_json(),
                    now,
                ),
            )
            conn.execute(
                "UPDATE runs SET "
                "completed_count = (SELECT COUNT(*) FROM results WHERE run_name = ?), "
                "updated_at = ? "
                "WHERE run_name = ?",
                (run_name, now, run_name),
            )

    def load_results(self, run_name: str) -> list[CaseResult]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT result_json FROM results WHERE run_name = ? ORDER BY created_at, case_id",
                (run_name,),
            ).fetchall()
        return [CaseResult.model_validate_json(str(row["result_json"])) for row in rows]

    def completed_case_ids(self, run_name: str) -> set[str]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT case_id FROM results WHERE run_name = ?",
                (run_name,),
            ).fetchall()
        return {str(row["case_id"]) for row in rows}

    def set_baseline(
        self,
        baseline_name: str,
        run_name: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.get_run(run_name) is None:
            raise RunNotFoundError(f"run '{run_name}' does not exist")
        now = datetime.now(UTC).isoformat()
        with self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO baselines ("
                "baseline_name, run_name, created_at, metadata_json"
                ") VALUES (?, ?, ?, ?)",
                (
                    baseline_name,
                    run_name,
                    now,
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )

    def get_baseline(self, baseline_name: str) -> str | None:
        with self._session() as conn:
            row = conn.execute(
                "SELECT run_name FROM baselines WHERE baseline_name = ?",
                (baseline_name,),
            ).fetchone()
        return None if row is None else str(row["run_name"])


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_name=str(row["run_name"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        config_snapshot=json.loads(str(row["config_json"])),
        test_set_hash=str(row["test_set_hash"]),
        git_sha=None if row["git_sha"] is None else str(row["git_sha"]),
        case_count=int(row["case_count"]),
        completed_count=int(row["completed_count"]),
    )
