"""Baseline export and import as portable JSON artifacts.

CI keeps a baseline as an artifact or commits the JSON file; importing it
recreates the run and its results in the local store and points a named
baseline alias at it, so ``evalora gate`` can compare against it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from evalora.schemas import CaseResult
from evalora.store.db import RunRecord, RunStore, StoreError

BASELINE_FORMAT_VERSION = 1


class BaselineFileError(Exception):
    """Raised when a baseline JSON file is missing or invalid."""


class BaselineExport(BaseModel):
    format_version: int = BASELINE_FORMAT_VERSION
    run_name: str
    test_set_hash: str
    git_sha: str | None
    config_snapshot: dict[str, object] = Field(default_factory=dict)
    case_count: int
    completed_count: int
    created_at: str
    exported_at: str
    results: list[CaseResult] = Field(default_factory=list)


def export_baseline(store: RunStore, run_name: str) -> BaselineExport:
    """Snapshot a stored run as a portable baseline artifact."""
    run = store.get_run(run_name)
    if run is None:
        raise StoreError(f"run '{run_name}' not found")
    results = store.load_results(run_name)
    return BaselineExport(
        run_name=run.run_name,
        test_set_hash=run.test_set_hash,
        git_sha=run.git_sha,
        config_snapshot=run.config_snapshot,
        case_count=run.case_count,
        completed_count=len(results),
        created_at=run.created_at.isoformat(),
        exported_at=datetime.now(UTC).isoformat(),
        results=results,
    )


def write_baseline_file(export: BaselineExport, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(export.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path


def read_baseline_file(path: str | Path) -> BaselineExport:
    baseline_path = Path(path)
    if not baseline_path.is_file():
        raise BaselineFileError(f"Baseline file not found: {baseline_path}")
    try:
        export = BaselineExport.model_validate_json(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise BaselineFileError(f"Invalid baseline file {baseline_path}: {exc}") from exc
    if export.format_version != BASELINE_FORMAT_VERSION:
        raise BaselineFileError(
            f"Unsupported baseline format version {export.format_version}; "
            f"this tool writes version {BASELINE_FORMAT_VERSION}"
        )
    if not export.results:
        raise BaselineFileError(f"Baseline file {baseline_path} contains no results")
    return export


def import_baseline(
    store: RunStore,
    export: BaselineExport,
    *,
    baseline_name: str,
) -> RunRecord:
    """Recreate an exported run locally and register it as a named baseline.

    Importing is idempotent: if the run already exists with the same test set
    hash, results are replaced in place; a hash mismatch raises the store's
    mismatch error rather than silently mixing test sets.
    """
    record = store.get_or_create_run(
        export.run_name,
        config_snapshot=dict(export.config_snapshot),
        test_set_hash=export.test_set_hash,
        case_count=max(export.case_count, len(export.results)),
        git_sha=export.git_sha,
    )
    for result in export.results:
        store.save_result(export.run_name, result)
    store.set_baseline(
        baseline_name,
        export.run_name,
        metadata={
            "imported_at": datetime.now(UTC).isoformat(),
            "exported_at": export.exported_at,
            "results": len(export.results),
        },
    )
    return record
