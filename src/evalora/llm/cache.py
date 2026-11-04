"""SQLite response cache for LLM calls."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def default_cache_path() -> Path:
    override = os.environ.get("EVALORA_CACHE_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".evalora" / "llm_cache.db"


def hash_payload(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else default_cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_SCHEMA)
            conn.commit()

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        try:
            yield conn
        finally:
            conn.close()

    def get(self, key: str) -> str | None:
        with self._session() as conn:
            row = conn.execute("SELECT payload FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def put(self, key: str, *, provider: str, model: str, payload: str) -> None:
        with self._session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, provider, model, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, provider, model, payload, datetime.now(UTC).isoformat()),
            )
            conn.commit()
