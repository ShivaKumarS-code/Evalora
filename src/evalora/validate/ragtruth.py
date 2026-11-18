"""RAGTruth loading and reproducible sampling.

RAGTruth is MIT-licensed: https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from evalora.schemas import RetrievedContext
from evalora.validate.labels import LabelItem, LabelValue

RAGTRUTH_LICENSE = "MIT"
RAGTRUTH_LICENSE_URL = "https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE"
RAGTRUTH_REPOSITORY_URL = "https://github.com/ParticleMedia/RAGTruth"

_PASSAGE_RE = re.compile(r"(?=passage\s+\d+\s*:)", re.IGNORECASE)
T = TypeVar("T", bound=BaseModel)


class RagTruthError(Exception):
    """Raised when RAGTruth data cannot be read or sampled."""


class RagTruthResponse(BaseModel):
    id: str
    source_id: str
    model: str | None = None
    temperature: float | None = None
    labels: list[dict[str, Any]] = Field(default_factory=list)
    split: str | None = None
    quality: str | None = None
    response: str


class RagTruthSource(BaseModel):
    source_id: str
    task_type: str | None = None
    source: str | None = None
    source_info: Any = None
    prompt: str | None = None


class RagTruthSampleManifest(BaseModel):
    dataset: str = "RAGTruth"
    license: str = RAGTRUTH_LICENSE
    license_url: str = RAGTRUTH_LICENSE_URL
    repository_url: str = RAGTRUTH_REPOSITORY_URL
    split: str | None
    sample_size: int
    sample_seed: int
    eligible_count: int
    hallucination_items: int
    supported_items: int
    task_counts: dict[str, int]
    created_at: str


def _read_jsonl(path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        raise RagTruthError(f"RAGTruth file not found: {path}")
    records: list[T] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise RagTruthError(
                    f"{path}:{line_number}: invalid RAGTruth record: {exc}"
                ) from exc
    if not records:
        raise RagTruthError(f"RAGTruth file is empty: {path}")
    return records


def load_ragtruth(
    responses_path: str | Path,
    sources_path: str | Path,
) -> tuple[list[RagTruthResponse], dict[str, RagTruthSource]]:
    responses = _read_jsonl(Path(responses_path), RagTruthResponse)
    sources = _read_jsonl(Path(sources_path), RagTruthSource)
    source_map: dict[str, RagTruthSource] = {}
    for source in sources:
        if source.source_id in source_map:
            raise RagTruthError(f"duplicate RAGTruth source_id: {source.source_id}")
        source_map[source.source_id] = source
    return responses, source_map


def _contexts_for(source: RagTruthSource) -> list[RetrievedContext]:
    info = source.source_info
    if source.task_type == "QA" and isinstance(info, dict):
        passages = info.get("passages")
        if isinstance(passages, str):
            parts = [part.strip() for part in _PASSAGE_RE.split(passages) if part.strip()]
            if len(parts) > 1:
                return [
                    RetrievedContext(
                        id=f"{source.source_id}#passage-{index}",
                        text=part,
                    )
                    for index, part in enumerate(parts, start=1)
                ]
            return [RetrievedContext(id=source.source_id, text=passages)]
    if isinstance(info, str):
        return [RetrievedContext(id=source.source_id, text=info)]
    return [
        RetrievedContext(
            id=source.source_id,
            text=json.dumps(info, ensure_ascii=False, sort_keys=True),
        )
    ]


def _question_for(source: RagTruthSource) -> str:
    info = source.source_info
    if isinstance(info, dict):
        question = info.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    return source.task_type or source.source or "RAGTruth item"


def _build_item(
    response: RagTruthResponse,
    source: RagTruthSource,
    *,
    sample_size: int,
    sample_seed: int,
) -> LabelItem:
    implicit_true = sum(1 for label in response.labels if bool(label.get("implicit_true")))
    return LabelItem(
        item_id=f"ragtruth:{response.id}",
        case_id=None,
        question=_question_for(source),
        contexts=_contexts_for(source),
        answer=response.response,
        claim=None,
        label=(LabelValue.UNSUPPORTED if response.labels else LabelValue.SUPPORTED),
        source="RAGTruth",
        metadata={
            "response_id": response.id,
            "source_id": response.source_id,
            "task_type": source.task_type,
            "model": response.model,
            "temperature": response.temperature,
            "split": response.split,
            "quality": response.quality,
            "label_count": len(response.labels),
            "implicit_true_labels": implicit_true,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "license": RAGTRUTH_LICENSE,
        },
    )


def build_ragtruth_sample(
    responses: Sequence[RagTruthResponse],
    sources: dict[str, RagTruthSource],
    *,
    sample_size: int = 250,
    seed: int = 42,
    split: str | None = "test",
    require_quality_good: bool = True,
) -> tuple[list[LabelItem], RagTruthSampleManifest]:
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1")
    eligible = [
        response
        for response in responses
        if (split is None or response.split == split)
        and (not require_quality_good or response.quality == "good")
        and response.source_id in sources
        and response.response.strip()
    ]
    if len(eligible) < sample_size:
        raise RagTruthError(
            f"only {len(eligible)} eligible RAGTruth responses, requested {sample_size}"
        )

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(eligible)), sample_size))
    selected = [eligible[index] for index in selected_indices]
    items = [
        _build_item(
            response,
            sources[response.source_id],
            sample_size=sample_size,
            sample_seed=seed,
        )
        for response in selected
    ]
    hallucination_items = sum(1 for item in items if item.label == LabelValue.UNSUPPORTED)
    task_counts: dict[str, int] = {}
    for item in items:
        task_type = str(item.metadata.get("task_type") or "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    manifest = RagTruthSampleManifest(
        split=split,
        sample_size=sample_size,
        sample_seed=seed,
        eligible_count=len(eligible),
        hallucination_items=hallucination_items,
        supported_items=sample_size - hallucination_items,
        task_counts=task_counts,
        created_at=datetime.now(UTC).isoformat(),
    )
    return items, manifest


def write_manifest(manifest: RagTruthSampleManifest, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return output_path
