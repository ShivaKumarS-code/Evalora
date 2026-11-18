from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalora.validate.labels import LabelValue
from evalora.validate.ragtruth import (
    RagTruthError,
    build_ragtruth_sample,
    load_ragtruth,
    write_manifest,
)

RESPONSES = [
    {
        "id": "1",
        "source_id": "s1",
        "model": "m",
        "temperature": 0.0,
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "Answer one",
    },
    {
        "id": "2",
        "source_id": "s1",
        "model": "m",
        "temperature": 0.0,
        "labels": [{"start": 0, "end": 3, "text": "Bad", "label_type": "x"}],
        "split": "test",
        "quality": "good",
        "response": "Bad answer",
    },
    {
        "id": "3",
        "source_id": "s2",
        "model": "m",
        "labels": [],
        "split": "train",
        "quality": "good",
        "response": "Train answer",
    },
    {
        "id": "4",
        "source_id": "s2",
        "model": "m",
        "labels": [],
        "split": "test",
        "quality": "incorrect_refusal",
        "response": "Refusal",
    },
    {
        "id": "5",
        "source_id": "s3",
        "model": "m",
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "Summary answer",
    },
    {
        "id": "6",
        "source_id": "missing",
        "model": "m",
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "No source",
    },
]
SOURCES = [
    {
        "source_id": "s1",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "When was TechMart founded?",
            "passages": "passage 1:TechMart was founded in 1994.\n\npassage 2:It is in Portland.",
        },
        "prompt": "prompt",
    },
    {
        "source_id": "s2",
        "task_type": "Data2txt",
        "source": "Yelp",
        "source_info": {"name": "Subway", "city": "Santa Barbara"},
        "prompt": "prompt",
    },
    {
        "source_id": "s3",
        "task_type": "Summary",
        "source": "CNN/DM",
        "source_info": "A long article body.",
        "prompt": "prompt",
    },
]


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_jsonl(tmp_path / "response.jsonl", RESPONSES),
        _write_jsonl(tmp_path / "source_info.jsonl", SOURCES),
    )


def test_load_ragtruth(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)

    responses, sources = load_ragtruth(responses_path, sources_path)

    assert len(responses) == 6
    assert set(sources) == {"s1", "s2", "s3"}
    assert sources["s1"].task_type == "QA"


def test_build_sample_filters_and_labels(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)
    responses, sources = load_ragtruth(responses_path, sources_path)

    items, manifest = build_ragtruth_sample(
        responses,
        sources,
        sample_size=3,
        seed=7,
        split="test",
    )

    assert {item.item_id for item in items} == {"ragtruth:1", "ragtruth:2", "ragtruth:5"}
    assert manifest.eligible_count == 3
    assert manifest.sample_size == 3
    assert manifest.sample_seed == 7
    assert manifest.hallucination_items == 1
    assert manifest.supported_items == 2
    assert manifest.task_counts == {"QA": 2, "Summary": 1}
    assert manifest.license == "MIT"
    qa_item = next(item for item in items if item.item_id == "ragtruth:2")
    assert qa_item.label == LabelValue.UNSUPPORTED
    assert len(qa_item.contexts) == 2
    assert qa_item.metadata["sample_seed"] == 7


def test_build_sample_is_reproducible(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)
    responses, sources = load_ragtruth(responses_path, sources_path)

    first, _ = build_ragtruth_sample(responses, sources, sample_size=2, seed=7)
    second, _ = build_ragtruth_sample(responses, sources, sample_size=2, seed=7)

    assert [item.item_id for item in first] == [item.item_id for item in second]


def test_data2txt_context_is_serialized(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)
    responses, sources = load_ragtruth(responses_path, sources_path)

    items, _ = build_ragtruth_sample(
        responses,
        sources,
        sample_size=1,
        seed=7,
        split="train",
    )

    assert items[0].item_id == "ragtruth:3"
    assert '"name": "Subway"' in items[0].contexts[0].text


def test_too_small_dataset_raises(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)
    responses, sources = load_ragtruth(responses_path, sources_path)

    with pytest.raises(RagTruthError, match="eligible"):
        build_ragtruth_sample(responses, sources, sample_size=10, seed=7)


def test_invalid_jsonl_raises(tmp_path: Path) -> None:
    responses_path = tmp_path / "response.jsonl"
    sources_path = tmp_path / "source_info.jsonl"
    responses_path.write_text("not json\n", encoding="utf-8")
    _write_jsonl(sources_path, SOURCES)

    with pytest.raises(RagTruthError, match="invalid RAGTruth record"):
        load_ragtruth(responses_path, sources_path)


def test_write_manifest(tmp_path: Path) -> None:
    responses_path, sources_path = _fixtures(tmp_path)
    responses, sources = load_ragtruth(responses_path, sources_path)
    _, manifest = build_ragtruth_sample(
        responses,
        sources,
        sample_size=2,
        seed=7,
    )
    path = tmp_path / "manifest.json"

    built = write_manifest(manifest, path)

    assert built.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["sample_seed"] == 7
