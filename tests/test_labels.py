from __future__ import annotations

from pathlib import Path

import pytest

from evalora.schemas import ClaimVerdict, RetrievedContext
from evalora.validate.labels import (
    LabelExportError,
    LabelFileError,
    LabelItem,
    LabelValue,
    build_label_sheet,
    collect_claim_items,
    read_labels,
    write_label_sheet,
    write_labels,
)
from tests.factories import make_result


def _result(case_id: str, claims: list[str]) -> object:
    return make_result(
        case_id,
        contexts=[RetrievedContext(id="c1", text="TechMart was founded in 1994.")],
        claim_verdicts=[
            ClaimVerdict(
                claim=claim,
                supported=True,
                evidence_context_index=0,
                rationale="test",
            )
            for claim in claims
        ],
    )


def test_collect_claim_items_creates_one_item_per_claim() -> None:
    items = collect_claim_items(
        [_result("case-1", ["TechMart was founded in 1994.", "It is in Portland."])]
    )

    assert [item.item_id for item in items] == ["case-1:claim:0", "case-1:claim:1"]
    assert all(item.label is None for item in items)
    assert all(item.claim for item in items)
    assert items[0].contexts[0].id == "c1"


def test_build_label_sheet_samples_reproducibly() -> None:
    results = [_result(f"case-{index}", [f"claim {index}"]) for index in range(5)]

    first = build_label_sheet(results, run_name="run", max_items=2, seed=7)
    second = build_label_sheet(results, run_name="run", max_items=2, seed=7)

    assert first.sampled is True
    assert first.seed == 7
    assert first.available_count == 5
    assert len(first.items) == 2
    assert [item.item_id for item in first.items] == [item.item_id for item in second.items]


def test_build_label_sheet_requires_claims() -> None:
    with pytest.raises(LabelExportError, match="no extracted claims"):
        build_label_sheet([make_result("case-1")], run_name="run")


def test_label_file_roundtrip(tmp_path: Path) -> None:
    item = LabelItem(
        item_id="item-1",
        contexts=[RetrievedContext(id="c1", text="context")],
        answer="answer",
        claim="claim",
        label=LabelValue.SUPPORTED,
    )
    path = tmp_path / "labels.jsonl"

    write_labels(path, [item])
    loaded = read_labels(path)

    assert loaded == [item]


def test_unlabelled_items_are_rejected(tmp_path: Path) -> None:
    item = LabelItem(item_id="item-1", answer="answer", contexts=[])
    path = tmp_path / "labels.jsonl"
    write_labels(path, [item])

    assert read_labels(path, require_labels=False) == [item]
    with pytest.raises(LabelFileError, match="missing a label"):
        read_labels(path)


def test_duplicate_item_ids_are_rejected(tmp_path: Path) -> None:
    item = LabelItem(
        item_id="item-1",
        answer="answer",
        contexts=[],
        label=LabelValue.SUPPORTED,
    )
    path = tmp_path / "labels.jsonl"
    write_labels(path, [item, item])

    with pytest.raises(LabelFileError, match="duplicate item_id"):
        read_labels(path)


def test_label_sheet_html_is_self_contained(tmp_path: Path) -> None:
    sheet = build_label_sheet(
        [_result("case-1", ["TechMart was founded in 1994."])],
        run_name="run-1",
    )
    path = tmp_path / "labels.html"

    write_label_sheet(sheet, path)

    html = path.read_text(encoding="utf-8")
    assert "<style>" in html
    assert "<script>" in html
    assert "<script src=" not in html
    assert "<link " not in html
    assert "TechMart was founded in 1994." in html
    assert "items-data" in html
    assert "Download labels.jsonl" in html
