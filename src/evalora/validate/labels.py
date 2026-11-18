"""Label file IO and HTML labelling-sheet export."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, ValidationError

from evalora.schemas import CaseResult, RetrievedContext

_TEMPLATE_DIR = Path(__file__).parent / "templates"


class LabelFileError(Exception):
    """Raised when a label file is missing, invalid, or incomplete."""


class LabelExportError(Exception):
    """Raised when a run cannot produce a claim-level labelling sheet."""


class LabelValue(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class LabelItem(BaseModel):
    item_id: str
    case_id: str | None = None
    question: str | None = None
    contexts: list[RetrievedContext] = Field(default_factory=list)
    answer: str | None = None
    claim: str | None = None
    label: LabelValue | None = None
    source: str = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabelSheet(BaseModel):
    run_name: str
    items: list[LabelItem]
    available_count: int
    sampled: bool
    seed: int | None = None


def read_labels(path: str | Path, *, require_labels: bool = True) -> list[LabelItem]:
    label_path = Path(path)
    if not label_path.is_file():
        raise LabelFileError(f"Label file not found: {label_path}")
    items: list[LabelItem] = []
    seen_ids: set[str] = set()
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = LabelItem.model_validate_json(line)
            except ValidationError as exc:
                raise LabelFileError(
                    f"{label_path}:{line_number}: invalid label item: {exc}"
                ) from exc
            if item.item_id in seen_ids:
                raise LabelFileError(
                    f"{label_path}:{line_number}: duplicate item_id '{item.item_id}'"
                )
            seen_ids.add(item.item_id)
            items.append(item)
    if not items:
        raise LabelFileError(f"Label file is empty: {label_path}")
    if require_labels:
        missing = [item.item_id for item in items if item.label is None]
        if missing:
            preview = ", ".join(missing[:5])
            raise LabelFileError(f"{len(missing)} label item(s) are missing a label: {preview}")
    return items


def write_labels(path: str | Path, items: Sequence[LabelItem]) -> None:
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    item.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )


def collect_claim_items(results: Sequence[CaseResult]) -> list[LabelItem]:
    items: list[LabelItem] = []
    for result in results:
        for index, verdict in enumerate(result.claim_verdicts):
            claim = verdict.claim.strip()
            if not claim:
                continue
            items.append(
                LabelItem(
                    item_id=f"{result.case.case_id}:claim:{index}",
                    case_id=result.case.case_id,
                    question=result.case.question,
                    contexts=list(result.response.contexts),
                    answer=result.response.answer,
                    claim=claim,
                    label=None,
                    source="manual",
                    metadata={"case_type": result.case.type.value},
                )
            )
    return items


def build_label_sheet(
    results: Sequence[CaseResult],
    *,
    run_name: str,
    max_items: int = 150,
    seed: int = 42,
) -> LabelSheet:
    if max_items < 1:
        raise ValueError("max_items must be at least 1")
    items = collect_claim_items(results)
    if not items:
        raise LabelExportError(
            "the run contains no extracted claims; run it with judging enabled first"
        )
    available_count = len(items)
    sampled = available_count > max_items
    if sampled:
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(range(available_count), max_items))
        items = [items[index] for index in selected_indices]
    return LabelSheet(
        run_name=run_name,
        items=items,
        available_count=available_count,
        sampled=sampled,
        seed=seed if sampled else None,
    )


def write_label_sheet(sheet: LabelSheet, path: str | Path) -> Path:
    output_path = Path(path)
    environment = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    template = environment.get_template("labels.html.j2")
    items_data = [item.model_dump(mode="json") for item in sheet.items]
    html = template.render(
        sheet=sheet,
        items_data=items_data,
        item_count=len(sheet.items),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
