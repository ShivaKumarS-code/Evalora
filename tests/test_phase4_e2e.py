"""End-to-end tests for the Phase 4 judge-validation workflow.

These exercise the two documented loops with synthetic but realistic data:

1. Manual loop: a seeded judged run -> ``export-labels`` -> simulated hand
   labelling of the exported sheet -> ``validate-judge`` -> hand-checked
   agreement metrics -> numbers embedded in the HTML report.
2. Public benchmark loop: RAGTruth-style fixtures -> ``ragtruth-sample`` ->
   ``validate-judge`` on answer-level items with the dataset auto-named.

The fake provider is scripted so the judge reproduces the verdicts stored in
the seeded run, which makes every agreement number hand-computable: the
deliberate judge/human disagreements produce a known confusion matrix.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from evalora.cli import app
from evalora.llm.base import LLMClient, Message
from evalora.llm.fake import FakeProvider, ResponseHandler
from evalora.schemas import ClaimVerdict, QuestionType, RetrievedContext
from evalora.store.db import RunStore
from evalora.validate.labels import LabelItem, LabelValue, write_labels
from tests.factories import make_result

runner = CliRunner()

_ITEMS_DATA_MARK = '<script type="application/json" id="items-data">'

# Twelve unique claims with the verdict the seeded judge recorded and the
# label a human applied. Four disagreements (two missed hallucinations, two
# over-flags) give a hand-computable confusion matrix: TP=4, FP=1, FN=2, TN=5.
CASE_CLAIMS: list[tuple[str, QuestionType, list[tuple[str, bool, LabelValue]]]] = [
    (
        "case-1",
        QuestionType.SINGLE_HOP,
        [
            ("TechMart was founded in 1994.", True, LabelValue.SUPPORTED),
            ("TechMart operates 500 stores.", True, LabelValue.UNSUPPORTED),
            ("TechMart was founded in 1984.", False, LabelValue.UNSUPPORTED),
        ],
    ),
    (
        "case-2",
        QuestionType.MULTI_HOP,
        [
            ("TechMart is headquartered in Portland.", True, LabelValue.SUPPORTED),
            ("TechMart ships to Canada.", False, LabelValue.UNSUPPORTED),
            ("TechMart sells fresh produce.", True, LabelValue.SUPPORTED),
        ],
    ),
    (
        "case-3",
        QuestionType.UNANSWERABLE,
        [
            ("TechMart merged with a rival chain.", False, LabelValue.SUPPORTED),
            ("TechMart filed for bankruptcy.", False, LabelValue.UNSUPPORTED),
            ("TechMart accepts crypto payments.", True, LabelValue.UNSUPPORTED),
        ],
    ),
    (
        "case-4",
        QuestionType.ADVERSARIAL,
        [
            ("TechMart prices match its rivals.", True, LabelValue.SUPPORTED),
            ("TechMart prices beat every rival.", False, LabelValue.UNSUPPORTED),
            ("TechMart was founded in Portland.", True, LabelValue.SUPPORTED),
        ],
    ),
]

EXPECTED_CONFUSION = {"tp": 4, "fp": 1, "fn": 2, "tn": 5}


def _judge_verdicts() -> dict[str, bool]:
    return {claim: supported for _, _, specs in CASE_CLAIMS for claim, supported, _ in specs}


def _hand_labels() -> dict[str, LabelValue]:
    return {claim: label for _, _, specs in CASE_CLAIMS for claim, _, label in specs}


def _seed_judged_run(db_path: Path) -> None:
    """Store a run with four cases and three judged claims per case."""
    store = RunStore(db_path)
    store.get_or_create_run(
        "e2e-run",
        config_snapshot={"gate": {"ci_confidence": 0.95}},
        test_set_hash="e2e-hash",
        case_count=len(CASE_CLAIMS),
    )
    for index, (case_id, question_type, specs) in enumerate(CASE_CLAIMS, start=1):
        store.save_result(
            "e2e-run",
            make_result(
                case_id,
                question_type,
                contexts=[RetrievedContext(id=f"c{index}", text="TechMart was founded in 1994.")],
                claim_verdicts=[
                    ClaimVerdict(
                        claim=claim,
                        supported=supported,
                        evidence_context_index=0 if supported else None,
                        rationale="stored judge verdict",
                    )
                    for claim, supported, _ in specs
                ],
            ),
        )


def _verification_handler(verdicts: Mapping[str, bool]) -> ResponseHandler:
    """Script the judge to reproduce the stored verdict for each claim."""

    def handler(messages: Sequence[Message], response_schema: type[Any] | None) -> str | Exception:
        user = messages[-1].content
        claim = user.split("Claim:\n", 1)[1].split("\n\nContext:", 1)[0]
        supported = verdicts[claim]
        return json.dumps(
            {
                "supported": supported,
                "evidence_context_index": 0 if supported else None,
                "rationale": "scripted judge verdict",
            }
        )

    return handler


def _read_sheet_items(html: str) -> list[dict[str, Any]]:
    """Extract the embedded items-data blob the labelling page downloads."""
    start = html.index(_ITEMS_DATA_MARK) + len(_ITEMS_DATA_MARK)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _patch_judge(monkeypatch: pytest.MonkeyPatch, handler: ResponseHandler) -> None:
    client = LLMClient(FakeProvider(handler=handler), max_retries=0)
    monkeypatch.setattr("evalora.cli.build_llm_client", lambda **kwargs: client)


def test_manual_label_workflow_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "runs.db"
    _seed_judged_run(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))

    # Step 1: export the labelling sheet from the judged run.
    sheet_path = tmp_path / "labels.html"
    exported = runner.invoke(
        app,
        ["export-labels", "--run", "e2e-run", "--out", str(sheet_path)],
    )
    assert exported.exit_code == 0, exported.output
    html = sheet_path.read_text(encoding="utf-8")
    assert "12 claim items" in html
    for case_id, _, specs in CASE_CLAIMS:
        assert f"case {case_id}" in html
        for claim, _, _ in specs:
            assert claim in html

    # Step 2: simulate the human labeller filling the sheet in the browser.
    sheet_items = _read_sheet_items(html)
    assert len(sheet_items) == 12
    labelled = []
    for raw in sheet_items:
        raw["label"] = _hand_labels()[raw["claim"]].value
        labelled.append(LabelItem.model_validate(raw))
    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, labelled)

    # Step 3: validate the judge against the hand labels.
    _patch_judge(monkeypatch, _verification_handler(_judge_verdicts()))
    validation_path = tmp_path / "judge_validation.json"
    validated = runner.invoke(
        app,
        [
            "validate-judge",
            "--labels",
            str(labels_path),
            "--out",
            str(validation_path),
            "--dataset",
            "manual",
            "--resamples",
            "200",
            "--seed",
            "7",
        ],
    )
    assert validated.exit_code == 0, validated.output
    assert "TP / FP / FN / TN" in validated.output
    assert "4 / 1 / 2 / 5" in validated.output

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["n"] == 12
    assert payload["dataset"] == "manual"
    assert payload["prompt_version"] == "v1"
    assert payload["confusion"] == EXPECTED_CONFUSION
    assert payload["precision"]["value"] == pytest.approx(0.8)
    assert payload["recall"]["value"] == pytest.approx(4 / 6)
    assert payload["f1"]["value"] == pytest.approx(8 / 11)
    assert payload["accuracy"]["value"] == pytest.approx(0.75)
    assert payload["kappa"]["value"] == pytest.approx(0.5)
    for name in ("precision", "recall", "f1", "accuracy", "kappa"):
        metric = payload[name]
        assert metric["ci_low"] <= metric["value"] <= metric["ci_high"]

    # Per-item results: the two false negatives and the one false positive are
    # exactly the deliberately mislabelled claims.
    items = {item["item_id"]: item for item in payload["items"]}
    assert len(items) == 12
    fn_one = items["case-1:claim:1"]
    assert fn_one["true_label"] == "unsupported"
    assert fn_one["predicted_label"] == "supported"
    assert fn_one["correct"] is False
    fn_two = items["case-3:claim:2"]
    assert fn_two["correct"] is False
    fp = items["case-3:claim:0"]
    assert fp["true_label"] == "supported"
    assert fp["predicted_label"] == "unsupported"
    assert fp["correct"] is False
    assert sum(1 for item in payload["items"] if item["correct"]) == 9

    # One judge call per claim, all fresh.
    assert payload["token_usage"]["calls"] == 12

    # Step 4: embed the measured agreement numbers in the HTML report.
    report_path = tmp_path / "report.html"
    reported = runner.invoke(
        app,
        [
            "report",
            "--run",
            "e2e-run",
            "--out",
            str(report_path),
            "--judge-validation",
            str(validation_path),
        ],
    )
    assert reported.exit_code == 0, reported.output
    report_html = report_path.read_text(encoding="utf-8")
    assert "Judge validation" in report_html
    assert "manual" in report_html
    assert "0.800" in report_html
    assert "0.500" in report_html


def test_validate_judge_records_prompt_version_for_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The judge-comparison pass: same labels, second prompt version recorded."""
    db_path = tmp_path / "runs.db"
    _seed_judged_run(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    sheet_path = tmp_path / "labels.html"
    exported = runner.invoke(
        app,
        ["export-labels", "--run", "e2e-run", "--out", str(sheet_path)],
    )
    assert exported.exit_code == 0, exported.output
    labelled = []
    for raw in _read_sheet_items(sheet_path.read_text(encoding="utf-8")):
        raw["label"] = _hand_labels()[raw["claim"]].value
        labelled.append(LabelItem.model_validate(raw))
    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, labelled)

    _patch_judge(monkeypatch, _verification_handler(_judge_verdicts()))
    validation_path = tmp_path / "judge_validation_v2.json"
    validated = runner.invoke(
        app,
        [
            "validate-judge",
            "--labels",
            str(labels_path),
            "--out",
            str(validation_path),
            "--prompt-version",
            "v2",
            "--resamples",
            "200",
        ],
    )
    assert validated.exit_code == 0, validated.output

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["prompt_version"] == "v2"
    assert payload["confusion"] == EXPECTED_CONFUSION


def test_validate_judge_rejects_unlabelled_sheet_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running validate-judge on a sheet export nobody labelled fails loudly."""
    db_path = tmp_path / "runs.db"
    _seed_judged_run(db_path)
    monkeypatch.setenv("EVALORA_DB_PATH", str(db_path))
    sheet_path = tmp_path / "labels.html"
    exported = runner.invoke(
        app,
        ["export-labels", "--run", "e2e-run", "--out", str(sheet_path)],
    )
    assert exported.exit_code == 0, exported.output

    unlabelled = [
        LabelItem.model_validate(raw)
        for raw in _read_sheet_items(sheet_path.read_text(encoding="utf-8"))
    ]
    labels_path = tmp_path / "labels.jsonl"
    write_labels(labels_path, unlabelled)

    validated = runner.invoke(
        app,
        ["validate-judge", "--labels", str(labels_path)],
    )

    assert validated.exit_code == 1
    assert "missing a label" in validated.output


# RAGTruth-style fixtures: eight responses, four eligible (test split, good
# quality, known source, non-empty), plus one of each exclusion reason.
RAGTRUTH_RESPONSES = [
    {
        "id": "1",
        "source_id": "s1",
        "model": "chatgpt",
        "temperature": 0.0,
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "TechMart was founded in 1994.",
    },
    {
        "id": "2",
        "source_id": "s1",
        "model": "chatgpt",
        "temperature": 0.0,
        "labels": [{"start": 0, "end": 25, "text": "founded in 1984"}],
        "split": "test",
        "quality": "good",
        "response": "TechMart was founded in 1984.",
    },
    {
        "id": "3",
        "source_id": "s2",
        "model": "chatgpt",
        "temperature": 0.0,
        "labels": [{"start": 0, "end": 11, "text": "is bankrupt"}],
        "split": "test",
        "quality": "good",
        "response": "TechMart is bankrupt.",
    },
    {
        "id": "4",
        "source_id": "s3",
        "model": "chatgpt",
        "temperature": 0.0,
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "The article discusses retail trends.",
    },
    {
        "id": "5",
        "source_id": "s1",
        "labels": [],
        "split": "train",
        "quality": "good",
        "response": "Train split.",
    },
    {
        "id": "6",
        "source_id": "s1",
        "labels": [],
        "split": "test",
        "quality": "incorrect_refusal",
        "response": "I cannot answer.",
    },
    {
        "id": "7",
        "source_id": "missing",
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "No source.",
    },
    {
        "id": "8",
        "source_id": "s3",
        "labels": [],
        "split": "test",
        "quality": "good",
        "response": "   ",
    },
]

RAGTRUTH_SOURCES = [
    {
        "source_id": "s1",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "When was TechMart founded?",
            "passages": (
                "passage 1:TechMart was founded in 1994.\n\n"
                "passage 2:It is headquartered in Portland."
            ),
        },
        "prompt": "prompt",
    },
    {
        "source_id": "s2",
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "Is TechMart bankrupt?",
            "passages": "passage 1:TechMart is a healthy chain.",
        },
        "prompt": "prompt",
    },
    {
        "source_id": "s3",
        "task_type": "Summary",
        "source": "CNN/DM",
        "source_info": "A long article body about retail.",
        "prompt": "prompt",
    },
]

# Scripted judge behaviour on the sampled answers: one missed hallucination
# ("is bankrupt" judged supported) and one over-flagged faithful summary,
# giving TP=1, FP=1, FN=1, TN=1.
ANSWER_CLAIMS: dict[str, list[str]] = {
    "TechMart was founded in 1994.": ["TechMart was founded in 1994."],
    "TechMart was founded in 1984.": ["TechMart was founded in 1984."],
    "TechMart is bankrupt.": ["TechMart is bankrupt."],
    "The article discusses retail trends.": ["The article discusses retail trends."],
}

ANSWER_CLAIM_VERDICTS = {
    "TechMart was founded in 1994.": True,
    "TechMart was founded in 1984.": False,
    "TechMart is bankrupt.": True,
    "The article discusses retail trends.": False,
}


def _write_ragtruth_files(tmp_path: Path) -> tuple[Path, Path]:
    responses_path = tmp_path / "response.jsonl"
    sources_path = tmp_path / "source_info.jsonl"
    responses_path.write_text(
        "\n".join(json.dumps(record) for record in RAGTRUTH_RESPONSES) + "\n",
        encoding="utf-8",
    )
    sources_path.write_text(
        "\n".join(json.dumps(record) for record in RAGTRUTH_SOURCES) + "\n",
        encoding="utf-8",
    )
    return responses_path, sources_path


def _answer_level_handler() -> ResponseHandler:
    """Script extraction (one claim per answer) and per-claim verification."""

    def handler(messages: Sequence[Message], response_schema: type[Any] | None) -> str | Exception:
        user = messages[-1].content
        if user.startswith("Split the answer"):
            answer = user.rsplit("Answer:\n", 1)[1]
            return json.dumps({"claims": ANSWER_CLAIMS[answer]})
        claim = user.split("Claim:\n", 1)[1].split("\n\nContext:", 1)[0]
        supported = ANSWER_CLAIM_VERDICTS[claim]
        return json.dumps(
            {
                "supported": supported,
                "evidence_context_index": 0 if supported else None,
                "rationale": "scripted judge verdict",
            }
        )

    return handler


def test_ragtruth_benchmark_workflow_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses_path, sources_path = _write_ragtruth_files(tmp_path)

    # Step 1: draw the labelled sample with a fixed seed.
    sample_path = tmp_path / "ragtruth_sample.jsonl"
    manifest_path = tmp_path / "ragtruth_manifest.json"
    sampled = runner.invoke(
        app,
        [
            "ragtruth-sample",
            "--responses",
            str(responses_path),
            "--sources",
            str(sources_path),
            "--out",
            str(sample_path),
            "--manifest",
            str(manifest_path),
            "--sample-size",
            "4",
            "--seed",
            "7",
        ],
    )
    assert sampled.exit_code == 0, sampled.output
    assert "hallucination=2, supported=2, eligible=4" in sampled.output

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["license"] == "MIT"
    assert manifest["split"] == "test"
    assert manifest["sample_size"] == 4
    assert manifest["sample_seed"] == 7
    assert manifest["hallucination_items"] == 2
    assert manifest["supported_items"] == 2
    assert manifest["task_counts"] == {"QA": 3, "Summary": 1}

    # Sampling is byte-for-byte reproducible.
    repeat_path = tmp_path / "ragtruth_sample_repeat.jsonl"
    resampled = runner.invoke(
        app,
        [
            "ragtruth-sample",
            "--responses",
            str(responses_path),
            "--sources",
            str(sources_path),
            "--out",
            str(repeat_path),
            "--manifest",
            str(tmp_path / "ragtruth_manifest_repeat.json"),
            "--sample-size",
            "4",
            "--seed",
            "7",
        ],
    )
    assert resampled.exit_code == 0, resampled.output
    assert repeat_path.read_bytes() == sample_path.read_bytes()

    # Step 2: validate the judge on the public benchmark sample. Items carry
    # answer-level data, so the judge runs extraction then verification.
    _patch_judge(monkeypatch, _answer_level_handler())
    validation_path = tmp_path / "ragtruth_validation.json"
    validated = runner.invoke(
        app,
        [
            "validate-judge",
            "--labels",
            str(sample_path),
            "--out",
            str(validation_path),
            "--resamples",
            "200",
        ],
    )
    assert validated.exit_code == 0, validated.output

    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["n"] == 4
    # Dataset auto-named from the item source, seed and size from metadata.
    assert payload["dataset"] == "RAGTruth"
    assert payload["sample_seed"] == 7
    assert payload["sample_size"] == 4
    assert payload["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert payload["precision"]["value"] == pytest.approx(0.5)
    assert payload["recall"]["value"] == pytest.approx(0.5)
    assert payload["f1"]["value"] == pytest.approx(0.5)
    assert payload["accuracy"]["value"] == pytest.approx(0.5)
    assert payload["kappa"]["value"] == pytest.approx(0.0)
    # One extraction plus one verification per item.
    assert payload["token_usage"]["calls"] == 8
