from __future__ import annotations

import pytest

from evalora.config import ResponseMapping
from evalora.run.adapter import ContractError, extract_path, parse_pipeline_response


def test_extract_path_supports_nested_keys_and_indices() -> None:
    payload = {"data": {"answer": "x"}, "items": [{"text": "a"}]}

    assert extract_path(payload, "$.data.answer") == "x"
    assert extract_path(payload, "items.0.text") == "a"


def test_parse_valid_response() -> None:
    payload = {
        "answer": "1994",
        "contexts": [
            {"id": "c1", "text": "TechMart was founded in 1994."},
            {"text": "The first store opened in March 1994."},
        ],
    }

    parsed = parse_pipeline_response(payload, ResponseMapping(), latency_ms=12.5)

    assert parsed.answer == "1994"
    assert parsed.latency_ms == 12.5
    assert parsed.contexts[0].id == "c1"
    assert parsed.contexts[1].id is None


def test_parse_nested_mapping() -> None:
    payload = {"data": {"answer": "x", "documents": [{"id": "c1", "text": "y"}]}}
    mapping = ResponseMapping(answer="data.answer", contexts="data.documents")

    parsed = parse_pipeline_response(payload, mapping, latency_ms=1.0)

    assert parsed.answer == "x"
    assert parsed.contexts[0].text == "y"


def test_empty_contexts_are_allowed() -> None:
    parsed = parse_pipeline_response(
        {"answer": "I don't know", "contexts": []},
        ResponseMapping(),
        latency_ms=1.0,
    )
    assert parsed.contexts == []


def test_missing_answer_path_raises() -> None:
    with pytest.raises(ContractError, match=r"answer.*not found"):
        parse_pipeline_response({"contexts": []}, ResponseMapping(), latency_ms=1.0)


def test_non_string_answer_raises() -> None:
    with pytest.raises(ContractError, match="string answer"):
        parse_pipeline_response(
            {"answer": 1994, "contexts": []},
            ResponseMapping(),
            latency_ms=1.0,
        )


def test_non_list_contexts_raises() -> None:
    with pytest.raises(ContractError, match="list of contexts"):
        parse_pipeline_response(
            {"answer": "x", "contexts": {"text": "y"}},
            ResponseMapping(),
            latency_ms=1.0,
        )


def test_context_must_be_an_object() -> None:
    with pytest.raises(ContractError, match="index 0 must be an object"):
        parse_pipeline_response(
            {"answer": "x", "contexts": ["y"]},
            ResponseMapping(),
            latency_ms=1.0,
        )


def test_context_must_have_string_text() -> None:
    with pytest.raises(ContractError, match="string 'text' field"):
        parse_pipeline_response(
            {"answer": "x", "contexts": [{"id": "c1"}]},
            ResponseMapping(),
            latency_ms=1.0,
        )


def test_context_id_must_be_a_string() -> None:
    with pytest.raises(ContractError, match="non-string 'id'"):
        parse_pipeline_response(
            {"answer": "x", "contexts": [{"id": 1, "text": "y"}]},
            ResponseMapping(),
            latency_ms=1.0,
        )


def test_empty_mapping_path_raises() -> None:
    with pytest.raises(ContractError, match="must not be empty"):
        extract_path({"answer": "x"}, "")
