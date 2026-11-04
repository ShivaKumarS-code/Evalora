from __future__ import annotations

import hashlib
import itertools

import pytest

from evalora.ingest.chunker import chunk_documents, make_chunk_id
from evalora.ingest.loaders import Document


def test_chunk_ids_are_stable_and_match_formula() -> None:
    document = Document(
        doc_id="doc.md",
        text="First paragraph with a fact.\n\nSecond paragraph with another fact.",
    )

    first = chunk_documents([document], chunk_size=100, chunk_overlap=10)
    second = chunk_documents([document], chunk_size=100, chunk_overlap=10)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    for chunk in first:
        expected = hashlib.sha1(f"{chunk.doc_id}{chunk.ordinal}{chunk.text}".encode()).hexdigest()[
            :12
        ]
        assert chunk.chunk_id == expected
        assert chunk.chunk_id == make_chunk_id(chunk.doc_id, chunk.ordinal, chunk.text)


def test_chunk_ids_change_with_content() -> None:
    assert make_chunk_id("doc.md", 0, "one") != make_chunk_id("doc.md", 0, "two")


def test_chunks_respect_size_and_overlap() -> None:
    text = " ".join(f"Sentence number {index} talks about topic {index}." for index in range(40))
    document = Document(doc_id="doc.md", text=text)

    chunks = chunk_documents([document], chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    for previous, current in itertools.pairwise(chunks):
        assert previous.text[-20:] in current.text


def test_sentence_boundaries_are_respected() -> None:
    text = " ".join(f"Sentence number {index} talks about topic {index}." for index in range(20))
    document = Document(doc_id="doc.md", text=text)

    chunks = chunk_documents([document], chunk_size=120, chunk_overlap=0)

    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_ordinals_continue_across_pages() -> None:
    documents = [
        Document(doc_id="doc.pdf", text="Page one sentence. " * 30, page=1),
        Document(doc_id="doc.pdf", text="Page two sentence. " * 30, page=2),
    ]

    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=10)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page == 1
    assert chunks[-1].page == 2


def test_long_sentence_is_hard_split() -> None:
    document = Document(doc_id="doc.md", text="x" * 250)

    chunks = chunk_documents([document], chunk_size=100, chunk_overlap=0)

    assert [len(chunk.text) for chunk in chunks] == [100, 100, 50]


def test_invalid_settings_raise() -> None:
    document = Document(doc_id="doc.md", text="text")
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([document], chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_documents([document], chunk_size=100, chunk_overlap=-1)
    with pytest.raises(ValueError, match="smaller"):
        chunk_documents([document], chunk_size=100, chunk_overlap=100)
