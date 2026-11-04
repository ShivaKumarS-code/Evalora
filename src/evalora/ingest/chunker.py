"""Boundary-aware text chunking with stable chunk IDs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from evalora.ingest.loaders import Document
from evalora.schemas import Chunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class _Segment:
    text: str
    glue: str


def make_chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    payload = f"{doc_id}{ordinal}{text}".encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def chunk_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be zero or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    next_ordinal: dict[str, int] = {}
    for document in documents:
        text = _normalize(document.text)
        if not text:
            continue
        ordinal = next_ordinal.get(document.doc_id, 0)
        for piece in _pack(_segments(text, chunk_size), chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.doc_id, ordinal, piece),
                    doc_id=document.doc_id,
                    ordinal=ordinal,
                    text=piece,
                    page=document.page,
                )
            )
            ordinal += 1
        next_ordinal[document.doc_id] = ordinal
    return chunks


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _segments(text: str, chunk_size: int) -> Iterator[_Segment]:
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            yield _Segment(paragraph, "\n\n")
            continue
        sentences = [
            sentence.strip() for sentence in _SENTENCE_SPLIT.split(paragraph) if sentence.strip()
        ]
        for index, sentence in enumerate(sentences):
            glue = "\n\n" if index == 0 else " "
            if len(sentence) <= chunk_size:
                yield _Segment(sentence, glue)
                continue
            for piece_index, piece in enumerate(_hard_split(sentence, chunk_size)):
                yield _Segment(piece, glue if piece_index == 0 else "")


def _hard_split(text: str, chunk_size: int) -> Iterator[str]:
    for start in range(0, len(text), chunk_size):
        piece = text[start : start + chunk_size].strip()
        if piece:
            yield piece


def _pack(
    segments: Iterator[_Segment],
    chunk_size: int,
    chunk_overlap: int,
) -> Iterator[str]:
    current = ""
    for segment in segments:
        candidate = segment.text if not current else f"{current}{segment.glue}{segment.text}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            yield current.strip()
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            allowed_tail = chunk_size - len(segment.text) - len(segment.glue)
            tail = tail[-allowed_tail:] if allowed_tail > 0 else ""
            current = f"{tail}{segment.glue}{segment.text}" if tail else segment.text
        else:
            current = segment.text
    if current.strip():
        yield current.strip()
