"""Document loaders for markdown, text, and PDF files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".pdf"})


class DocumentLoadError(Exception):
    """Raised when configured document paths cannot be read."""


class Document(BaseModel):
    doc_id: str
    text: str
    page: int | None = None


def load_documents(paths: Sequence[str | Path]) -> list[Document]:
    if not paths:
        raise DocumentLoadError("No document paths configured")
    files = _collect_files(paths)
    if not files:
        raise DocumentLoadError(
            "No .md, .txt or .pdf files found in: " + ", ".join(str(path) for path in paths)
        )
    documents: list[Document] = []
    for file_path in files:
        documents.extend(_load_file(file_path))
    if not documents:
        raise DocumentLoadError("Documents were found but none contained extractable text")
    return documents


def _collect_files(paths: Sequence[str | Path]) -> list[Path]:
    files: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise DocumentLoadError(f"Document path not found: {path}")
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                files[path.resolve().as_posix()] = path
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                files[candidate.resolve().as_posix()] = candidate
    return sorted(files.values(), key=_doc_id)


def _doc_id(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load_file(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DocumentLoadError(f"Could not read {path}: {exc}") from exc
    if not text.strip():
        return []
    return [Document(doc_id=_doc_id(path), text=text)]


def _load_pdf(path: Path) -> list[Document]:
    doc_id = _doc_id(path)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise DocumentLoadError(f"Could not read PDF {path}: {exc}") from exc
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not extract text from {path} page {page_number}: {exc}"
            ) from exc
        if text.strip():
            documents.append(Document(doc_id=doc_id, text=text, page=page_number))
    return documents
