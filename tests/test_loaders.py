from __future__ import annotations

from pathlib import Path

import pytest

from evalora.ingest import loaders
from evalora.ingest.loaders import DocumentLoadError, load_documents


def test_loads_supported_files_recursively_in_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "b.txt").write_text("Beta text", encoding="utf-8")
    (docs / "a.md").write_text("# Alpha\n\nAlpha text", encoding="utf-8")
    nested = docs / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("Gamma text", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    documents = load_documents([docs])

    assert [document.doc_id for document in documents] == [
        "docs/a.md",
        "docs/b.txt",
        "docs/nested/c.md",
    ]
    assert documents[0].text.startswith("# Alpha")
    assert all(document.page is None for document in documents)


def test_skips_unsupported_and_empty_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.md").write_text("   \n", encoding="utf-8")
    (docs / "notes.rst").write_text("unsupported", encoding="utf-8")
    (docs / "real.txt").write_text("Real text", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    documents = load_documents([docs])

    assert len(documents) == 1
    assert documents[0].doc_id == "docs/real.txt"


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not found"):
        load_documents([tmp_path / "missing"])


def test_no_supported_files_raises(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.rst").write_text("unsupported", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match=r"No \.md, \.txt or \.pdf"):
        load_documents([docs])


def test_all_empty_documents_raise(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty.md").write_text("", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="none contained extractable text"):
        load_documents([docs])


def test_pdf_pages_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class FakeReader:
        def __init__(self, path: str) -> None:
            self.pages = [FakePage("Page one"), FakePage(""), FakePage("Page three")]

    monkeypatch.setattr(loaders, "PdfReader", FakeReader)
    monkeypatch.chdir(tmp_path)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    documents = load_documents([pdf_path])

    assert [document.page for document in documents] == [1, 3]
    assert [document.text for document in documents] == ["Page one", "Page three"]
    assert all(document.doc_id == "doc.pdf" for document in documents)


def test_pdf_reader_failure_is_clear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenReader:
        def __init__(self, path: str) -> None:
            raise RuntimeError("bad pdf")

    monkeypatch.setattr(loaders, "PdfReader", BrokenReader)
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(DocumentLoadError, match="Could not read PDF"):
        load_documents([pdf_path])
