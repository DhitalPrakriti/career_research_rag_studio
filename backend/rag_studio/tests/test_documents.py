"""Document management tests.

These endpoints write and delete files with no authentication in front of them, so the
filename handling is the security boundary. Traversal and type checks are tested directly
rather than through the API, because that is where the guarantee lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_studio.api.documents import (
    DocumentError,
    delete_document,
    list_documents,
    resolve_inside,
    safe_filename,
    save_document,
    writes_enabled,
)


class TestSafeFilename:
    @pytest.mark.parametrize(
        "name",
        [
            "resume.pdf",
            "Prakriti Dhital Resume.pdf",
            "resume_v2-final.PDF",
            "notes.md",
            "notes.txt",
        ],
    )
    def test_accepts_ordinary_names(self, name: str) -> None:
        assert safe_filename(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "../../.env",
            "../secrets.pdf",
            "..\\..\\windows\\system32\\evil.pdf",
            "/etc/passwd.pdf",
            "C:\\Windows\\evil.pdf",
        ],
    )
    def test_strips_or_rejects_directory_traversal(self, name: str) -> None:
        """Any surviving name must be a bare filename with no directory component."""
        try:
            cleaned = safe_filename(name)
        except DocumentError:
            return
        assert "/" not in cleaned and "\\" not in cleaned
        assert not cleaned.startswith("..")

    @pytest.mark.parametrize("name", ["script.exe", "archive.zip", "data.json", "noext"])
    def test_rejects_unsupported_types(self, name: str) -> None:
        with pytest.raises(DocumentError, match="not a supported file type|not an acceptable"):
            safe_filename(name)

    def test_rejects_an_empty_name(self) -> None:
        with pytest.raises(DocumentError):
            safe_filename("   ")

    def test_rejects_a_leading_dot(self) -> None:
        with pytest.raises(DocumentError):
            safe_filename(".env.pdf")


class TestResolveInside:
    def test_resolves_within_the_directory(self, tmp_path: Path) -> None:
        assert resolve_inside(tmp_path, "resume.pdf").parent == tmp_path.resolve()

    def test_a_traversal_attempt_cannot_escape(self, tmp_path: Path) -> None:
        """The guarantee is containment, not rejection.

        safe_filename reduces "../escaped.pdf" to "escaped.pdf", so the path is neutralised
        rather than refused. What matters is that the result never lands outside the
        directory; the explicit parent check in resolve_inside is belt-and-braces behind it.
        """
        nested = tmp_path / "docs"
        nested.mkdir()

        resolved = resolve_inside(nested, "../escaped.pdf")

        assert resolved.parent == nested.resolve()
        assert resolved.name == "escaped.pdf"


class TestSaveAndDelete:
    def test_saves_and_lists_a_document(self, tmp_path: Path) -> None:
        save_document(tmp_path, "resume.pdf", b"%PDF-1.4 content")

        listed = list_documents(tmp_path)

        assert [item.name for item in listed] == ["resume.pdf"]
        assert listed[0].size_bytes == len(b"%PDF-1.4 content")
        assert listed[0].modified

    def test_uploading_the_same_name_replaces_it(self, tmp_path: Path) -> None:
        """Replacing a resume with a newer version is the point; do not accumulate copies."""
        save_document(tmp_path, "resume.pdf", b"old version")
        save_document(tmp_path, "resume.pdf", b"new version which is longer")

        listed = list_documents(tmp_path)

        assert len(listed) == 1
        assert (tmp_path / "resume.pdf").read_bytes() == b"new version which is longer"

    def test_rejects_an_empty_upload(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentError, match="empty"):
            save_document(tmp_path, "resume.pdf", b"")

    def test_rejects_an_oversized_upload(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentError, match="larger than"):
            save_document(tmp_path, "resume.pdf", b"x" * (10 * 1024 * 1024 + 1))

    def test_listing_ignores_unrelated_files(self, tmp_path: Path) -> None:
        (tmp_path / "resume.pdf").write_bytes(b"pdf")
        (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        assert [item.name for item in list_documents(tmp_path)] == ["resume.pdf"]

    def test_listing_a_missing_directory_is_empty(self, tmp_path: Path) -> None:
        assert list_documents(tmp_path / "nope") == []

    def test_deletes_a_document(self, tmp_path: Path) -> None:
        save_document(tmp_path, "resume.pdf", b"content")

        delete_document(tmp_path, "resume.pdf")

        assert list_documents(tmp_path) == []

    def test_deleting_something_absent_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentError, match="not in the documents directory"):
            delete_document(tmp_path, "ghost.pdf")

    def test_delete_cannot_escape_the_directory(self, tmp_path: Path) -> None:
        outside = tmp_path / "keep.pdf"
        outside.write_bytes(b"important")
        nested = tmp_path / "docs"
        nested.mkdir()

        with pytest.raises(DocumentError):
            delete_document(nested, "../keep.pdf")

        assert outside.exists()


class TestWriteGate:
    def test_enabled_by_default(self) -> None:
        assert writes_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", " OFF "])
    def test_disabled_by_falsey_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("ALLOW_DOCUMENT_WRITES", value)

        assert writes_enabled() is False

    def test_enabled_by_anything_else(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOW_DOCUMENT_WRITES", "true")

        assert writes_enabled() is True
