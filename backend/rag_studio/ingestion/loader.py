from __future__ import annotations

from pathlib import Path

from rag_studio.schema import Document


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_document(path: str | Path) -> list[Document]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Document does not exist: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(
            f"Unsupported document type '{suffix}'. Supported: {supported}"
        )

    if suffix == ".pdf":
        return _load_pdf(source_path)

    return [
        Document(
            text=source_path.read_text(encoding="utf-8"),
            metadata={
                "source_path": str(source_path),
                "title": source_path.name,
                "type": suffix.lstrip("."),
                "doc_type": _infer_doc_type(source_path),  # ← added
            },
        )
    ]


def load_documents(paths: list[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for path in paths:
        documents.extend(load_document(path))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Install pypdf to ingest PDF files: pip install pypdf"
        ) from exc

    reader = PdfReader(str(path))
    doc_type = _infer_doc_type(path)  # ← infer once per file
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        documents.append(
            Document(
                text=text,
                metadata={
                    "source_path": str(path),
                    "title": path.name,
                    "type": "pdf",
                    "page": page_index,
                    "doc_type": doc_type,  # ← added
                },
            )
        )
    return documents


def _infer_doc_type(path: Path) -> str:
    """
    Infer document category from filename.
    This metadata flows through the entire pipeline
    and enables metadata filtering in Phase 2 retrieval.
    """
    name = path.stem.lower()

    if any(k in name for k in ["resume", "cv"]):
        return "resume"
    if any(k in name for k in ["job", "jd", "description", "posting"]):
        return "job_description"
    if any(k in name for k in ["cover", "letter"]):
        return "cover_letter"
    if any(k in name for k in ["note", "class", "lecture", "week"]):
        return "class_notes"
    if any(k in name for k in ["project", "proposal", "spec"]):
        return "project_doc"
    if any(k in name for k in ["paper", "arxiv", "research"]):
        return "research_paper"
    return "general"