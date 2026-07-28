from pathlib import Path

from rag_studio.ingestion.loader import load_document


def test_load_pdf_adds_expected_metadata() -> None:
    project_root = Path(__file__).resolve().parents[3]
    docs = load_document(project_root / "docs" / "Prakriti_Dhital_Resume_AI_ML.pdf")

    assert docs
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["doc_type"] == "resume"
    assert docs[0].metadata["title"] == "Prakriti_Dhital_Resume_AI_ML.pdf"
