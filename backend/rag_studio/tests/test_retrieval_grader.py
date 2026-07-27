import pytest

from rag_studio.retrieval_grader import RetrievalGrader
from rag_studio.schema import Chunk, RetrievedChunk


def test_retrieval_grader_marks_overlapping_context_as_relevant() -> None:
    grade = RetrievalGrader().grade(
        "What database was used for memory?",
        [_result("Firestore was used for multi-turn conversational memory.")],
    )

    assert grade.is_relevant is True
    assert grade.score > 0


def test_retrieval_grader_marks_empty_contexts_as_not_relevant() -> None:
    grade = RetrievalGrader().grade("What database was used for memory?", [])

    assert grade.is_relevant is False
    assert grade.score == 0.0


def test_retrieval_grader_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="min_overlap"):
        RetrievalGrader(min_overlap=1.5)


def test_retrieval_grader_ranks_most_relevant_context_first() -> None:
    contexts = [
        _result("Vector databases include FAISS."),
        _result("Firestore was used for multi-turn conversation memory."),
    ]

    ranked = RetrievalGrader().rank(
        "database memory Firestore conversation",
        contexts,
    )

    assert "Firestore" in ranked[0].chunk.text


def _result(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id="chunk-1", text=text, metadata={}),
        score=1.0,
    )
