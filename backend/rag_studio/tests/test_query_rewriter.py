from rag_studio.query_rewriter import QueryRewriter


def test_query_rewriter_expands_ai_questions() -> None:
    rewritten = QueryRewriter().rewrite("What AI skills are shown?")

    assert "machine learning" in rewritten
    assert "multi-agent" in rewritten


def test_query_rewriter_expands_memory_questions() -> None:
    rewritten = QueryRewriter().rewrite("What database was used for memory?")

    assert "Firestore" in rewritten
    assert "multi-turn memory" in rewritten


def test_query_rewriter_adds_default_career_terms() -> None:
    rewritten = QueryRewriter().rewrite("Tell me about this person")

    assert "resume" in rewritten
    assert "experience" in rewritten
