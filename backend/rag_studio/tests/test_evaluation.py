from rag_studio.evaluation.golden_set import contains_expected_term, doc_title_hit, term_recall


def test_term_recall_scores_expected_terms_found_in_contexts() -> None:
    score = term_recall(
        ["This resume mentions PyTorch, RAG pipelines, and LangChain."],
        ["PyTorch", "RAG", "Docker"],
    )

    assert score == 2 / 3


def test_term_recall_matches_simple_singular_plural_variants() -> None:
    score = term_recall(
        ["Designed a 5-agent AI tutoring system."],
        ["agents"],
    )

    assert score == 1.0


def test_contains_expected_term_matches_multi_word_terms_by_tokens() -> None:
    assert contains_expected_term(
        "Achieved 94.28% Binary F1 using Late Fusion.",
        "Binary F1",
    )


def test_doc_title_hit_detects_expected_source_title() -> None:
    score = doc_title_hit(
        ["Prakriti_Dhital_Resume_AI_ML.pdf"],
        ["Prakriti_Dhital_Resume_AI_ML.pdf"],
    )

    assert score == 1.0
