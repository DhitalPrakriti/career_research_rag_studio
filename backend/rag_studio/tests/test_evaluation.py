from rag_studio.evaluation.golden_set import (
    contains_expected_term,
    doc_precision,
    doc_title_hit,
    summarize_records,
    term_recall,
)


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


def test_doc_title_hit_saturates_once_everything_is_retrieved() -> None:
    """Returning all three resumes guarantees a hit, which is why it stopped
    discriminating between configurations on a corpus this small."""
    all_three = [
        "Prakriti_Dhital_Resume_AI_ML.pdf",
        "Prakriti-Dhital-Resume-Software-Developer.pdf",
        "Prakriti Dhital IT support Resume.pdf",
    ]

    assert doc_title_hit(all_three, ["Prakriti_Dhital_Resume_AI_ML.pdf"]) == 1.0
    assert doc_precision(all_three, ["Prakriti_Dhital_Resume_AI_ML.pdf"]) == 1 / 3


def test_doc_precision_is_one_when_every_chunk_is_from_an_expected_doc() -> None:
    score = doc_precision(
        ["Prakriti_Dhital_Resume_AI_ML.pdf", "Prakriti_Dhital_Resume_AI_ML.pdf"],
        ["Prakriti_Dhital_Resume_AI_ML.pdf"],
    )

    assert score == 1.0


def test_doc_precision_counts_every_expected_title() -> None:
    """Some questions are answered by two resumes, so both count as precise."""
    score = doc_precision(
        ["Prakriti_Dhital_Resume_AI_ML.pdf", "Prakriti-Dhital-Resume-Software-Developer.pdf"],
        ["Prakriti_Dhital_Resume_AI_ML.pdf", "Prakriti-Dhital-Resume-Software-Developer.pdf"],
    )

    assert score == 1.0


def test_doc_precision_of_an_empty_retrieval_is_zero() -> None:
    assert doc_precision([], ["Prakriti_Dhital_Resume_AI_ML.pdf"]) == 0.0


def test_summary_skips_a_metric_older_records_do_not_carry() -> None:
    """Averaging a missing doc_precision as 0.0 would report a collapse that never
    happened, so records that predate the metric are skipped rather than counted."""
    records = [
        {
            "term_recall": 1.0,
            "doc_title_hit": 1.0,
            "doc_precision": 0.5,
            "is_negative_control": False,
        },
        {"term_recall": 1.0, "doc_title_hit": 1.0, "is_negative_control": False},
    ]

    summary = summarize_records(records)

    assert summary["answerable_doc_precision"] == 0.5
    assert summary["answerable_term_recall"] == 1.0
