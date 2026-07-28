"""Negative controls need their own metrics.

A negative control asks something the documents deliberately do not answer, so the only
correct behaviour is refusal. Retrieval metrics do not apply — worse, term_recall returns
1.0 for an empty expected_terms list, so averaging negative controls into the headline
number is a free pass that hides real regressions.
"""

from rag_studio.evaluation.golden_set import (
    NOT_IN_DOCUMENTS,
    answer_refuses,
    is_negative_control,
    summarize_records,
)


def test_is_negative_control_matches_the_marker() -> None:
    assert is_negative_control(NOT_IN_DOCUMENTS) is True
    assert is_negative_control("not_in_documents") is True
    assert is_negative_control("  NOT_IN_DOCUMENTS  ") is True
    assert is_negative_control("Achieved 94.28% Binary F1.") is False


def test_answer_refuses_detects_a_refusal() -> None:
    assert answer_refuses("Prakriti's specific GPA is not mentioned [1].") is True
    assert answer_refuses("The sources do not specify a salary.") is True
    assert answer_refuses("No information about work experience is provided.") is True


def test_answer_refuses_accepts_a_grounded_no_for_absence_verification() -> None:
    """"Has Prakriti worked at Google?" is answerable from a complete history.

    A confident "there is no record" is the ideal answer, not a hedge, and must count as
    correct: the metric measures "did not fabricate", not "used hedging language".
    """
    assert (
        answer_refuses(
            "Based on the provided source, there is no record of Prakriti working at "
            "Google or Microsoft [1]."
        )
        is True
    )


def test_answer_refuses_detects_the_agents_own_low_confidence_message() -> None:
    """The graph emits this when retrieval grades irrelevant after the last retry."""
    assert (
        answer_refuses(
            "I found retrieved context, but it does not look relevant enough to answer "
            "confidently. Token overlap 0.12 using terms: gpa."
        )
        is True
    )


def test_answer_refuses_is_false_for_a_committed_answer() -> None:
    assert answer_refuses("The Transformer model achieved an 85.98% Macro F1 score [1].") is False


def _record(
    example_id: str,
    term_recall: float,
    negative: bool,
    refusal_correct: bool | None = None,
) -> dict[str, object]:
    return {
        "id": example_id,
        "term_recall": term_recall,
        "doc_title_hit": 1.0,
        "is_negative_control": negative,
        "refusal_correct": refusal_correct,
    }


def test_summary_separates_answerable_from_negative_controls() -> None:
    records = [
        _record("real_1", 1.0, negative=False),
        _record("real_2", 0.5, negative=False),
        # Negative controls score 1.0 on term_recall for free: empty expected_terms.
        _record("negative_gpa", 1.0, negative=True, refusal_correct=True),
        _record("negative_salary", 1.0, negative=True, refusal_correct=False),
    ]

    summary = summarize_records(records)

    # The blended figure is flattered by the two free passes...
    assert summary["term_recall"] == 0.875
    # ...while the answerable-only figure shows what retrieval actually did.
    assert summary["answerable_term_recall"] == 0.75
    assert summary["refusal_accuracy"] == 0.5
    assert summary["answerable_count"] == 2.0
    assert summary["negative_control_count"] == 2.0


def test_summary_handles_a_set_with_no_negative_controls() -> None:
    summary = summarize_records([_record("real_1", 1.0, negative=False)])

    assert summary["answerable_term_recall"] == 1.0
    assert summary["refusal_accuracy"] == 0.0
    assert summary["negative_control_count"] == 0.0


def test_summary_of_an_empty_run_is_all_zero() -> None:
    summary = summarize_records([])

    assert summary["answerable_term_recall"] == 0.0
    assert summary["refusal_accuracy"] == 0.0
    assert summary["answerable_count"] == 0.0
