import json

import pytest

from rag_studio.evaluation.failure_analysis import (
    FailureExample,
    failed_examples,
    format_failure_report,
    load_failure_examples,
    missing_terms,
    worst_examples,
)


def test_missing_terms_returns_terms_not_found_in_contexts() -> None:
    missing = missing_terms(
        ["The resume mentions PyTorch and Docker."],
        ["PyTorch", "LangChain", "Docker"],
    )

    assert missing == ["LangChain"]


def test_missing_terms_uses_normalized_term_matching() -> None:
    missing = missing_terms(
        ["Designed a 5-agent AI tutoring system."],
        ["agents"],
    )

    assert missing == []


def test_worst_examples_sorts_lowest_scores_first() -> None:
    examples = [
        _example("good", term_recall=1.0, doc_title_hit=1.0),
        _example("weak", term_recall=0.5, doc_title_hit=1.0),
        _example("weakest", term_recall=0.5, doc_title_hit=0.0),
    ]

    assert [example.id for example in worst_examples(examples, limit=2)] == [
        "weakest",
        "weak",
    ]


def test_load_failure_examples_reads_jsonl(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(_record("example")) + "\n", encoding="utf-8")

    examples = load_failure_examples(path)

    assert len(examples) == 1
    assert examples[0].id == "example"


def test_format_failure_report_includes_debug_fields() -> None:
    example = _example("example", term_recall=0.0)

    report = format_failure_report([example])

    assert "Question:" in report
    assert "Missing terms:" in report
    assert "Retrieved titles:" in report


def test_worst_examples_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        worst_examples([], limit=0)


def test_failed_examples_keeps_only_non_perfect_examples() -> None:
    examples = [
        _example("good", term_recall=1.0, doc_title_hit=1.0),
        _example("weak", term_recall=0.5, doc_title_hit=1.0),
    ]

    assert [example.id for example in failed_examples(examples)] == ["weak"]


def _record(
    record_id: str,
    term_recall: float = 1.0,
    doc_title_hit: float = 1.0,
) -> dict[str, object]:
    return {
        "id": record_id,
        "question": "What skill is shown?",
        "answer": "The answer mentions PyTorch.",
        "reference": "The reference mentions PyTorch and LangChain.",
        "contexts": ["The context mentions PyTorch."],
        "expected_terms": ["PyTorch", "LangChain"],
        "retrieved_titles": ["resume.pdf"],
        "term_recall": term_recall,
        "doc_title_hit": doc_title_hit,
    }


def _example(
    example_id: str,
    term_recall: float = 1.0,
    doc_title_hit: float = 1.0,
) -> FailureExample:
    return FailureExample(
        id=example_id,
        question="What skill is shown?",
        answer="The answer mentions PyTorch.",
        reference="The reference mentions PyTorch and LangChain.",
        retrieved_titles=["resume.pdf"],
        term_recall=term_recall,
        doc_title_hit=doc_title_hit,
        missing_terms=["LangChain"],
    )
