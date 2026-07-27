import json

from rag_studio.agent_evaluation import (
    _count_trace_events,
    _last_trace_detail,
    _trace_detail,
    summarize_agent_records,
)
from rag_studio.agent_trace import add_trace_event


def test_trace_helpers_extract_agent_metadata() -> None:
    trace = add_trace_event([], "route_query", "Selected route.", route="retrieve")
    trace = add_trace_event(trace, "rewrite_query", "Rewrote query.")
    trace = add_trace_event(trace, "grade_retrieval", "Graded.", score="0.500")
    trace = add_trace_event(trace, "grade_retrieval", "Graded.", score="0.750")

    assert _trace_detail(trace, "route_query", "route") == "retrieve"
    assert _count_trace_events(trace, "rewrite_query") == 1
    assert _last_trace_detail(trace, "grade_retrieval", "score") == "0.750"


def test_summarize_agent_records_includes_rewrite_and_grade_metrics() -> None:
    records = [
        _record(term_recall=1.0, doc_title_hit=1.0, rewrite_count=1, grade="0.5"),
        _record(term_recall=0.5, doc_title_hit=1.0, rewrite_count=0, grade="1.0"),
    ]

    summary = summarize_agent_records(records)

    assert summary["term_recall"] == 0.75
    assert summary["doc_title_hit"] == 1.0
    assert summary["rewrite_rate"] == 0.5
    assert summary["average_retrieval_grade"] == 0.75


def test_agent_record_shape_matches_jsonl_output() -> None:
    record = _record(term_recall=1.0, doc_title_hit=1.0, rewrite_count=0, grade="1.0")

    encoded = json.dumps(record)
    decoded = json.loads(encoded)

    assert decoded["route"] == "retrieve"
    assert decoded["retrieval_grade_score"] == "1.0"


def _record(
    term_recall: float,
    doc_title_hit: float,
    rewrite_count: int,
    grade: str,
) -> dict[str, object]:
    return {
        "id": "example",
        "question": "Question?",
        "answer": "Answer.",
        "contexts": ["Context"],
        "reference": "Reference.",
        "ground_truth": "Reference.",
        "expected_terms": ["Context"],
        "expected_doc_titles": ["resume.pdf"],
        "retrieved_titles": ["resume.pdf"],
        "term_recall": term_recall,
        "doc_title_hit": doc_title_hit,
        "route": "retrieve",
        "retriever": "hybrid",
        "parent_context": True,
        "multi_query": True,
        "hyde": False,
        "rewrite_count": rewrite_count,
        "retrieval_grade_score": grade,
        "retrieval_grade_relevant": True,
    }
