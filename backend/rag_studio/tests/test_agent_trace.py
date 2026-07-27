from rag_studio.agent_trace import add_trace_event, format_trace


def test_add_trace_event_appends_without_mutating_original_trace() -> None:
    original = []

    updated = add_trace_event(original, "route_query", "Selected route.", route="retrieve")

    assert original == []
    assert len(updated) == 1
    assert updated[0].node == "route_query"
    assert updated[0].details["route"] == "retrieve"


def test_format_trace_prints_events_and_details() -> None:
    trace = add_trace_event([], "grade_retrieval", "Graded context.", score="0.500")

    report = format_trace(trace)

    assert "grade_retrieval" in report
    assert "score: 0.500" in report


def test_format_trace_handles_empty_trace() -> None:
    assert format_trace([]) == "Trace: no events recorded."
