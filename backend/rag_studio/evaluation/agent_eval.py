from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_studio.agents.graph import CareerResearchAgent
from rag_studio.agents.trace import AgentTraceEvent
from rag_studio.evaluation.golden_set import (
    doc_title_hit,
    load_golden_set,
    summarize_records,
    term_recall,
)


def run_agent_evaluation(
    golden_path: str | Path,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in load_golden_set(golden_path):
        agent = CareerResearchAgent()
        agent.ingest(example.docs)
        state = agent.run(example.question, top_k=top_k)
        result = state["answer"]
        contexts = [context.chunk.text for context in result.contexts]
        retrieved_titles = [
            str(context.chunk.metadata.get("title", "")) for context in result.contexts
        ]
        trace = state.get("trace", [])
        records.append(
            {
                "id": example.id,
                "question": example.question,
                "answer": result.answer,
                "contexts": contexts,
                "reference": example.reference,
                "ground_truth": example.reference,
                "expected_terms": example.expected_terms,
                "expected_doc_titles": example.expected_doc_titles,
                "retrieved_titles": retrieved_titles,
                "term_recall": term_recall(contexts, example.expected_terms),
                "doc_title_hit": doc_title_hit(retrieved_titles, example.expected_doc_titles),
                "route": _trace_detail(trace, "route_query", "route"),
                "retriever": _trace_detail(trace, "route_query", "retriever"),
                "parent_context": _trace_detail(trace, "route_query", "parent_context"),
                "multi_query": _trace_detail(trace, "route_query", "multi_query"),
                "hyde": _trace_detail(trace, "route_query", "hyde"),
                "rewrite_count": _count_trace_events(trace, "rewrite_query"),
                "retrieval_grade_score": _last_trace_detail(
                    trace,
                    "grade_retrieval",
                    "score",
                ),
                "retrieval_grade_relevant": _last_trace_detail(
                    trace,
                    "grade_retrieval",
                    "is_relevant",
                ),
            }
        )
    return records


def summarize_agent_records(records: list[dict[str, Any]]) -> dict[str, float]:
    summary = summarize_records(records)
    if not records:
        return {
            **summary,
            "rewrite_rate": 0.0,
            "average_retrieval_grade": 0.0,
        }

    grade_values = [
        float(record["retrieval_grade_score"])
        for record in records
        if record.get("retrieval_grade_score") is not None
    ]
    return {
        **summary,
        "rewrite_rate": sum(int(record["rewrite_count"] > 0) for record in records)
        / len(records),
        "average_retrieval_grade": (
            sum(grade_values) / len(grade_values) if grade_values else 0.0
        ),
    }


def _count_trace_events(trace: list[AgentTraceEvent], node: str) -> int:
    return sum(1 for event in trace if event.node == node)


def _trace_detail(trace: list[AgentTraceEvent], node: str, key: str) -> Any:
    for event in trace:
        if event.node == node:
            return event.details.get(key)
    return None


def _last_trace_detail(trace: list[AgentTraceEvent], node: str, key: str) -> Any:
    for event in reversed(trace):
        if event.node == node:
            return event.details.get(key)
    return None
