"""Golden-set evaluation, agent-level metrics, and failure inspection."""

from rag_studio.evaluation.agent_eval import (
    run_agent_evaluation,
    summarize_agent_records,
)
from rag_studio.evaluation.failure_analysis import (
    FailureExample,
    failed_examples,
    format_failure_report,
    load_failure_examples,
    missing_terms,
    worst_examples,
)
from rag_studio.evaluation.golden_set import (
    GoldenExample,
    contains_expected_term,
    doc_title_hit,
    load_golden_set,
    run_evaluation,
    save_jsonl,
    summarize_records,
    term_recall,
)

__all__ = [
    "FailureExample",
    "GoldenExample",
    "contains_expected_term",
    "doc_title_hit",
    "failed_examples",
    "format_failure_report",
    "load_failure_examples",
    "load_golden_set",
    "missing_terms",
    "run_agent_evaluation",
    "run_evaluation",
    "save_jsonl",
    "summarize_agent_records",
    "summarize_records",
    "term_recall",
    "worst_examples",
]
