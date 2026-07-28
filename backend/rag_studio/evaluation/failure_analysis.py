from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_studio.evaluation.golden_set import contains_expected_term


@dataclass(frozen=True)
class FailureExample:
    id: str
    question: str
    answer: str
    reference: str
    retrieved_titles: list[str]
    term_recall: float
    doc_title_hit: float
    missing_terms: list[str]
    # Defaults to 1.0 so a run written before the metric existed ranks neutrally rather
    # than sorting to the top as the worst example on missing data.
    doc_precision: float = 1.0

    @property
    def failure_score(self) -> float:
        """How badly the example failed to retrieve what it needed.

        Deliberately excludes `doc_precision`. Dragging in two irrelevant resumes is
        waste, not failure — the answer is still supported — so counting it here would
        report most of the set as failing. It ranks the report instead.
        """
        return (1.0 - self.term_recall) + (1.0 - self.doc_title_hit)


def load_failure_examples(path: str | Path) -> list[FailureExample]:
    examples: list[FailureExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(_record_to_failure_example(record))
    return examples


def worst_examples(examples: list[FailureExample], limit: int = 5) -> list[FailureExample]:
    """Rank by failure, then by imprecision.

    With term recall and doc-title hit both saturated at 1.000, every failure_score is 0
    and this degenerated into listing the set in file order — the report had nothing to
    say. Precision breaks that tie, so the examples that pulled in resumes they did not
    need surface first.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    return sorted(
        examples,
        key=lambda example: (
            example.failure_score,
            1.0 - example.doc_precision,
            1.0 - example.term_recall,
            1.0 - example.doc_title_hit,
        ),
        reverse=True,
    )[:limit]


def failed_examples(examples: list[FailureExample]) -> list[FailureExample]:
    return [example for example in examples if example.failure_score > 0]


def format_failure_report(examples: list[FailureExample]) -> str:
    if not examples:
        return "No evaluation records found."

    sections: list[str] = []
    for index, example in enumerate(examples, start=1):
        missing_terms = ", ".join(example.missing_terms) or "none"
        titles = ", ".join(example.retrieved_titles) or "none"
        sections.append(
            "\n".join(
                [
                    f"{index}. {example.id}",
                    f"Question: {example.question}",
                    f"Term recall: {example.term_recall:.3f}",
                    f"Doc title hit: {example.doc_title_hit:.3f}",
                    f"Doc precision: {example.doc_precision:.3f}",
                    f"Missing terms: {missing_terms}",
                    f"Retrieved titles: {titles}",
                    f"Reference: {_compact(example.reference)}",
                    f"Answer: {_compact(example.answer)}",
                ]
            )
        )
    return "\n\n".join(sections)


def _record_to_failure_example(record: dict[str, Any]) -> FailureExample:
    contexts = [str(context) for context in record.get("contexts", [])]
    expected_terms = [str(term) for term in record.get("expected_terms", [])]
    return FailureExample(
        id=str(record["id"]),
        question=str(record["question"]),
        answer=str(record["answer"]),
        reference=str(record["reference"]),
        retrieved_titles=[str(title) for title in record.get("retrieved_titles", [])],
        term_recall=float(record.get("term_recall", 0.0)),
        doc_title_hit=float(record.get("doc_title_hit", 0.0)),
        missing_terms=missing_terms(contexts, expected_terms),
        doc_precision=float(record.get("doc_precision", 1.0)),
    )


def missing_terms(contexts: list[str], expected_terms: list[str]) -> list[str]:
    combined_context = " ".join(contexts)
    return [
        term for term in expected_terms if not contains_expected_term(combined_context, term)
    ]


def _compact(text: str, max_chars: int = 240) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 3].rstrip() + "..."
