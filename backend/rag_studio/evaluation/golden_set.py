from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_studio.pipeline import RagPipeline


# Marker used in the golden set for questions whose answer is deliberately absent
# from the documents. These are negative controls: the only correct behaviour is to
# refuse, so retrieval and relevancy metrics do not apply to them.
NOT_IN_DOCUMENTS = "NOT_IN_DOCUMENTS"

# Heuristic, and deliberately so: refusal detection stays deterministic and testable
# rather than depending on another LLM call to score the thing under test.
#
# Negative controls come in two shapes and both count as correct here, because what is
# actually being measured is "did not fabricate":
#   1. Unanswerable value questions ("What is the GPA?") where the documents are silent,
#      so the right answer is "not mentioned".
#   2. Absence-verification questions ("Has Prakriti worked at Google?") where the
#      documents list the full history, so a confident "there is no record of that" is the
#      ideal answer rather than a hedge.
# Markers for both appear below.
_REFUSAL_MARKERS = (
    "not mentioned",
    "not specified",
    "not stated",
    "not provided",
    "not included",
    "not listed",
    "not available",
    "not found",
    "not in the source",
    "not present",
    "does not mention",
    "does not specify",
    "does not state",
    "does not include",
    "does not contain",
    "do not mention",
    "do not specify",
    "do not state",
    "do not include",
    "do not contain",
    "don't mention",
    "doesn't mention",
    "isn't mentioned",
    "aren't mentioned",
    "no information",
    "no mention",
    # Absence-verification phrasing: a grounded "no" rather than a hedge.
    "no record",
    "no evidence",
    "no indication",
    "does not appear",
    "do not appear",
    "not appear to",
    "cannot determine",
    "cannot find",
    "could not find",
    "unable to determine",
    "is missing",
    "insufficient",
    # The agent's own low-confidence message, emitted when retrieval grades irrelevant
    # after the last retry. It is a refusal and must count as one.
    "not look relevant enough",
    "not relevant enough",
)


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    docs: list[str]
    reference: str
    expected_terms: list[str]
    expected_doc_titles: list[str]

    @property
    def is_negative_control(self) -> bool:
        return is_negative_control(self.reference)


def load_golden_set(path: str | Path) -> list[GoldenExample]:
    examples: list[GoldenExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                GoldenExample(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    docs=[str(doc) for doc in raw["docs"]],
                    reference=str(raw["reference"]),
                    expected_terms=[str(term) for term in raw.get("expected_terms", [])],
                    expected_doc_titles=[
                        str(title) for title in raw.get("expected_doc_titles", [])
                    ],
                )
            )
    return examples


def run_evaluation(
    golden_path: str | Path,
    retriever: str = "hybrid",
    top_k: int = 3,
    rerank: bool = False,
    candidate_k: int | None = None,
    parent_context: bool = True,
    multi_query: bool = True,
    hyde: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in load_golden_set(golden_path):
        pipeline = RagPipeline()
        pipeline.ingest(
            example.docs,
            build_dense_index=retriever in {"dense", "hybrid"},
            parent_child=parent_context,
        )
        result = pipeline.answer(
            example.question,
            top_k=top_k,
            retriever=retriever,
            rerank=rerank,
            candidate_k=candidate_k,
            parent_context=parent_context,
            multi_query=multi_query,
            hyde=hyde,
        )
        contexts = [context.chunk.text for context in result.contexts]
        retrieved_titles = [
            str(context.chunk.metadata.get("title", "")) for context in result.contexts
        ]
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
                "doc_precision": doc_precision(retrieved_titles, example.expected_doc_titles),
                # Diagnostic, not a score: makes it visible when the retrieved context has
                # grown large enough that recall metrics cannot fail.
                "context_chars": sum(len(context) for context in contexts),
                "is_negative_control": example.is_negative_control,
                "refusal_correct": (
                    answer_refuses(result.answer) if example.is_negative_control else None
                ),
            }
        )
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_negative_control(reference: str) -> bool:
    return reference.strip().upper() == NOT_IN_DOCUMENTS


def answer_refuses(answer: str) -> bool:
    """Whether an answer admits the information is not in the sources."""
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    """Summarise a run, keeping answerable questions and negative controls apart.

    Averaging the two classes together is misleading in both directions. Retrieval
    metrics are meaningless for a negative control — there is nothing to retrieve, and
    term_recall returns 1.0 for an empty expected_terms list, so every negative control
    is a free pass that inflates the headline number. Conversely the only thing worth
    measuring on a negative control is whether the system refused instead of
    hallucinating, which is what refusal_accuracy reports.
    """
    if not records:
        return {
            "term_recall": 0.0,
            "doc_title_hit": 0.0,
            "answerable_term_recall": 0.0,
            "answerable_doc_title_hit": 0.0,
            "answerable_doc_precision": 0.0,
            "mean_context_chars": 0.0,
            "refusal_accuracy": 0.0,
            "answerable_count": 0.0,
            "negative_control_count": 0.0,
        }

    answerable = [record for record in records if not record.get("is_negative_control")]
    negatives = [record for record in records if record.get("is_negative_control")]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        """Average over the rows that carry the metric.

        Records written by an older run predate the newer metrics. Averaging a missing
        value as 0.0 would report a precision collapse that never happened, so absent
        keys are skipped and a metric no row carries reads 0.0, the same as no data.
        """
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            return 0.0
        return sum(values) / len(values)

    return {
        # Kept for continuity with earlier runs, but read the answerable_* figures.
        "term_recall": mean(records, "term_recall"),
        "doc_title_hit": mean(records, "doc_title_hit"),
        "answerable_term_recall": mean(answerable, "term_recall"),
        "answerable_doc_title_hit": mean(answerable, "doc_title_hit"),
        # The figure with headroom once the two above saturate.
        "answerable_doc_precision": mean(answerable, "doc_precision"),
        "mean_context_chars": mean(records, "context_chars"),
        "refusal_accuracy": (
            sum(1 for row in negatives if row.get("refusal_correct")) / len(negatives)
            if negatives
            else 0.0
        ),
        "answerable_count": float(len(answerable)),
        "negative_control_count": float(len(negatives)),
    }


def term_recall(contexts: list[str], expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    combined_context = " ".join(contexts)
    hits = sum(1 for term in expected_terms if contains_expected_term(combined_context, term))
    return hits / len(expected_terms)


def contains_expected_term(text: str, expected_term: str) -> bool:
    if expected_term.lower() in text.lower():
        return True

    text_tokens = set(_normalized_tokens(text))
    term_tokens = _normalized_tokens(expected_term)
    return bool(term_tokens) and all(token in text_tokens for token in term_tokens)


def doc_title_hit(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    if not expected_titles:
        return 1.0
    retrieved = set(retrieved_titles)
    return 1.0 if any(title in retrieved for title in expected_titles) else 0.0


def doc_precision(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    """Share of retrieved chunks that came from a document expected to hold the answer.

    `doc_title_hit` asks whether the right document appeared at all, which saturates the
    moment retrieval returns enough of the corpus to be sure of including it. On a corpus
    of three one-page resumes, page-level parent context at top_k=3 returns up to 84% of
    every character available, so hitting the right document is close to unavoidable and
    the metric stops telling two configurations apart.

    Precision is what still has headroom: returning all three resumes for a question only
    one of them answers scores 0.333 and pushes the work of ignoring two irrelevant
    resumes onto the generator.
    """
    if not expected_titles:
        return 1.0
    if not retrieved_titles:
        return 0.0
    expected = set(expected_titles)
    return sum(1 for title in retrieved_titles if title in expected) / len(retrieved_titles)


def _normalized_tokens(text: str) -> list[str]:
    return [_singularize(token) for token in re.findall(r"[a-z0-9.#+]+", text.lower())]


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
