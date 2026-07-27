from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_studio.pipeline import RagPipeline


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    docs: list[str]
    reference: str
    expected_terms: list[str]
    expected_doc_titles: list[str]


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
            }
        )
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"term_recall": 0.0, "doc_title_hit": 0.0}
    return {
        "term_recall": sum(float(record["term_recall"]) for record in records) / len(records),
        "doc_title_hit": sum(float(record["doc_title_hit"]) for record in records) / len(records),
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


def _normalized_tokens(text: str) -> list[str]:
    return [_singularize(token) for token in re.findall(r"[a-z0-9.#+]+", text.lower())]


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
