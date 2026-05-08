from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


class SentenceTransformerLangchainEmbeddings:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS metrics with local Ollama.")
    parser.add_argument(
        "--input",
        default="evaluation/runs/all_resumes_baseline.jsonl",
        help="JSONL records produced by rag_studio.eval_cli.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/runs/ragas_ollama_scores.jsonl",
        help="Where to write row-level RAGAS scores.",
    )
    parser.add_argument("--judge-model", default="llama3", help="Ollama model for RAGAS judging.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local embedding model for embedding-based RAGAS metrics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples. Useful because local judging is slow.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Maximum retrieved context characters to send to the RAGAS judge per example.",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=4,
        help="CPU threads Ollama should use for judging.",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=0,
        help="GPU layers Ollama should use. Use 0 to force CPU-only judging.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=600,
        help="Seconds before a RAGAS judge call times out.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        help="RAGAS metrics to run. Local Ollama is slow, so running one metric at a time helps.",
    )
    args = parser.parse_args()

    records = _load_records(args.input, limit=args.limit)
    if not records:
        raise SystemExit(f"No records found in {args.input}. Run rag_studio.eval_cli first.")

    scores = run_ragas_with_ollama(
        records,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        max_context_chars=args.max_context_chars,
        num_thread=args.num_thread,
        num_gpu=args.num_gpu,
        judge_timeout=args.judge_timeout,
        metric_names=args.metrics,
    )
    _save_jsonl(scores, args.output)
    _print_summary(scores, args.output)


def run_ragas_with_ollama(
    records: list[dict[str, Any]],
    judge_model: str,
    embedding_model: str,
    max_context_chars: int,
    num_thread: int,
    num_gpu: int,
    judge_timeout: int,
    metric_names: list[str],
) -> list[dict[str, Any]]:
    from datasets import Dataset
    from langchain_ollama import ChatOllama
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    dataset = Dataset.from_list(
        [
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": _trim_context_texts(
                    record["contexts"],
                    max_context_chars,
                ),
                "reference": record["reference"],
            }
            for record in records
        ]
    )
    llm = ChatOllama(
        model=judge_model,
        temperature=0,
        num_thread=num_thread,
        num_gpu=num_gpu,
        num_predict=512,
    )
    embeddings = SentenceTransformerLangchainEmbeddings(embedding_model)
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    selected_metrics = _build_metrics(metric_names, ragas_llm, ragas_embeddings)

    result = evaluate(
        dataset,
        metrics=selected_metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=judge_timeout, max_workers=1, max_retries=1),
        raise_exceptions=False,
    )
    score_rows = result.to_pandas().to_dict(orient="records")
    return [
        {
            "id": source["id"],
            "question": source["question"],
            **_json_safe(score_row),
        }
        for source, score_row in zip(records, score_rows, strict=True)
    ]


def _load_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_summary(records: list[dict[str, Any]], output: str) -> None:
    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    print(f"Examples: {len(records)}")
    print(f"Output: {output}")
    for metric in metric_names:
        values = [
            float(record[metric])
            for record in records
            if metric in record and record[metric] is not None and not np.isnan(float(record[metric]))
        ]
        if values:
            print(f"Average {metric}: {sum(values) / len(values):.3f}")
        else:
            print(f"Average {metric}: n/a")


def _build_metrics(
    metric_names: list[str],
    ragas_llm: Any,
    ragas_embeddings: Any,
) -> list[Any]:
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    metrics: list[Any] = []
    for metric_name in metric_names:
        if metric_name == "faithfulness":
            metrics.append(Faithfulness(llm=ragas_llm))
        elif metric_name == "answer_relevancy":
            metrics.append(ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings))
        elif metric_name == "context_precision":
            metrics.append(LLMContextPrecisionWithReference(llm=ragas_llm))
        elif metric_name == "context_recall":
            metrics.append(LLMContextRecall(llm=ragas_llm))
        else:
            raise ValueError(f"Unknown RAGAS metric: {metric_name}")
    return metrics


def _trim_context_texts(contexts: list[str], max_context_chars: int) -> list[str]:
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    trimmed: list[str] = []
    used_chars = 0
    for context in contexts:
        separator_chars = 2 if trimmed else 0
        next_total = used_chars + separator_chars + len(context)
        if next_total <= max_context_chars:
            trimmed.append(context)
            used_chars = next_total
            continue

        remaining = max_context_chars - used_chars - separator_chars
        if remaining > 50:
            trimmed.append(context[: remaining - 3].rstrip() + "...")
        break
    return trimmed


def _json_safe(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float) and np.isnan(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


if __name__ == "__main__":
    main()
