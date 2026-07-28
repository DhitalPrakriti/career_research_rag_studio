from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rag_studio.evaluation.golden_set import is_negative_control
from rag_studio.llm import (
    DEFAULT_GEMINI_MODEL,
    gemini_api_key,
    gemini_model_name,
    litellm_api_key,
    litellm_base_url,
    litellm_model_name,
)


class SentenceTransformerLangchainEmbeddings:
    """Minimal LangChain-style embeddings backed by sentence-transformers.

    `model` must stay a plain string. RAGAS reads it with
    getattr(embeddings, "model", None) for its usage telemetry and validates it as
    Optional[str], so exposing the SentenceTransformer object here raises a pydantic
    ValidationError that silently turns every embedding-based metric (that is,
    answer_relevancy) into a null score. The encoder lives on `_encoder` instead.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = model_name
        self._encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32").tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run RAGAS metrics with a Gemini or Ollama judge.")
    parser.add_argument(
        "--input",
        default="evaluation/runs/all_resumes_baseline.jsonl",
        help="JSONL records produced by rag_studio.eval_cli.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/runs/ragas_scores.jsonl",
        help="Where to write row-level RAGAS scores.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["gemini", "litellm", "ollama"],
        default="gemini",
        help=(
            "Which judge to use. gemini and litellm are API calls; ollama needs a local "
            "server. Use litellm to keep the proxy's budget caps and caching."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge model. Defaults to GEMINI_MODEL (or "
            f"{DEFAULT_GEMINI_MODEL}) for Gemini and llama3 for Ollama. Consider "
            "gemini-3.5-flash-lite to keep judging cost down."
        ),
    )
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
        help="CPU threads Ollama should use for judging. Ignored for Gemini.",
    )
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=0,
        help="GPU layers Ollama should use. Use 0 to force CPU-only. Ignored for Gemini.",
    )
    parser.add_argument(
        "--answer-relevancy-strictness",
        type=int,
        default=1,
        help=(
            "How many questions answer_relevancy reverse-generates per example. Must be "
            "1 for Gemini flash models, which reject multiple candidates per call."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Concurrent judge calls. Defaults to 4 for Gemini and 1 for Ollama, since "
            "local judging is the bottleneck but API judging is not."
        ),
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

    if args.judge_model:
        judge_model = args.judge_model
    elif args.judge_provider == "gemini":
        judge_model = gemini_model_name()
    elif args.judge_provider == "litellm":
        judge_model = litellm_model_name()
    else:
        judge_model = "llama3"

    max_workers = args.max_workers
    if max_workers is None:
        max_workers = 1 if args.judge_provider == "ollama" else 4

    print(f"Judge: {args.judge_provider} / {judge_model} (max_workers={max_workers})")

    scores = run_ragas(
        records,
        judge_model=judge_model,
        embedding_model=args.embedding_model,
        max_context_chars=args.max_context_chars,
        num_thread=args.num_thread,
        num_gpu=args.num_gpu,
        judge_timeout=args.judge_timeout,
        metric_names=args.metrics,
        judge_provider=args.judge_provider,
        max_workers=max_workers,
        answer_relevancy_strictness=args.answer_relevancy_strictness,
    )
    _save_jsonl(scores, args.output)
    _print_summary(scores, args.output)


def _build_judge_llm(
    provider: str,
    judge_model: str,
    num_thread: int,
    num_gpu: int,
):
    """Construct the RAGAS judge model for the chosen provider."""
    if provider == "gemini":
        api_key = gemini_api_key()
        if not api_key:
            raise SystemExit(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) before "
                "judging with --judge-provider gemini."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise SystemExit(
                "Install the Gemini judge dependency: pip install langchain-google-genai"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=judge_model,
            temperature=0,
            google_api_key=api_key,
        )

    if provider == "litellm":
        api_key = litellm_api_key()
        if not api_key:
            raise SystemExit(
                "No LiteLLM key found. Set LITELLM_API_KEY (or LITELLM_MASTER_KEY) before "
                "judging with --judge-provider litellm."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise SystemExit(
                "Install the OpenAI-compatible judge dependency: pip install langchain-openai"
            ) from exc
        return ChatOpenAI(
            model=judge_model,
            temperature=0,
            api_key=api_key,
            base_url=litellm_base_url(),
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=judge_model,
        temperature=0,
        num_thread=num_thread,
        num_gpu=num_gpu,
        num_predict=512,
    )


def run_ragas(
    records: list[dict[str, Any]],
    judge_model: str,
    embedding_model: str,
    max_context_chars: int,
    num_thread: int,
    num_gpu: int,
    judge_timeout: int,
    metric_names: list[str],
    judge_provider: str = "gemini",
    max_workers: int = 1,
    answer_relevancy_strictness: int = 1,
) -> list[dict[str, Any]]:
    from datasets import Dataset
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
    llm = _build_judge_llm(judge_provider, judge_model, num_thread, num_gpu)
    embeddings = SentenceTransformerLangchainEmbeddings(embedding_model)
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    selected_metrics = _build_metrics(
        metric_names,
        ragas_llm,
        ragas_embeddings,
        answer_relevancy_strictness=answer_relevancy_strictness,
    )

    result = evaluate(
        dataset,
        metrics=selected_metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=judge_timeout, max_workers=max_workers, max_retries=1),
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

    def values_for(rows: list[dict[str, Any]], metric: str) -> list[float]:
        return [
            float(row[metric])
            for row in rows
            if metric in row and row[metric] is not None and not np.isnan(float(row[metric]))
        ]

    answerable = [r for r in records if not is_negative_control(str(r.get("reference", "")))]
    negatives = [r for r in records if is_negative_control(str(r.get("reference", "")))]

    print(f"Examples: {len(records)}")
    print(f"Output: {output}")
    for metric in metric_names:
        overall = values_for(records, metric)
        if not overall:
            continue
        line = f"Average {metric}: {sum(overall) / len(overall):.3f}"
        # answer_relevancy floors any "not in the sources" answer to 0 as noncommittal,
        # so on negative controls it penalises exactly the behaviour we want. Report the
        # answerable-only figure beside it rather than letting the blended number stand.
        if metric == "answer_relevancy" and negatives:
            subset = values_for(answerable, metric)
            if subset:
                print(f"{line}   [answerable only: {sum(subset) / len(subset):.3f}]")
                continue
        print(line)

    if negatives:
        print(
            f"\nNote: {len(negatives)} negative controls are included above. RAGAS scores a "
            "correct refusal as 0.0 answer_relevancy, so use the answerable-only figure to "
            "judge answer quality and refusal_accuracy from the eval CLIs to judge refusals."
        )


def _build_metrics(
    metric_names: list[str],
    ragas_llm: Any,
    ragas_embeddings: Any,
    answer_relevancy_strictness: int = 1,
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
            # strictness is how many questions RAGAS reverse-generates from the
            # answer, requested as n candidates in a single call. Gemini flash models
            # reject n > 1 with "Multiple candidates is not enabled for this model",
            # so the default here is 1. Raise it for judges that allow candidates:
            # a higher value averages over more generated questions and is less noisy.
            metrics.append(
                ResponseRelevancy(
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                    strictness=answer_relevancy_strictness,
                )
            )
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
