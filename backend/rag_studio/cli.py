from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rag_studio.pipeline import RagPipeline


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run the Phase 1 RAG baseline.")
    parser.add_argument("question", help="Question to answer from the ingested documents.")
    parser.add_argument("--docs", nargs="+", required=True, help="PDF, txt, or markdown files.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help=(
            "Retrieval strategy. dense uses embeddings + FAISS; bm25 uses keyword scoring; "
            "hybrid combines dense and BM25 with RRF."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank retrieved candidates with a cross-encoder before generation.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Number of candidates to retrieve before reranking. Defaults to 3 * top-k.",
    )
    parser.add_argument(
        "--parent-context",
        action="store_true",
        help="Retrieve small child chunks, then send their larger parent chunks to generation.",
    )
    parser.add_argument(
        "--multi-query",
        action="store_true",
        help="Run several query variants and merge their results with RRF.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="Maximum query variants to use with --multi-query.",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Use a hypothetical answer as the retrieval query before answering the original question.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Exact metadata filter. Can be repeated, for example: --filter type=pdf --filter page=1",
    )
    args = parser.parse_args()

    pipeline = RagPipeline()
    chunks = pipeline.ingest(
        [Path(path) for path in args.docs],
        build_dense_index=args.retriever in {"dense", "hybrid"},
        parent_child=args.parent_context,
    )
    metadata_filter = _parse_metadata_filters(args.filter)
    result = pipeline.answer(
        args.question,
        top_k=args.top_k,
        metadata_filter=metadata_filter,
        retriever=args.retriever,
        rerank=args.rerank,
        candidate_k=args.candidate_k,
        parent_context=args.parent_context,
        multi_query=args.multi_query,
        max_queries=args.max_queries,
        hyde=args.hyde,
    )

    print(f"Ingested chunks: {len(chunks)}")
    print(f"Retriever: {args.retriever}")
    print(f"Rerank: {args.rerank}")
    print(f"Parent context: {args.parent_context}")
    print(f"Multi-query: {args.multi_query}")
    print(f"HyDE: {args.hyde}")
    print()
    print(result.answer)
    print()
    print("Citations")
    for citation in result.citations:
        print(
            f"[{citation.source_id}] {citation.title} "
            f"({citation.location}, score={citation.score:.3f})"
        )


def _parse_metadata_filters(raw_filters: list[str]) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    for raw_filter in raw_filters:
        if "=" not in raw_filter:
            raise SystemExit(f"Invalid filter '{raw_filter}'. Expected KEY=VALUE.")
        key, value = raw_filter.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid filter '{raw_filter}'. Filter key cannot be empty.")
        filters[key] = _parse_filter_value(value.strip())
    return filters or None


def _parse_filter_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
