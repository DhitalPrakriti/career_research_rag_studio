from __future__ import annotations

import argparse

from rag_studio.evaluation import run_evaluation, save_jsonl, summarize_records


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run RAG evaluation over a golden set.")
    parser.add_argument(
        "--golden",
        default="evaluation/golden_set.jsonl",
        help="Path to the golden JSONL test set.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/runs/latest.jsonl",
        help="Where to write RAGAS-compatible evaluation records.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--parent-context", action="store_true", default=True)
    parser.add_argument("--no-parent-context", action="store_false", dest="parent_context")
    parser.add_argument("--multi-query", action="store_true", default=True)
    parser.add_argument("--no-multi-query", action="store_false", dest="multi_query")
    parser.add_argument("--hyde", action="store_true")
    args = parser.parse_args()

    records = run_evaluation(
        golden_path=args.golden,
        retriever=args.retriever,
        top_k=args.top_k,
        rerank=args.rerank,
        candidate_k=args.candidate_k,
        parent_context=args.parent_context,
        multi_query=args.multi_query,
        hyde=args.hyde,
    )
    save_jsonl(records, args.output)
    summary = summarize_records(records)

    print(f"Examples: {len(records)}")
    print(f"Output: {args.output}")
    print(f"Average term recall: {summary['term_recall']:.3f}")
    print(f"Average doc title hit: {summary['doc_title_hit']:.3f}")


if __name__ == "__main__":
    main()
