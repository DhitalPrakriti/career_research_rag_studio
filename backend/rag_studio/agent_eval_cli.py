from __future__ import annotations

import argparse

from rag_studio.evaluation.agent_eval import run_agent_evaluation, summarize_agent_records
from rag_studio.evaluation.golden_set import save_jsonl


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run agentic RAG evaluation.")
    parser.add_argument(
        "--golden",
        default="evaluation/golden_set.jsonl",
        help="Path to the golden JSONL test set.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/runs/agent_latest.jsonl",
        help="Where to write agent evaluation records.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    records = run_agent_evaluation(
        golden_path=args.golden,
        top_k=args.top_k,
    )
    save_jsonl(records, args.output)
    summary = summarize_agent_records(records)

    answerable = int(summary["answerable_count"])
    negatives = int(summary["negative_control_count"])

    print(f"Examples: {len(records)}  ({answerable} answerable, {negatives} negative controls)")
    print(f"Output: {args.output}")
    print(f"Answerable term recall:   {summary['answerable_term_recall']:.3f}  (n={answerable})")
    print(f"Answerable doc title hit: {summary['answerable_doc_title_hit']:.3f}  (n={answerable})")
    print(f"Refusal accuracy:         {summary['refusal_accuracy']:.3f}  (n={negatives})")
    print(f"Rewrite rate:             {summary['rewrite_rate']:.3f}")
    print(f"Average retrieval grade:  {summary['average_retrieval_grade']:.3f}")


if __name__ == "__main__":
    main()
