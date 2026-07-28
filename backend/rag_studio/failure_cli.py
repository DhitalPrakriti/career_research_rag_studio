from __future__ import annotations

import argparse

from rag_studio.evaluation.failure_analysis import (
    failed_examples,
    format_failure_report,
    load_failure_examples,
    worst_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show weakest RAG evaluation examples.")
    parser.add_argument(
        "--input",
        default="evaluation/runs/all_resumes_baseline.jsonl",
        help="Evaluation JSONL produced by rag_studio.eval_cli.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        help="Number of weakest examples to print.",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Show only examples with term recall or document-title misses.",
    )
    args = parser.parse_args()

    examples = load_failure_examples(args.input)
    if args.only_failures:
        examples = failed_examples(examples)
    failures = worst_examples(examples, limit=args.worst)
    print(format_failure_report(failures))


if __name__ == "__main__":
    main()
