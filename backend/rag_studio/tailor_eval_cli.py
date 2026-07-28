from __future__ import annotations

import argparse

from rag_studio.evaluation.golden_set import save_jsonl
from rag_studio.evaluation.tailoring_eval import (
    load_tailoring_set,
    run_tailoring_evaluation,
    summarize_tailoring,
)
from rag_studio.tailoring import ResumeTailor


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate resume tailoring against labelled job descriptions."
    )
    parser.add_argument(
        "--golden",
        default="evaluation/tailoring_set.jsonl",
        help="Labelled tailoring set.",
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=None,
        help="Resume files to tailor from. Defaults to everything in docs/.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/runs/tailoring_latest.jsonl",
        help="Where to write per-example records.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only the first N postings.")
    args = parser.parse_args()

    examples = load_tailoring_set(args.golden)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit(f"No examples found in {args.golden}.")

    from pathlib import Path

    docs = args.docs
    if docs is None:
        docs = [
            str(path)
            for path in sorted(Path("docs").glob("*"))
            if path.suffix.lower() in {".pdf", ".txt", ".md"}
        ]
    if not docs:
        raise SystemExit("No resume documents found. Pass --docs or populate docs/.")

    tailor = ResumeTailor()
    tailor.pipeline.ingest(docs, build_dense_index=True, parent_child=True)

    records = run_tailoring_evaluation(tailor, examples)
    save_jsonl(records, args.output)
    summary = summarize_tailoring(records)

    print(f"Postings: {int(summary['examples'])}")
    print(f"Output: {args.output}")
    print()
    print("Fabrication (the guarantee that matters)")
    print(f"  bullets generated       : {int(summary['bullets'])}")
    print(f"  fabrication rate        : {summary['fabrication_rate']:.3f}  (claim in no evidence)")
    print(f"  misattribution rate     : {summary['misattribution_rate']:.3f}  (grounded, wrong requirement)")
    print(f"  bullets fully grounded  : {int(summary['clean_bullets'])}")
    print()
    print(f"Classification ({int(summary['labels'])} labels)")
    print(f"  overall accuracy        : {summary['classification_accuracy']:.3f}")
    print(f"  present recalled        : {summary['present_accuracy']:.3f}")
    print(f"  absent correctly gapped : {summary['absent_accuracy']:.3f}")
    print(f"  extraction recall       : {summary['extraction_recall']:.3f}")

    misses = [
        (record["id"], row)
        for record in records
        for row in record["labels"]
        if not row["correct"]
    ]
    if misses:
        print()
        print("Disagreements with the labels:")
        for example_id, row in misses:
            actual = ",".join(row["actual"]) if row["actual"] else "not extracted"
            print(f"  [{example_id}] {row['keyword']}: expected {row['expected']}, got {actual}")

    fabricated = [
        (record["id"], row) for record in records for row in record["bullets"] if row["fabricated"]
    ]
    if fabricated:
        print()
        print("Invented claims (absent from ALL retrieved evidence):")
        for example_id, row in fabricated:
            print(f"  [{example_id}] {', '.join(row['invented_claims'])}")
            print(f"      {row['text'][:110]}")

    misattributed = [
        (record["id"], row)
        for record in records
        for row in record["bullets"]
        if row["misattributed"]
    ]
    if misattributed:
        print()
        print("Misattributed claims (real, but from another requirement's evidence):")
        for example_id, row in misattributed:
            print(f"  [{example_id}] {', '.join(row['unsupported_claims'])}")


if __name__ == "__main__":
    main()
