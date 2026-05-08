from __future__ import annotations

import argparse
from pathlib import Path

from rag_studio.agent_graph import CareerResearchAgent


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Run the LangGraph career RAG agent.")
    parser.add_argument("question", help="Question to answer or route.")
    parser.add_argument("--docs", nargs="+", required=True, help="Documents to ingest.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--show-route",
        action="store_true",
        help="Print the router decision before the answer.",
    )
    args = parser.parse_args()

    agent = CareerResearchAgent()
    agent.ingest([Path(path) for path in args.docs])
    decision = agent.route(args.question)
    result = agent.answer(args.question, top_k=args.top_k)

    if args.show_route:
        print(f"Route: {decision.route}")
        print(f"Retriever: {decision.retriever}")
        print(f"Parent context: {decision.parent_context}")
        print(f"Multi-query: {decision.multi_query}")
        print(f"HyDE: {decision.hyde}")
        print(f"Reason: {decision.reason}")
        print()

    print(result.answer)
    if result.citations:
        print()
        print("Citations")
        for citation in result.citations:
            print(
                f"[{citation.source_id}] {citation.title} "
                f"({citation.location}, score={citation.score:.3f})"
            )


if __name__ == "__main__":
    main()
