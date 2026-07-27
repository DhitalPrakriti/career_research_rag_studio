from __future__ import annotations

import argparse
from pathlib import Path

from rag_studio.agents.graph import CareerResearchAgent
from rag_studio.agents.trace import format_trace
from rag_studio.agents.langsmith import configure_langsmith, langsmith_run_config


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
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print local trace events for the LangGraph run.",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Send this LangGraph run to LangSmith. Requires LANGSMITH_API_KEY.",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project name. Defaults to career-research-rag-studio.",
    )
    args = parser.parse_args()
    try:
        langsmith_settings = configure_langsmith(
            enabled=args.langsmith,
            project=args.langsmith_project,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    agent = CareerResearchAgent()
    agent.ingest([Path(path) for path in args.docs])
    decision = agent.route(args.question)
    state = agent.run(
        args.question,
        top_k=args.top_k,
        run_config=langsmith_run_config(
            langsmith_settings,
            run_name="career-research-agent",
        ),
    )
    result = state["answer"]

    if args.show_route:
        print(f"Route: {decision.route}")
        print(f"Retriever: {decision.retriever}")
        print(f"Parent context: {decision.parent_context}")
        print(f"Multi-query: {decision.multi_query}")
        print(f"HyDE: {decision.hyde}")
        print(f"Reason: {decision.reason}")
        print()
    if args.show_trace:
        print(format_trace(state.get("trace", [])))
        print()
    if langsmith_settings.enabled:
        print(f"LangSmith tracing: enabled ({langsmith_settings.project})")
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
