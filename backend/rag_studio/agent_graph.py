from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from rag_studio.pipeline import RagPipeline
from rag_studio.query_router import QueryRouter, RouteDecision
from rag_studio.schema import RagAnswer


class AgentState(TypedDict, total=False):
    question: str
    top_k: int | None
    decision: RouteDecision
    answer: RagAnswer


class CareerResearchAgent:
    def __init__(
        self,
        pipeline: RagPipeline | None = None,
        router: QueryRouter | None = None,
    ) -> None:
        self.pipeline = pipeline or RagPipeline()
        self.router = router or QueryRouter()
        self.graph = self._build_graph()

    def ingest(self, paths: list[str | Path]) -> None:
        self.pipeline.ingest(
            paths,
            build_dense_index=True,
            parent_child=True,
        )

    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        state = self.graph.invoke({"question": question, "top_k": top_k})
        return state["answer"]

    def route(self, question: str) -> RouteDecision:
        return self.router.route(question)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_query", self._route_query)
        graph.add_node("direct_answer", self._direct_answer)
        graph.add_node("retrieve_answer", self._retrieve_answer)
        graph.set_entry_point("route_query")
        graph.add_conditional_edges(
            "route_query",
            _next_node,
            {
                "direct": "direct_answer",
                "retrieve": "retrieve_answer",
            },
        )
        graph.add_edge("direct_answer", END)
        graph.add_edge("retrieve_answer", END)
        return graph.compile()

    def _route_query(self, state: AgentState) -> dict[str, Any]:
        return {"decision": self.router.route(state["question"])}

    def _direct_answer(self, state: AgentState) -> dict[str, Any]:
        answer = RagAnswer(
            question=state["question"],
            answer=(
                "Ask a career research question about the provided resumes, job "
                "descriptions, notes, or projects."
            ),
            citations=[],
            contexts=[],
        )
        return {"answer": answer}

    def _retrieve_answer(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        answer = self.pipeline.answer(
            state["question"],
            top_k=state.get("top_k"),
            retriever=decision.retriever,
            parent_context=decision.parent_context,
            multi_query=decision.multi_query,
            hyde=decision.hyde,
        )
        return {"answer": answer}


def _next_node(state: AgentState) -> str:
    return state["decision"].route
