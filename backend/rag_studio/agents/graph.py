from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from rag_studio.pipeline import RagPipeline
from rag_studio.agents.router import QueryRouter, RouteDecision
from rag_studio.generation.generator import build_citations
from rag_studio.agents.grader import RetrievalGrade, RetrievalGrader
from rag_studio.agents.rewriter import QueryRewriter
from rag_studio.schema import RagAnswer, RetrievedChunk
from rag_studio.agents.trace import AgentTraceEvent, add_trace_event


class AgentState(TypedDict, total=False):
    question: str
    retrieval_question: str
    top_k: int | None
    retry_count: int
    decision: RouteDecision
    contexts: list[RetrievedChunk]
    retrieval_grade: RetrievalGrade
    answer: RagAnswer
    trace: list[AgentTraceEvent]


class CareerResearchAgent:
    def __init__(
        self,
        pipeline: RagPipeline | None = None,
        router: QueryRouter | None = None,
        retrieval_grader: RetrievalGrader | None = None,
        query_rewriter: QueryRewriter | None = None,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.pipeline = pipeline or RagPipeline()
        self.router = router or QueryRouter()
        self.retrieval_grader = retrieval_grader or RetrievalGrader()
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.max_retries = max_retries
        self.graph = self._build_graph()

    def ingest(self, paths: list[str | Path]) -> None:
        self.pipeline.ingest(
            paths,
            build_dense_index=True,
            parent_child=True,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        run_config: dict[str, object] | None = None,
    ) -> RagAnswer:
        return self.run(question, top_k=top_k, run_config=run_config)["answer"]

    def run(
        self,
        question: str,
        top_k: int | None = None,
        run_config: dict[str, object] | None = None,
    ) -> AgentState:
        state = self.graph.invoke(
            {
                "question": question,
                "retrieval_question": question,
                "top_k": top_k,
                "retry_count": 0,
                "trace": [],
            },
            config=run_config,
        )
        return state

    def route(self, question: str) -> RouteDecision:
        return self.router.route(question)

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("route_query", self._route_query)
        graph.add_node("direct_answer", self._direct_answer)
        graph.add_node("retrieve_answer", self._retrieve_answer)
        graph.add_node("grade_retrieval", self._grade_retrieval)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate", self._generate)
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
        graph.add_edge("retrieve_answer", "grade_retrieval")
        graph.add_conditional_edges(
            "grade_retrieval",
            self._after_grade,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            },
        )
        graph.add_edge("rewrite_query", "retrieve_answer")
        graph.add_edge("generate", END)
        return graph.compile()

    def _route_query(self, state: AgentState) -> dict[str, Any]:
        decision = self.router.route(state["question"])
        trace_details: dict[str, Any] = {
            "route": decision.route,
            "retriever": decision.retriever,
            "parent_context": decision.parent_context,
            "multi_query": decision.multi_query,
            "hyde": decision.hyde,
            "reason": decision.reason,
        }
        update: dict[str, Any] = {
            "decision": decision,
            "trace": add_trace_event(
                state.get("trace", []),
                "route_query",
                "Selected route.",
                **trace_details,
            ),
        }
        if decision.rewrite_before_retrieval:
            rewritten = self.query_rewriter.rewrite(state["question"])
            update["retrieval_question"] = rewritten
            update["trace"] = add_trace_event(
                update["trace"],
                "rewrite_query",
                "Rewrote query before first retrieval.",
                rewritten_query=rewritten,
                retry_count=0,
            )
        return update

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
        return {
            "answer": answer,
            "trace": add_trace_event(
                state.get("trace", []),
                "direct_answer",
                "Answered without retrieval.",
            ),
        }

    def _retrieve_answer(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        contexts = self.pipeline.retrieve(
            state.get("retrieval_question", state["question"]),
            top_k=state.get("top_k"),
            retriever=decision.retriever,
            parent_context=decision.parent_context,
            multi_query=decision.multi_query,
            hyde=decision.hyde,
        )
        if decision.rewrite_before_retrieval:
            contexts = self.retrieval_grader.rank(
                state.get("retrieval_question", state["question"]),
                contexts,
            )
        return {
            "contexts": contexts,
            "trace": add_trace_event(
                state.get("trace", []),
                "retrieve_answer",
                "Retrieved context chunks.",
                retrieval_question=state.get("retrieval_question", state["question"]),
                chunks=len(contexts),
            ),
        }

    def _grade_retrieval(self, state: AgentState) -> dict[str, Any]:
        grade = self.retrieval_grader.grade(state["question"], state.get("contexts", []))
        return {
            "retrieval_grade": grade,
            "trace": add_trace_event(
                state.get("trace", []),
                "grade_retrieval",
                "Graded retrieved context relevance.",
                is_relevant=grade.is_relevant,
                score=f"{grade.score:.3f}",
                reason=grade.reason,
                retry_count=state.get("retry_count", 0),
            ),
        }

    def _rewrite_query(self, state: AgentState) -> dict[str, Any]:
        rewritten = self.query_rewriter.rewrite(state["question"])
        return {
            "retrieval_question": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": add_trace_event(
                state.get("trace", []),
                "rewrite_query",
                "Rewrote query for another retrieval attempt.",
                rewritten_query=rewritten,
                retry_count=state.get("retry_count", 0) + 1,
            ),
        }

    def _generate(self, state: AgentState) -> dict[str, Any]:
        contexts = state.get("contexts", [])
        grade = state["retrieval_grade"]
        if not grade.is_relevant:
            answer_text = (
                "I found retrieved context, but it does not look relevant enough to answer "
                f"confidently. {grade.reason}"
            )
        else:
            answer_text = self.pipeline.generator.generate(state["question"], contexts)

        answer = RagAnswer(
            question=state["question"],
            answer=answer_text,
            citations=build_citations(contexts),
            contexts=contexts,
        )
        return {
            "answer": answer,
            "trace": add_trace_event(
                state.get("trace", []),
                "generate",
                "Produced final answer.",
                citations=len(answer.citations),
                retry_count=state.get("retry_count", 0),
            ),
        }

    def _after_grade(self, state: AgentState) -> str:
        grade = state["retrieval_grade"]
        if grade.is_relevant:
            return "generate"
        if state.get("retry_count", 0) < self.max_retries:
            return "rewrite_query"
        return "generate"


def _next_node(state: AgentState) -> str:
    return state["decision"].route
