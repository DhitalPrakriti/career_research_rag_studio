from pathlib import Path
from typing import Any

from rag_studio.agents.graph import CareerResearchAgent
from rag_studio.agents.router import QueryRouter
from rag_studio.schema import Chunk, RagAnswer, RetrievedChunk


class FakeGenerator:
    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        return "generated answer"


class FakePipeline:
    def __init__(self) -> None:
        self.ingested_paths: list[str | Path] = []
        self.answer_kwargs: dict[str, Any] = {}
        self.retrieval_questions: list[str] = []
        self.generator = FakeGenerator()

    def ingest(
        self,
        paths: list[str | Path],
        build_dense_index: bool = True,
        parent_child: bool = False,
    ) -> None:
        self.ingested_paths = paths
        self.answer_kwargs["build_dense_index"] = build_dense_index
        self.answer_kwargs["parent_child"] = parent_child

    def retrieve(self, question: str, **kwargs: Any) -> list[RetrievedChunk]:
        self.retrieval_questions.append(question)
        self.answer_kwargs.update(kwargs)
        if "Related terms" in question:
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        id="vector-1",
                        text="Vector databases include FAISS and MongoDB.",
                        metadata={"title": "resume.pdf"},
                    ),
                    score=1.0,
                ),
                RetrievedChunk(
                    chunk=Chunk(
                        id="memory-1",
                        text="Firestore was used for multi-turn conversation memory.",
                        metadata={"title": "resume.pdf"},
                    ),
                    score=1.0,
                )
            ]
        return [
            RetrievedChunk(
                chunk=Chunk(
                    id="resume-1",
                    text="This resume is a strong job fit with Python and RAG experience.",
                    metadata={"title": "resume.pdf"},
                ),
                score=1.0,
            )
        ]


def test_agent_ingests_with_parent_child_context_enabled() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    agent.ingest(["resume.pdf"])

    assert pipeline.ingested_paths == ["resume.pdf"]
    assert pipeline.answer_kwargs["build_dense_index"] is True
    assert pipeline.answer_kwargs["parent_child"] is True


def test_agent_graph_returns_direct_answer_without_retrieval() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    result = agent.answer("hello what can you do?")

    assert "career research question" in result.answer
    assert "retriever" not in pipeline.answer_kwargs


def test_agent_graph_calls_pipeline_with_router_settings() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    result = agent.answer("Which resume is the best fit for this job?", top_k=2)

    assert result.answer == "generated answer"
    assert pipeline.answer_kwargs["top_k"] == 2
    assert pipeline.answer_kwargs["retriever"] == "hybrid"
    assert pipeline.answer_kwargs["parent_context"] is True
    assert pipeline.answer_kwargs["multi_query"] is True
    assert pipeline.answer_kwargs["hyde"] is True


def test_agent_run_returns_trace_events() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    state = agent.run("Which resume is the best fit for this job?", top_k=2)

    assert state["answer"].answer == "generated answer"
    assert [event.node for event in state["trace"]] == [
        "route_query",
        "retrieve_answer",
        "grade_retrieval",
        "generate",
    ]


def test_agent_graph_refuses_to_generate_when_retrieval_is_not_relevant() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(
        pipeline=pipeline,
        router=QueryRouter(),
        max_retries=0,
    )  # type: ignore[arg-type]

    result = agent.answer("What API was used for memory?", top_k=2)

    assert "does not look relevant enough" in result.answer


def test_agent_graph_retrieves_with_the_original_question_first() -> None:
    """No pre-emptive rewrite any more.

    The router used to expand a query before the first retrieval when it matched
    "database" AND "memory" — a rule that existed for one golden example. The
    grade-then-rewrite loop covers weak retrieval generally, so the first attempt now
    always uses the question as asked.
    """
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter(mode="rules"))  # type: ignore[arg-type]

    agent.answer("What database was used for memory?", top_k=2)

    assert pipeline.retrieval_questions[0] == "What database was used for memory?"
    assert "Related terms:" not in pipeline.retrieval_questions[0]


def test_agent_graph_reranks_contexts_after_a_rewrite() -> None:
    """A rewritten query carries expansion terms, so ranking matters on the retry."""
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter(mode="rules"))  # type: ignore[arg-type]

    state = agent.run("What API was used for memory?", top_k=2)

    assert state["retry_count"] == 1
    # The Firestore chunk answers the rewritten query better than the FAISS/MongoDB one,
    # and reranking moves it first.
    assert "Firestore" in state["contexts"][0].chunk.text


def test_agent_trace_records_rewrite_loop() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    state = agent.run("What database was used for memory?", top_k=2)

    assert "rewrite_query" in [event.node for event in state["trace"]]


def test_agent_graph_rewrites_query_once_when_retrieval_is_not_relevant() -> None:
    pipeline = FakePipeline()
    agent = CareerResearchAgent(pipeline=pipeline, router=QueryRouter())  # type: ignore[arg-type]

    result = agent.answer("What API was used for memory?", top_k=2)

    assert result.answer == "generated answer"
    assert len(pipeline.retrieval_questions) == 2
    assert pipeline.retrieval_questions[0] == "What API was used for memory?"
    assert "Related terms:" in pipeline.retrieval_questions[1]


def test_agent_rejects_negative_retry_limit() -> None:
    pipeline = FakePipeline()

    try:
        CareerResearchAgent(pipeline=pipeline, max_retries=-1)  # type: ignore[arg-type]
    except ValueError as exc:
        assert "max_retries" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
