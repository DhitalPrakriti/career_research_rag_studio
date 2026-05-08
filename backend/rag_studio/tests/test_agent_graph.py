from pathlib import Path
from typing import Any

from rag_studio.agent_graph import CareerResearchAgent
from rag_studio.query_router import QueryRouter
from rag_studio.schema import RagAnswer


class FakePipeline:
    def __init__(self) -> None:
        self.ingested_paths: list[str | Path] = []
        self.answer_kwargs: dict[str, Any] = {}

    def ingest(
        self,
        paths: list[str | Path],
        build_dense_index: bool = True,
        parent_child: bool = False,
    ) -> None:
        self.ingested_paths = paths
        self.answer_kwargs["build_dense_index"] = build_dense_index
        self.answer_kwargs["parent_child"] = parent_child

    def answer(self, question: str, **kwargs: Any) -> RagAnswer:
        self.answer_kwargs.update(kwargs)
        return RagAnswer(
            question=question,
            answer="retrieved answer",
            citations=[],
            contexts=[],
        )


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

    assert result.answer == "retrieved answer"
    assert pipeline.answer_kwargs["top_k"] == 2
    assert pipeline.answer_kwargs["retriever"] == "hybrid"
    assert pipeline.answer_kwargs["parent_context"] is True
    assert pipeline.answer_kwargs["multi_query"] is True
    assert pipeline.answer_kwargs["hyde"] is True
