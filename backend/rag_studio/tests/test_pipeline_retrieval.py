import numpy as np

from rag_studio.pipeline import RagPipeline
from rag_studio.schema import Chunk, RetrievedChunk


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "pytorch" in lowered or "transformer" in lowered:
                vectors.append([1.0, 0.0])
            elif "cloud" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype="float32")


class FakeReranker:
    def rerank(
        self,
        question: str,
        results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        reranked = sorted(results, key=lambda result: result.chunk.id, reverse=True)
        return [
            RetrievedChunk(chunk=result.chunk, score=100.0 - index)
            for index, result in enumerate(reranked[:top_k])
        ]


class FakeHydeGenerator:
    def generate(self, question: str) -> str:
        return "PyTorch Transformer skills"


def test_pipeline_hybrid_retriever_combines_dense_and_bm25() -> None:
    pipeline = RagPipeline()
    pipeline.embedder = FakeEmbedder()  # type: ignore[assignment]

    chunks = [
        Chunk(id="ml", text="PyTorch Transformer deep learning", metadata={}),
        Chunk(id="cloud", text="Cloud Run Docker deployment", metadata={}),
    ]
    vectors = pipeline.embedder.embed([chunk.text for chunk in chunks])
    pipeline.vector_store.add(chunks, vectors)
    pipeline.bm25_retriever.add(chunks)

    results = pipeline.retrieve("PyTorch Transformer", top_k=1, retriever="hybrid")

    assert [result.chunk.id for result in results] == ["ml"]


def test_pipeline_can_rerank_retrieved_candidates() -> None:
    pipeline = RagPipeline()
    pipeline.embedder = FakeEmbedder()  # type: ignore[assignment]
    pipeline.reranker = FakeReranker()  # type: ignore[assignment]

    chunks = [
        Chunk(id="a", text="PyTorch Transformer deep learning", metadata={}),
        Chunk(id="b", text="PyTorch production model deployment", metadata={}),
    ]
    vectors = pipeline.embedder.embed([chunk.text for chunk in chunks])
    pipeline.vector_store.add(chunks, vectors)
    pipeline.bm25_retriever.add(chunks)

    results = pipeline.retrieve(
        "PyTorch",
        top_k=1,
        retriever="hybrid",
        rerank=True,
        candidate_k=2,
    )

    assert [result.chunk.id for result in results] == ["b"]
    assert results[0].score == 100.0


def test_pipeline_multi_query_merges_results_from_query_variants() -> None:
    pipeline = RagPipeline()
    pipeline.embedder = FakeEmbedder()  # type: ignore[assignment]

    chunks = [
        Chunk(id="ml", text="PyTorch Transformer deep learning", metadata={}),
        Chunk(id="cloud", text="Cloud Run Docker deployment", metadata={}),
    ]
    vectors = pipeline.embedder.embed([chunk.text for chunk in chunks])
    pipeline.vector_store.add(chunks, vectors)
    pipeline.bm25_retriever.add(chunks)

    results = pipeline.retrieve(
        "What PyTorch Transformer skills are shown?",
        top_k=2,
        retriever="hybrid",
        multi_query=True,
    )

    assert results
    assert results[0].chunk.id == "ml"


def test_pipeline_hyde_uses_hypothetical_answer_for_retrieval() -> None:
    pipeline = RagPipeline()
    pipeline.embedder = FakeEmbedder()  # type: ignore[assignment]
    pipeline.hyde_generator = FakeHydeGenerator()  # type: ignore[assignment]

    chunks = [
        Chunk(id="ml", text="PyTorch Transformer deep learning", metadata={}),
        Chunk(id="cloud", text="Cloud Run Docker deployment", metadata={}),
    ]
    vectors = pipeline.embedder.embed([chunk.text for chunk in chunks])
    pipeline.vector_store.add(chunks, vectors)
    pipeline.bm25_retriever.add(chunks)

    results = pipeline.retrieve(
        "What technical background is shown?",
        top_k=1,
        retriever="hybrid",
        hyde=True,
    )

    assert [result.chunk.id for result in results] == ["ml"]
