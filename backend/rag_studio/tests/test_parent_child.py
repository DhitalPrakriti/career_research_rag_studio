import numpy as np

from rag_studio.ingestion.parent_child import ParentChildChunker, ParentContextResolver
from rag_studio.pipeline import RagPipeline
from rag_studio.schema import Chunk, Document, RetrievedChunk


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "target" in text.lower() else [0.0, 1.0])
        return np.asarray(vectors, dtype="float32")


def test_parent_child_chunker_adds_parent_id_to_children() -> None:
    chunker = ParentChildChunker(
        parent_chunk_size=8,
        parent_chunk_overlap=0,
        child_chunk_size=4,
        child_chunk_overlap=0,
    )
    parents, children = chunker.split(
        [
            Document(
                text="one two three target five six seven eight nine ten",
                metadata={"title": "sample"},
            )
        ]
    )

    assert parents
    assert children
    assert all("parent_id" in child.metadata for child in children)
    assert {child.metadata["parent_id"] for child in children}.issubset(
        {parent.id for parent in parents}
    )


def test_parent_context_resolver_deduplicates_parent_chunks() -> None:
    parent = Chunk(id="parent-1", text="large parent context", metadata={})
    child_a = Chunk(id="child-a", text="small child target", metadata={"parent_id": "parent-1"})
    child_b = Chunk(id="child-b", text="another child target", metadata={"parent_id": "parent-1"})
    resolver = ParentContextResolver()
    resolver.add([parent])

    results = resolver.resolve(
        [
            RetrievedChunk(chunk=child_a, score=0.9),
            RetrievedChunk(chunk=child_b, score=0.8),
        ],
        top_k=3,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "parent-1"
    assert results[0].score == 0.9


def test_pipeline_can_return_parent_context_from_child_retrieval() -> None:
    pipeline = RagPipeline()
    pipeline.embedder = FakeEmbedder()  # type: ignore[assignment]

    parent = Chunk(id="parent-1", text="large parent context with target details", metadata={})
    child = Chunk(
        id="child-1",
        text="target",
        metadata={"parent_id": "parent-1"},
    )
    pipeline.parent_resolver.add([parent])
    pipeline.parent_chunks = [parent]
    pipeline.chunks = [child]
    pipeline.vector_store.add([child], pipeline.embedder.embed([child.text]))
    pipeline.bm25_retriever.add([child])

    results = pipeline.retrieve(
        "target",
        top_k=1,
        retriever="hybrid",
        parent_context=True,
    )

    assert [result.chunk.id for result in results] == ["parent-1"]
