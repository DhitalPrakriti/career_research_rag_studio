from rag_studio.retrieval.bm25 import BM25Retriever
from rag_studio.schema import Chunk


def test_bm25_ranks_exact_keyword_matches_first() -> None:
    retriever = BM25Retriever()
    retriever.add(
        [
            Chunk(id="python", text="Python PyTorch Transformer machine learning", metadata={}),
            Chunk(id="sql", text="SQL database reporting dashboard", metadata={}),
            Chunk(id="cloud", text="Docker Cloud Run deployment", metadata={}),
        ]
    )

    results = retriever.search("PyTorch Transformer", top_k=2)

    assert [result.chunk.id for result in results] == ["python"]
    assert results[0].score > 0


def test_bm25_applies_metadata_filter() -> None:
    retriever = BM25Retriever()
    retriever.add(
        [
            Chunk(
                id="resume",
                text="Python PyTorch Transformer machine learning",
                metadata={"doc_type": "resume"},
            ),
            Chunk(
                id="job",
                text="Python PyTorch production requirements",
                metadata={"doc_type": "job_description"},
            ),
        ]
    )

    results = retriever.search(
        "Python PyTorch",
        top_k=2,
        metadata_filter={"doc_type": "job_description"},
    )

    assert [result.chunk.id for result in results] == ["job"]
