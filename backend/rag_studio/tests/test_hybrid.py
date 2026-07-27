from rag_studio.retrieval.hybrid import reciprocal_rank_fusion
from rag_studio.schema import Chunk, RetrievedChunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, text=text, metadata={})


def test_reciprocal_rank_fusion_prioritizes_chunks_that_appear_in_both_lists() -> None:
    chunk_a = make_chunk("a", "PyTorch CNNs BiLSTM deep learning")
    chunk_b = make_chunk("b", "multi-agent system Gemini Cloud Run")
    chunk_c = make_chunk("c", "SQL database travel booking system")
    chunk_d = make_chunk("d", "React Node.js portfolio website")

    dense = [
        RetrievedChunk(chunk=chunk_b, score=0.92),
        RetrievedChunk(chunk=chunk_a, score=0.85),
        RetrievedChunk(chunk=chunk_c, score=0.71),
    ]

    bm25 = [
        RetrievedChunk(chunk=chunk_a, score=12.4),
        RetrievedChunk(chunk=chunk_b, score=9.8),
        RetrievedChunk(chunk=chunk_d, score=6.1),
    ]

    results = reciprocal_rank_fusion(dense, bm25, top_k=3)

    assert [result.chunk.id for result in results] == ["b", "a", "c"]
    assert results[0].score > results[2].score
    assert results[1].score > results[2].score
