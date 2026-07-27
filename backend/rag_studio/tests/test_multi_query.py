from rag_studio.retrieval.multi_query import generate_query_variants, reciprocal_rank_fusion_many
from rag_studio.schema import Chunk, RetrievedChunk


def test_generate_query_variants_keeps_original_and_adds_keyword_variant() -> None:
    variants = generate_query_variants("What AI skills does my resume show?", max_queries=3)

    assert variants[0] == "What AI skills does my resume show?"
    assert "AI skills resume show" in variants
    assert len(variants) <= 3


def test_reciprocal_rank_fusion_many_merges_multiple_ranked_lists() -> None:
    chunk_a = Chunk(id="a", text="AI skills", metadata={})
    chunk_b = Chunk(id="b", text="Cloud skills", metadata={})
    chunk_c = Chunk(id="c", text="SQL skills", metadata={})

    results = reciprocal_rank_fusion_many(
        [
            [
                RetrievedChunk(chunk=chunk_a, score=0.9),
                RetrievedChunk(chunk=chunk_b, score=0.8),
            ],
            [
                RetrievedChunk(chunk=chunk_b, score=10.0),
                RetrievedChunk(chunk=chunk_c, score=9.0),
            ],
        ],
        top_k=3,
    )

    assert [result.chunk.id for result in results] == ["b", "a", "c"]
