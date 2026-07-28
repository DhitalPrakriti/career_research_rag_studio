"""Retrieval strategies: sparse, dense fusion, reranking, and query expansion."""

from rag_studio.retrieval.bm25 import BM25Retriever
from rag_studio.retrieval.hybrid import reciprocal_rank_fusion
from rag_studio.retrieval.hyde import HydeGenerator
from rag_studio.retrieval.multi_query import (
    generate_query_variants,
    reciprocal_rank_fusion_many,
)
from rag_studio.retrieval.reranker import CrossEncoderReranker

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "HydeGenerator",
    "generate_query_variants",
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_many",
]
