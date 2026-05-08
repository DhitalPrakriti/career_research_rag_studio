from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_studio.bm25 import BM25Retriever
from rag_studio.chunker import WordChunker
from rag_studio.config import RagConfig
from rag_studio.embeddings import SentenceTransformerEmbedder
from rag_studio.generation import AnswerGenerator, build_citations
from rag_studio.hybrid import reciprocal_rank_fusion
from rag_studio.hyde import HydeGenerator
from rag_studio.loader import load_documents
from rag_studio.multi_query import generate_query_variants, reciprocal_rank_fusion_many
from rag_studio.parent_child import ParentChildChunker, ParentContextResolver
from rag_studio.reranker import CrossEncoderReranker
from rag_studio.schema import Chunk, RagAnswer, RetrievedChunk
from rag_studio.vector_store import FaissVectorStore



class RagPipeline:
    def __init__(self, config: RagConfig | None = None) -> None:
        self.config = config or RagConfig.from_env()
        self.chunker = WordChunker(self.config.chunk_size, self.config.chunk_overlap)
        self.parent_child_chunker = ParentChildChunker(
            parent_chunk_size=self.config.parent_chunk_size,
            parent_chunk_overlap=self.config.parent_chunk_overlap,
            child_chunk_size=self.config.child_chunk_size,
            child_chunk_overlap=self.config.child_chunk_overlap,
        )
        self.embedder: SentenceTransformerEmbedder | None = None
        self.reranker: CrossEncoderReranker | None = None
        self.vector_store = FaissVectorStore()
        self.bm25_retriever = BM25Retriever()
        self.parent_resolver = ParentContextResolver()
        self.generator = AnswerGenerator(max_context_chars=self.config.max_context_chars)
        self.hyde_generator = HydeGenerator()
        self.chunks: list[Chunk] = []
        self.parent_chunks: list[Chunk] = []

    def ingest(
        self,
        paths: list[str | Path],
        build_dense_index: bool = True,
        parent_child: bool = False,
    ) -> list[Chunk]:
        documents = load_documents(paths)
        if parent_child:
            parent_chunks, chunks = self.parent_child_chunker.split(documents)
            self.parent_resolver.add(parent_chunks)
            self.parent_chunks = parent_chunks
        else:
            chunks = self.chunker.split(documents)
            self.parent_resolver.add([])
            self.parent_chunks = []

        if not chunks:
            raise ValueError("No text chunks were created from the provided documents")

        if build_dense_index:
            vectors = self._embedder().embed([chunk.text for chunk in chunks])
            self.vector_store.add(chunks, vectors)
        self.bm25_retriever.add(chunks)
        self.chunks = chunks
        return chunks

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        retriever: str = "dense",
        rerank: bool = False,
        candidate_k: int | None = None,
        parent_context: bool = False,
        multi_query: bool = False,
        max_queries: int = 3,
        hyde: bool = False,
    ) -> list[RetrievedChunk]:
        final_top_k = top_k or self.config.top_k
        retrieval_top_k = candidate_k or (final_top_k * 3 if rerank else final_top_k)
        child_top_k = retrieval_top_k * 3 if parent_context else retrieval_top_k
        retrieval_question = self.hyde_generator.generate(question) if hyde else question
        queries = (
            generate_query_variants(retrieval_question, max_queries)
            if multi_query
            else [retrieval_question]
        )
        ranked_lists = [
            self._retrieve_single_query(
                query,
                child_top_k,
                metadata_filter=metadata_filter,
                retriever=retriever,
            )
            for query in queries
        ]
        results = (
            reciprocal_rank_fusion_many(ranked_lists, top_k=child_top_k)
            if multi_query
            else ranked_lists[0]
        )

        if parent_context:
            results = self.parent_resolver.resolve(results, retrieval_top_k)

        if rerank:
            return self._reranker().rerank(question, results, final_top_k)
        return results[:final_top_k]

    def _retrieve_single_query(
        self,
        question: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
        retriever: str,
    ) -> list[RetrievedChunk]:
        if retriever == "bm25":
            return self.bm25_retriever.search(
                question,
                top_k,
                metadata_filter=metadata_filter,
            )
        if retriever == "hybrid":
            dense_results = self._dense_search(question, top_k, metadata_filter)
            bm25_results = self.bm25_retriever.search(
                question,
                top_k,
                metadata_filter=metadata_filter,
            )
            return reciprocal_rank_fusion(
                dense_results,
                bm25_results,
                top_k=top_k,
            )
        if retriever == "dense":
            return self._dense_search(question, top_k, metadata_filter)

        raise ValueError(
            f"Unsupported retriever '{retriever}'. Use 'dense', 'bm25', or 'hybrid'."
        )

    def _dense_search(
        self,
        question: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[RetrievedChunk]:
        query_vector = self._embedder().embed([question])
        return self.vector_store.search(
            query_vector,
            top_k,
            metadata_filter=metadata_filter,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
        retriever: str = "dense",
        rerank: bool = False,
        candidate_k: int | None = None,
        parent_context: bool = False,
        multi_query: bool = False,
        max_queries: int = 3,
        hyde: bool = False,
    ) -> RagAnswer:
        contexts = self.retrieve(
            question,
            top_k,
            metadata_filter=metadata_filter,
            retriever=retriever,
            rerank=rerank,
            candidate_k=candidate_k,
            parent_context=parent_context,
            multi_query=multi_query,
            max_queries=max_queries,
            hyde=hyde,
        )
        answer_text = self.generator.generate(question, contexts)
        return RagAnswer(
            question=question,
            answer=answer_text,
            citations=build_citations(contexts),
            contexts=contexts,
        )

    def _embedder(self) -> SentenceTransformerEmbedder:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.config.embedding_model)
        return self.embedder

    def _reranker(self) -> CrossEncoderReranker:
        if self.reranker is None:
            self.reranker = CrossEncoderReranker(self.config.reranker_model)
        return self.reranker

