from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_model: str | None = None
    chunk_size: int = 260
    chunk_overlap: int = 60
    parent_chunk_size: int = 520
    parent_chunk_overlap: int = 100
    child_chunk_size: int = 160
    child_chunk_overlap: int = 40
    max_context_chars: int = 6000
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "RagConfig":
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model),
            openai_model=os.getenv("OPENAI_MODEL") or None,
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", cls.parent_chunk_size)),
            parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", cls.parent_chunk_overlap)),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", cls.child_chunk_size)),
            child_chunk_overlap=int(os.getenv("CHILD_CHUNK_OVERLAP", cls.child_chunk_overlap)),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", cls.max_context_chars)),
            top_k=int(os.getenv("TOP_K", cls.top_k)),
        )
