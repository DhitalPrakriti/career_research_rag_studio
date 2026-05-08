from __future__ import annotations

from rag_studio.chunker import WordChunker
from rag_studio.schema import Chunk, Document, RetrievedChunk


class ParentChildChunker:
    def __init__(
        self,
        parent_chunk_size: int = 520,
        parent_chunk_overlap: int = 100,
        child_chunk_size: int = 160,
        child_chunk_overlap: int = 40,
    ) -> None:
        self.parent_chunker = WordChunker(parent_chunk_size, parent_chunk_overlap)
        self.child_chunker = WordChunker(child_chunk_size, child_chunk_overlap)

    def split(self, documents: list[Document]) -> tuple[list[Chunk], list[Chunk]]:
        parents = self.parent_chunker.split(documents)
        parent_documents = [
            Document(
                text=parent.text,
                metadata={
                    **parent.metadata,
                    "parent_id": parent.id,
                    "parent_chunk_index": parent.metadata.get("chunk_index"),
                },
            )
            for parent in parents
        ]
        children = self.child_chunker.split(parent_documents)
        children = [
            Chunk(
                id=child.id,
                text=child.text,
                metadata={
                    **child.metadata,
                    "child_chunk_index": child.metadata.get("chunk_index"),
                },
            )
            for child in children
        ]
        return parents, children


class ParentContextResolver:
    def __init__(self) -> None:
        self._parents_by_id: dict[str, Chunk] = {}

    def add(self, parents: list[Chunk]) -> None:
        self._parents_by_id = {parent.id: parent for parent in parents}

    def resolve(self, child_results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        parent_results: list[RetrievedChunk] = []
        seen_parent_ids: set[str] = set()

        for result in child_results:
            parent_id = result.chunk.metadata.get("parent_id")
            if not isinstance(parent_id, str) or parent_id in seen_parent_ids:
                continue
            parent = self._parents_by_id.get(parent_id)
            if parent is None:
                continue
            seen_parent_ids.add(parent_id)
            parent_results.append(RetrievedChunk(chunk=parent, score=result.score))
            if len(parent_results) >= top_k:
                break

        return parent_results
