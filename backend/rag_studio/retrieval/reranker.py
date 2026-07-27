from __future__ import annotations

from rag_studio.schema import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use cross-encoder reranking."
            ) from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not results:
            return []

        pairs = [(question, result.chunk.text) for result in results]
        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked = [
            RetrievedChunk(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]
