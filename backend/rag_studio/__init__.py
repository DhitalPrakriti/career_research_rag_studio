"""Career Research RAG Studio baseline package."""

__all__ = ["RagPipeline"]


def __getattr__(name: str):
    if name == "RagPipeline":
        from rag_studio.pipeline import RagPipeline

        return RagPipeline
    raise AttributeError(name)
