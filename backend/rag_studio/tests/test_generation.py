from rag_studio.generation import _build_grounded_prompt, trim_contexts
from rag_studio.schema import Chunk, RetrievedChunk


def test_grounded_prompt_discourages_contact_details_and_requires_citations() -> None:
    prompt = _build_grounded_prompt(
        "What AI skills does my resume show?",
        [
            RetrievedChunk(
                chunk=Chunk(
                    id="chunk-1",
                    text="Email: person@example.com. Skills: PyTorch and RAG pipelines.",
                    metadata={},
                ),
                score=0.9,
            )
        ],
    )

    assert "Cite each factual bullet or sentence" in prompt
    assert "Do not include contact details" in prompt
    assert "Question: What AI skills does my resume show?" in prompt
    assert "[1] Email: person@example.com" in prompt


def test_trim_contexts_keeps_highest_ranked_chunks_within_budget() -> None:
    contexts = [
        RetrievedChunk(chunk=Chunk(id="a", text="a" * 10, metadata={}), score=0.9),
        RetrievedChunk(chunk=Chunk(id="b", text="b" * 10, metadata={}), score=0.8),
        RetrievedChunk(chunk=Chunk(id="c", text="c" * 10, metadata={}), score=0.7),
    ]

    trimmed = trim_contexts(contexts, max_context_chars=22)

    assert [result.chunk.id for result in trimmed] == ["a", "b"]


def test_trim_contexts_truncates_first_chunk_if_it_exceeds_budget() -> None:
    contexts = [
        RetrievedChunk(chunk=Chunk(id="large", text="x" * 50, metadata={}), score=0.9)
    ]

    trimmed = trim_contexts(contexts, max_context_chars=12)

    assert len(trimmed) == 1
    assert trimmed[0].chunk.id == "large"
    assert trimmed[0].chunk.text == "x" * 9 + "..."
