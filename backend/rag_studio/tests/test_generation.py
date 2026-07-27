import pytest

from rag_studio.generation import generator as generator_module
from rag_studio.generation.generator import (
    AnswerGenerator,
    _build_grounded_prompt,
    _ollama_options,
    trim_contexts,
)
from rag_studio.schema import Chunk, RetrievedChunk


@pytest.fixture
def one_context() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(id="chunk-1", text="Achieved 94.28% Binary F1.", metadata={}),
            score=0.9,
        )
    ]


@pytest.fixture(autouse=True)
def clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OLLAMA_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_ollama_options_reads_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_NUM_GPU", "0")
    monkeypatch.setenv("OLLAMA_NUM_THREAD", "4")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "256")

    assert _ollama_options() == {
        "num_gpu": 0,
        "num_thread": 4,
        "num_predict": 256,
    }


def test_ollama_options_rejects_non_integer_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_NUM_GPU", "none")

    with pytest.raises(RuntimeError, match="OLLAMA_NUM_GPU"):
        _ollama_options()


def test_gemini_is_used_when_a_key_is_present(
    monkeypatch: pytest.MonkeyPatch,
    one_context: list[RetrievedChunk],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    def fake_generate(prompt: str, system_instruction: str | None = None, **kwargs: object) -> str:
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return "94.28% Binary F1 [1]."

    monkeypatch.setattr(generator_module, "generate_with_gemini", fake_generate)

    answer = AnswerGenerator().generate("What binary F1 score was achieved?", one_context)

    assert answer == "94.28% Binary F1 [1]."
    assert "Achieved 94.28% Binary F1." in str(captured["prompt"])
    assert "cite claims inline" in str(captured["system_instruction"])


def test_gemini_takes_precedence_over_openai_and_ollama(
    monkeypatch: pytest.MonkeyPatch,
    one_context: list[RetrievedChunk],
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setattr(
        generator_module,
        "generate_with_gemini",
        lambda *args, **kwargs: "from gemini",
    )

    assert AnswerGenerator().generate("Question?", one_context) == "from gemini"


def test_configured_gemini_failure_raises_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
    one_context: list[RetrievedChunk],
) -> None:
    """A broken key must not silently produce extractive text scored as generation."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fail(*args: object, **kwargs: object) -> str:
        raise RuntimeError("Gemini request failed for model gemini-3.6-flash: 401")

    monkeypatch.setattr(generator_module, "generate_with_gemini", fail)

    with pytest.raises(RuntimeError, match="Gemini request failed"):
        AnswerGenerator().generate("Question?", one_context)


def test_extractive_fallback_when_no_provider_is_configured(
    one_context: list[RetrievedChunk],
) -> None:
    answer = AnswerGenerator().generate("Question?", one_context)

    assert "No LLM model is configured" in answer
