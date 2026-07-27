from rag_studio.retrieval.hyde import HydeGenerator


def test_hyde_generator_has_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    hypothesis = HydeGenerator().generate("What AI skills does my resume show?")

    assert "What AI skills does my resume show?" in hypothesis
    assert "skills" in hypothesis
    assert "projects" in hypothesis
