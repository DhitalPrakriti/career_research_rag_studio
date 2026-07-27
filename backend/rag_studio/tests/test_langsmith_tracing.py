import pytest

from rag_studio.agents.langsmith import configure_langsmith, langsmith_run_config


def test_configure_langsmith_returns_disabled_settings_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    settings = configure_langsmith(enabled=False)

    assert settings.enabled is False
    assert settings.project == "career-research-rag-studio"
    assert __import__("os").environ["LANGSMITH_TRACING"] == "false"


def test_configure_langsmith_requires_api_key_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY"):
        configure_langsmith(enabled=True)


def test_configure_langsmith_sets_tracing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    settings = configure_langsmith(enabled=True, project="test-project")

    assert settings.enabled is True
    assert settings.project == "test-project"
    assert langsmith_run_config(settings, "agent") == {
        "run_name": "agent",
        "tags": ["career-rag-studio", "langgraph-agent"],
        "metadata": {
            "project": "test-project",
            "component": "CareerResearchAgent",
        },
    }


def test_langsmith_run_config_is_none_when_disabled() -> None:
    settings = configure_langsmith(enabled=False)

    assert langsmith_run_config(settings, "agent") is None
