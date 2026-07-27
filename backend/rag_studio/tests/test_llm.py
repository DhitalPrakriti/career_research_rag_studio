import pytest

from rag_studio.llm import (
    DEFAULT_GEMINI_MODEL,
    gemini_api_key,
    gemini_is_configured,
    gemini_model_name,
    generate_with_gemini,
)


@pytest.fixture(autouse=True)
def clear_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_model_name_defaults_when_unset() -> None:
    assert gemini_model_name() == DEFAULT_GEMINI_MODEL


def test_model_name_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    assert gemini_model_name() == "gemini-3.5-flash-lite"


def test_not_configured_without_a_key() -> None:
    assert gemini_is_configured() is False
    assert gemini_api_key() is None


def test_gemini_api_key_is_preferred_over_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gemini_api_key() == "gemini-key"


def test_google_api_key_is_accepted_as_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert gemini_is_configured() is True
    assert gemini_api_key() == "google-key"


def test_generate_raises_a_helpful_error_without_a_key() -> None:
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate_with_gemini("Any prompt")
