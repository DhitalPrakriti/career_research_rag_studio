import pytest

from rag_studio.llm import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LITELLM_BASE_URL,
    DEFAULT_LITELLM_MODEL,
    gemini_api_key,
    gemini_is_configured,
    gemini_model_name,
    generate_with_gemini,
    litellm_api_key,
    litellm_base_url,
    resolve_provider,
)

_ENV_VARS = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "LITELLM_API_KEY",
    "LITELLM_MASTER_KEY",
    "LITELLM_MODEL",
    "LITELLM_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OLLAMA_MODEL",
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_model_name_defaults_when_unset() -> None:
    assert gemini_model_name() == DEFAULT_GEMINI_MODEL


def test_model_name_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    assert gemini_model_name() == "gemini-3.5-flash-lite"


def test_gemini_api_key_is_preferred_over_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    assert gemini_api_key() == "gemini-key"


def test_google_api_key_is_accepted_as_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert gemini_is_configured() is True


def test_litellm_master_key_is_accepted_as_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", "master-key")

    assert litellm_api_key() == "master-key"


def test_litellm_base_url_strips_trailing_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/")

    assert litellm_base_url() == "http://localhost:4000"


def test_generate_raises_a_helpful_error_without_a_key() -> None:
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        generate_with_gemini("Any prompt")


class TestResolveProvider:
    def test_falls_back_to_extractive_with_no_credentials(self) -> None:
        config = resolve_provider()

        assert config.provider == "extractive"
        assert config.is_llm is False

    def test_auto_detects_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "key")
        config = resolve_provider()

        assert config.provider == "gemini"
        assert config.model == DEFAULT_GEMINI_MODEL
        assert config.is_llm is True

    def test_auto_detection_prefers_litellm_over_gemini(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The proxy keeps budget caps and caching in the path, so prefer it."""
        monkeypatch.setenv("GEMINI_API_KEY", "key")
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")
        config = resolve_provider()

        assert config.provider == "litellm"
        assert config.model == DEFAULT_LITELLM_MODEL
        assert config.base_url == DEFAULT_LITELLM_BASE_URL

    def test_explicit_provider_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "key")
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        assert resolve_provider().provider == "gemini"

    def test_explicit_provider_without_credentials_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming a provider then silently using another is worse than failing."""
        monkeypatch.setenv("LLM_PROVIDER", "litellm")
        monkeypatch.setenv("GEMINI_API_KEY", "key")

        with pytest.raises(RuntimeError, match="LITELLM_API_KEY"):
            resolve_provider()

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "bedrock")

        with pytest.raises(RuntimeError, match="not recognised"):
            resolve_provider()

    def test_extractive_can_be_requested_explicitly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "extractive")
        monkeypatch.setenv("GEMINI_API_KEY", "key")
        config = resolve_provider()

        assert config.provider == "extractive"
        assert config.is_llm is False

    def test_describe_is_readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        assert resolve_provider().describe() == (
            f"litellm / {DEFAULT_LITELLM_MODEL} via {DEFAULT_LITELLM_BASE_URL}"
        )
