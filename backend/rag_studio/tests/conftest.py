"""Shared test setup.

The suite must be deterministic and offline. Without this, a developer with a real
.env — or any code path that calls load_dotenv() at import time — makes tests reach a
live provider and bill real money, while assertions about extractive fallback text
mysteriously fail. Clear every provider variable by default; tests that want a provider
set it themselves.
"""

from __future__ import annotations

import pytest

_PROVIDER_ENV_VARS = (
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
    "OLLAMA_BASE_URL",
    # Auth and document-write gates: cleared so existing tests exercise the open path and
    # auth tests opt in explicitly, rather than depending on a developer's .env.
    "APP_PASSWORD",
    "APP_PASSWORD_HASH",
    "APP_SECRET_KEY",
    "ALLOW_DOCUMENT_WRITES",
)


@pytest.fixture(autouse=True)
def isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
