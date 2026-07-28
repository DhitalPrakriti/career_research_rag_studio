"""LLM provider selection shared by generation, HyDE and evaluation.

Lives at the package root rather than under generation/ because retrieval/hyde.py
needs it too, and retrieval should not have to import from generation.

Set LLM_PROVIDER to choose explicitly. Leaving it unset auto-detects from whichever
credentials are present, which is convenient locally but ambiguous in a deployment —
so production configs should always name the provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_LITELLM_MODEL = "gemini-flash"
DEFAULT_LITELLM_BASE_URL = "http://localhost:4000"

GEMINI = "gemini"
LITELLM = "litellm"
OPENAI = "openai"
OLLAMA = "ollama"
EXTRACTIVE = "extractive"

VALID_PROVIDERS = (GEMINI, LITELLM, OPENAI, OLLAMA, EXTRACTIVE)


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str | None
    base_url: str | None = None

    @property
    def is_llm(self) -> bool:
        return self.provider != EXTRACTIVE

    def describe(self) -> str:
        if self.provider == EXTRACTIVE:
            return "extractive (no LLM configured)"
        location = f" via {self.base_url}" if self.base_url else ""
        return f"{self.provider} / {self.model}{location}"


def gemini_api_key() -> str | None:
    """The first Gemini API key present in the environment, if any."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def litellm_api_key() -> str | None:
    for name in ("LITELLM_API_KEY", "LITELLM_MASTER_KEY"):
        value = os.getenv(name)
        if value:
            return value
    return None


def litellm_base_url() -> str:
    return (os.getenv("LITELLM_BASE_URL") or DEFAULT_LITELLM_BASE_URL).rstrip("/")


def litellm_model_name() -> str:
    return os.getenv("LITELLM_MODEL") or DEFAULT_LITELLM_MODEL


def gemini_is_configured() -> bool:
    return gemini_api_key() is not None


def litellm_is_configured() -> bool:
    return litellm_api_key() is not None


def resolve_provider() -> ProviderConfig:
    """Decide which provider to use, honouring LLM_PROVIDER when set."""
    requested = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if requested:
        if requested not in VALID_PROVIDERS:
            raise RuntimeError(
                f"LLM_PROVIDER={requested!r} is not recognised. "
                f"Choose one of: {', '.join(VALID_PROVIDERS)}."
            )
        return _config_for(requested, explicit=True)

    for candidate in (LITELLM, GEMINI, OPENAI, OLLAMA):
        config = _config_for(candidate, explicit=False)
        if config is not None:
            return config
    return ProviderConfig(EXTRACTIVE, None)


def _config_for(provider: str, explicit: bool) -> ProviderConfig | None:
    """Build a config for one provider, or None when its credentials are absent.

    With explicit=True a missing credential is an error rather than a skip: if a
    deployment names a provider, silently using a different one is worse than failing.
    """
    if provider == EXTRACTIVE:
        return ProviderConfig(EXTRACTIVE, None)

    if provider == LITELLM:
        if litellm_is_configured():
            return ProviderConfig(LITELLM, litellm_model_name(), litellm_base_url())
        if explicit:
            raise RuntimeError(
                "LLM_PROVIDER=litellm needs LITELLM_API_KEY (or LITELLM_MASTER_KEY). "
                f"Also set LITELLM_BASE_URL if the proxy is not at {DEFAULT_LITELLM_BASE_URL}."
            )
        return None

    if provider == GEMINI:
        if gemini_is_configured():
            return ProviderConfig(GEMINI, gemini_model_name())
        if explicit:
            raise RuntimeError(
                "LLM_PROVIDER=gemini needs GEMINI_API_KEY (or GOOGLE_API_KEY)."
            )
        return None

    if provider == OPENAI:
        model = os.getenv("OPENAI_MODEL")
        if os.getenv("OPENAI_API_KEY") and model:
            return ProviderConfig(OPENAI, model, os.getenv("OPENAI_BASE_URL"))
        if explicit:
            raise RuntimeError("LLM_PROVIDER=openai needs OPENAI_API_KEY and OPENAI_MODEL.")
        return None

    if provider == OLLAMA:
        model = os.getenv("OLLAMA_MODEL")
        if model:
            return ProviderConfig(OLLAMA, model, os.getenv("OLLAMA_BASE_URL"))
        if explicit:
            raise RuntimeError("LLM_PROVIDER=ollama needs OLLAMA_MODEL.")
        return None

    return None


def complete(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    config: ProviderConfig | None = None,
) -> str:
    """Send one prompt to the configured provider and return the text.

    Raises RuntimeError rather than degrading quietly. A silent fallback is how an
    entire evaluation run can end up scoring extractive text as if it were generated
    output, so failures here are loud on purpose.
    """
    config = config or resolve_provider()

    if config.provider == GEMINI:
        return _complete_with_gemini(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            model=config.model or DEFAULT_GEMINI_MODEL,
        )
    if config.provider in (LITELLM, OPENAI):
        return _complete_with_openai_compatible(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            config=config,
        )
    raise RuntimeError(
        f"complete() does not handle provider {config.provider!r}. "
        "Ollama and extractive answers are handled by their own code paths."
    )


def generate_with_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """Send one prompt to Gemini directly, bypassing provider selection."""
    if not gemini_is_configured():
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
            "environment or .env file."
        )
    return _complete_with_gemini(
        prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        model=model or gemini_model_name(),
    )


def _complete_with_gemini(
    prompt: str,
    system_instruction: str | None,
    temperature: float,
    max_output_tokens: int | None,
    model: str,
) -> str:
    api_key = gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your "
            "environment or .env file."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Install the Gemini SDK to use Gemini generation: pip install google-genai"
        ) from exc

    settings: dict[str, object] = {"temperature": temperature}
    if system_instruction is not None:
        settings["system_instruction"] = system_instruction
    if max_output_tokens is not None:
        settings["max_output_tokens"] = max_output_tokens

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**settings),
        )
    except Exception as exc:  # the SDK raises provider-specific errors
        raise RuntimeError(f"Gemini request failed for model {model}: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError(
            f"Gemini returned an empty response for model {model}. This usually means the "
            "response was blocked by a safety filter or hit the output token limit."
        )
    return text


def _complete_with_openai_compatible(
    prompt: str,
    system_instruction: str | None,
    temperature: float,
    max_output_tokens: int | None,
    config: ProviderConfig,
) -> str:
    """Chat completion against any OpenAI-compatible endpoint, including LiteLLM."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the OpenAI SDK for LiteLLM and OpenAI-compatible endpoints: "
            "pip install openai"
        ) from exc

    if config.provider == LITELLM:
        api_key = litellm_api_key()
        base_url = config.base_url or litellm_base_url()
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = config.base_url

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    model = config.model or DEFAULT_LITELLM_MODEL
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
    except Exception as exc:
        where = f" at {base_url}" if base_url else ""
        raise RuntimeError(
            f"{config.provider} request failed for model {model}{where}: {exc}"
        ) from exc

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"{config.provider} returned an empty response for model {model}.")
    return text
