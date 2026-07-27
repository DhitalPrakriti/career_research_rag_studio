"""Gemini access shared by answer generation and HyDE.

Kept at the package root rather than under generation/ because retrieval/hyde.py
needs it too, and retrieval should not have to import from generation.
"""

from __future__ import annotations

import os

# Overridable with GEMINI_MODEL. Matches the model behind the `gemini-flash` alias
# used in the email-automation-aiagent LiteLLM config, so both projects default to
# the same model. Set GEMINI_MODEL to gemini-3.5-flash-lite for cheaper bulk work
# such as RAGAS judging.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

_API_KEY_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def gemini_api_key() -> str | None:
    """The first Gemini API key present in the environment, if any."""
    for name in _API_KEY_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def gemini_is_configured() -> bool:
    return gemini_api_key() is not None


def generate_with_gemini(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """Send one prompt to Gemini and return the text.

    Raises RuntimeError rather than degrading quietly. A silent fallback is how
    an entire evaluation run can end up scoring extractive text as if it were
    generated output, so failures here are loud on purpose.
    """
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

    config: dict[str, object] = {"temperature": temperature}
    if system_instruction is not None:
        config["system_instruction"] = system_instruction
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens

    client = genai.Client(api_key=api_key)
    model_name = model or gemini_model_name()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config),
        )
    except Exception as exc:  # the SDK raises provider-specific errors
        raise RuntimeError(f"Gemini request failed for model {model_name}: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError(
            f"Gemini returned an empty response for model {model_name}. This usually means "
            "the response was blocked by a safety filter or hit the output token limit."
        )
    return text
