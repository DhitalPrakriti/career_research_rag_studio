from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rag_studio.schema import Citation, RetrievedChunk


class AnswerGenerator:
    def __init__(self, max_context_chars: int = 6000) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be positive")
        self.max_context_chars = max_context_chars

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        contexts = trim_contexts(contexts, self.max_context_chars)
        api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL")
        if api_key and openai_model:
            return _generate_with_openai(question, contexts, openai_model)

        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_model:
            try:
                return _generate_with_ollama(question, contexts, ollama_model)
            except RuntimeError as exc:
                fallback = _generate_extractive_answer(question, contexts)
                return f"Ollama generation failed: {exc}\n\n{fallback}"

        return _generate_extractive_answer(question, contexts)


def build_citations(contexts: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for source_id, result in enumerate(contexts, start=1):
        metadata = result.chunk.metadata
        location_parts = []
        if metadata.get("page"):
            location_parts.append(f"page {metadata['page']}")
        if metadata.get("chunk_index") is not None:
            location_parts.append(f"chunk {metadata['chunk_index']}")
        citations.append(
            Citation(
                source_id=source_id,
                title=str(metadata.get("title") or metadata.get("source_path") or "Untitled"),
                location=", ".join(location_parts) or "chunk",
                score=result.score,
            )
        )
    return citations


def _generate_with_openai(question: str, contexts: list[RetrievedChunk], model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai to use LLM generation: pip install openai") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a grounded career research assistant. Answer directly from "
                    "the provided sources and cite claims inline with bracketed source "
                    "numbers like [1]."
                ),
            },
            {
                "role": "user",
                "content": _build_grounded_prompt(question, contexts),
            },
        ],
    )
    return response.output_text.strip()


def _generate_with_ollama(question: str, contexts: list[RetrievedChunk], model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    payload = {
        "model": model,
        "prompt": _build_grounded_prompt(question, contexts),
        "stream": False,
        "options": _ollama_options(),
    }
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "could not reach Ollama. Start it with `ollama serve` or unset OLLAMA_MODEL."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned a response that was not valid JSON.") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")
    return answer


def _ollama_options() -> dict[str, int]:
    options: dict[str, int] = {}
    env_to_option = {
        "OLLAMA_NUM_GPU": "num_gpu",
        "OLLAMA_NUM_THREAD": "num_thread",
        "OLLAMA_NUM_PREDICT": "num_predict",
    }
    for env_name, option_name in env_to_option.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        try:
            options[option_name] = int(raw_value)
        except ValueError as exc:
            raise RuntimeError(f"{env_name} must be an integer.") from exc
    return options


def _build_grounded_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    context_block = "\n\n".join(
        f"[{index}] {result.chunk.text}" for index, result in enumerate(contexts, start=1)
    )
    return (
        "You are answering a career research question using retrieved source text.\n\n"
        "Rules:\n"
        "- Answer the question directly and concisely.\n"
        "- Use only facts supported by the sources.\n"
        "- Cite each factual bullet or sentence with source numbers like [1].\n"
        "- Do not include contact details, addresses, emails, phone numbers, or URLs unless the question asks for them.\n"
        "- Do not add generic uncertainty notes when the sources contain enough evidence to answer.\n"
        "- If the sources are truly insufficient, say exactly what information is missing.\n\n"
        f"Question: {question}\n\nSources:\n{context_block}"
    )


def trim_contexts(contexts: list[RetrievedChunk], max_context_chars: int) -> list[RetrievedChunk]:
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")

    trimmed: list[RetrievedChunk] = []
    used_chars = 0
    for result in contexts:
        text = result.chunk.text
        separator_chars = 2 if trimmed else 0
        next_total = used_chars + separator_chars + len(text)
        if next_total <= max_context_chars:
            trimmed.append(result)
            used_chars = next_total
            continue

        if not trimmed:
            truncated_text = text[: max_context_chars - 3].rstrip() + "..."
            trimmed.append(
                RetrievedChunk(
                    chunk=type(result.chunk)(
                        id=result.chunk.id,
                        text=truncated_text,
                        metadata=result.chunk.metadata,
                    ),
                    score=result.score,
                )
            )
        break
    return trimmed


def _generate_extractive_answer(question: str, contexts: list[RetrievedChunk]) -> str:
    if not contexts:
        return "I could not find relevant context to answer the question."

    lines = [
        "No LLM model is configured, so this is an extractive answer from the retrieved context.",
        f"Question: {question}",
        "",
    ]
    for index, result in enumerate(contexts, start=1):
        snippet = result.chunk.text.strip()
        if len(snippet) > 700:
            snippet = snippet[:697].rstrip() + "..."
        lines.append(f"[{index}] {snippet}")
    return "\n".join(lines)
