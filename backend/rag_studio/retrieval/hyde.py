from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rag_studio.llm import OLLAMA, complete, resolve_provider


HYDE_PROMPT = (
    "Write a concise hypothetical answer that would help retrieve relevant career "
    "documents for the question below. Include likely skills, tools, roles, and "
    "project terms. Do not cite sources. Do not say you lack context.\n\n"
    "Question: {question}"
)


class HydeGenerator:
    def generate(self, question: str) -> str:
        """Draft a hypothetical answer to retrieve against.

        Unlike answer generation, a failure here degrades to a deterministic
        template rather than raising: HyDE only shapes the retrieval query, so a
        weaker hypothesis costs some recall but still returns usable results.
        """
        config = resolve_provider()
        if not config.is_llm:
            return _generate_deterministic_hypothesis(question)

        try:
            if config.provider == OLLAMA:
                return _generate_with_ollama(question, config.model or "llama3")
            return complete(
                HYDE_PROMPT.format(question=question),
                temperature=0.3,
                config=config,
            )
        except RuntimeError:
            return _generate_deterministic_hypothesis(question)


def _generate_with_ollama(question: str, model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    prompt = HYDE_PROMPT.format(question=question)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
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
        raise RuntimeError("could not reach Ollama for HyDE generation") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON for HyDE generation") from exc

    hypothesis = str(data.get("response", "")).strip()
    if not hypothesis:
        raise RuntimeError("Ollama returned an empty HyDE hypothesis")
    return hypothesis


def _generate_deterministic_hypothesis(question: str) -> str:
    return (
        f"Relevant career document content for this question may discuss: {question}. "
        "Look for skills, tools, projects, responsibilities, metrics, experience, "
        "education, and role-specific evidence."
    )
