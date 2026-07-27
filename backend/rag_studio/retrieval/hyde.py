from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class HydeGenerator:
    def generate(self, question: str) -> str:
        ollama_model = os.getenv("OLLAMA_MODEL")
        if ollama_model:
            try:
                return _generate_with_ollama(question, ollama_model)
            except RuntimeError:
                pass
        return _generate_deterministic_hypothesis(question)


def _generate_with_ollama(question: str, model: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    prompt = (
        "Write a concise hypothetical answer that would help retrieve relevant career "
        "documents for the question below. Include likely skills, tools, roles, and "
        "project terms. Do not cite sources. Do not say you lack context.\n\n"
        f"Question: {question}"
    )
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
