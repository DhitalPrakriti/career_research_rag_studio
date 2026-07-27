from __future__ import annotations


class QueryRewriter:
    def rewrite(self, question: str) -> str:
        lowered = question.lower()
        expansions: list[str] = []

        if any(term in lowered for term in ["ai", "ml", "machine learning"]):
            expansions.extend(
                [
                    "artificial intelligence",
                    "machine learning",
                    "deep learning",
                    "RAG",
                    "LLM",
                    "multi-agent",
                ]
            )
        if any(term in lowered for term in ["job", "role", "fit", "match"]):
            expansions.extend(
                [
                    "skills",
                    "projects",
                    "experience",
                    "technical stack",
                    "requirements",
                ]
            )
        if any(term in lowered for term in ["database", "memory"]):
            expansions.extend(["Firestore", "conversation memory", "multi-turn memory"])

        if not expansions:
            expansions.extend(["resume", "skills", "projects", "experience"])

        unique_expansions = list(dict.fromkeys(expansions))
        return f"{question} Related terms: {', '.join(unique_expansions)}."
