from __future__ import annotations

import re
from dataclasses import dataclass

from rag_studio.schema import RetrievedChunk


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "does",
    "for",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "which",
}


@dataclass(frozen=True)
class RetrievalGrade:
    is_relevant: bool
    score: float
    reason: str


class RetrievalGrader:
    def __init__(self, min_overlap: float = 0.2) -> None:
        if not 0 <= min_overlap <= 1:
            raise ValueError("min_overlap must be between 0 and 1")
        self.min_overlap = min_overlap

    def grade(self, question: str, contexts: list[RetrievedChunk]) -> RetrievalGrade:
        if not contexts:
            return RetrievalGrade(False, 0.0, "No chunks were retrieved.")

        question_tokens = _content_tokens(question)
        if not question_tokens:
            return RetrievalGrade(True, 1.0, "The question has no content tokens to grade.")

        context_tokens = _content_tokens(" ".join(result.chunk.text for result in contexts))
        overlap = question_tokens.intersection(context_tokens)
        score = len(overlap) / len(question_tokens)
        is_relevant = score >= self.min_overlap
        reason = (
            f"Token overlap {score:.2f} using terms: {', '.join(sorted(overlap)) or 'none'}."
        )
        return RetrievalGrade(is_relevant, score, reason)

    def rank(self, question: str, contexts: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [
            context
            for _, _, context in sorted(
                (
                    (
                        self.grade(question, [context]).score,
                        -index,
                        context,
                    )
                    for index, context in enumerate(contexts)
                ),
                reverse=True,
            )
        ]


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {_singularize(token) for token in tokens if token not in _STOP_WORDS}


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
