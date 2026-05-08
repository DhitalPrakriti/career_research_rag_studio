from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    route: str
    retriever: str
    parent_context: bool
    multi_query: bool
    hyde: bool
    reason: str


class QueryRouter:
    def route(self, question: str) -> RouteDecision:
        lowered = question.lower()

        if _is_direct_question(lowered):
            return RouteDecision(
                route="direct",
                retriever="hybrid",
                parent_context=False,
                multi_query=False,
                hyde=False,
                reason="The question does not need document retrieval.",
            )

        if _asks_for_exact_fact(lowered):
            return RouteDecision(
                route="retrieve",
                retriever="hybrid",
                parent_context=True,
                multi_query=False,
                hyde=False,
                reason="The question asks for a specific fact, so use precise hybrid retrieval.",
            )

        if _asks_for_comparison_or_fit(lowered):
            return RouteDecision(
                route="retrieve",
                retriever="hybrid",
                parent_context=True,
                multi_query=True,
                hyde=True,
                reason="The question needs broader recall across career documents.",
            )

        return RouteDecision(
            route="retrieve",
            retriever="hybrid",
            parent_context=True,
            multi_query=True,
            hyde=False,
            reason="Default career research route with balanced recall and precision.",
        )


def _is_direct_question(question: str) -> bool:
    direct_phrases = {
        "what can you do",
        "how do i use",
    }
    direct_words = {
        "hello",
        "hi",
        "help",
    }
    words = set(re.findall(r"[a-z]+", question))
    return any(phrase in question for phrase in direct_phrases) or bool(
        words.intersection(direct_words)
    )


def _asks_for_exact_fact(question: str) -> bool:
    exact_markers = {
        "how many",
        "what score",
        "what database",
        "which database",
        "what f1",
        "percentage",
        "%",
    }
    return any(marker in question for marker in exact_markers)


def _asks_for_comparison_or_fit(question: str) -> bool:
    broad_markers = {
        "compare",
        "fit",
        "match",
        "job",
        "role",
        "jd",
        "job description",
        "which resume",
        "best resume",
    }
    return any(marker in question for marker in broad_markers)
