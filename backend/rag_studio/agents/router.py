"""Decide how to retrieve for a given question.

Two implementations of one decision. The LLM classifier is primary when a provider is
configured; the rule classifier is the fallback and keeps the pipeline working with zero
API keys (which is also what makes the test suite deterministic and offline).

The rules deliberately key on question *form* — "how many", a digit, "compare" — and not
on vocabulary from the corpus. An earlier version matched literals lifted straight from
the golden set ("binary f1", "macro f1", "score did", "what database"), which scored well
on those 14 sentences and generalised to nothing: rephrasing a question changed its
retrieval strategy. Anything added here must be a property of the question's shape, not a
word that happens to appear in these particular documents.

`rewrite_before_retrieval` is no longer set by routing. It previously fired on a rule
matching "database" AND "memory" — one golden example — to pre-expand that query. The
graph's grade-then-rewrite retry loop already handles weak retrieval generally, so the
pre-emptive special case went with the rest of the overfitting. The field stays on
RouteDecision because the graph reads it, and a future general signal could set it.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from rag_studio.llm import complete, resolve_provider

logger = logging.getLogger(__name__)

DIRECT = "direct_answer"
EXACT_FACT = "exact_fact"
BROAD = "broad_comparison"
BALANCED = "balanced"

CATEGORIES = (DIRECT, EXACT_FACT, BROAD, BALANCED)

# Providers complete() can drive. Ollama has its own code path in the generator and is
# not worth a second one here, so it routes by rules.
_LLM_ROUTABLE = ("gemini", "litellm", "openai")


@dataclass(frozen=True)
class RouteDecision:
    route: str
    retriever: str
    parent_context: bool
    multi_query: bool
    hyde: bool
    reason: str
    rewrite_before_retrieval: bool = False
    category: str = BALANCED
    decided_by: str = "rules"


def _decision(category: str, reason: str, decided_by: str) -> RouteDecision:
    """Map a category to retrieval settings.

    Keeping the mapping in one place means the LLM and rule classifiers cannot drift into
    disagreeing about what a category implies — they only choose the label.
    """
    if category == DIRECT:
        return RouteDecision(
            route="direct",
            retriever="hybrid",
            parent_context=False,
            multi_query=False,
            hyde=False,
            reason=reason,
            category=DIRECT,
            decided_by=decided_by,
        )
    if category == EXACT_FACT:
        return RouteDecision(
            route="retrieve",
            retriever="hybrid",
            parent_context=False,
            multi_query=False,
            hyde=False,
            reason=reason,
            category=EXACT_FACT,
            decided_by=decided_by,
        )
    if category == BROAD:
        return RouteDecision(
            route="retrieve",
            retriever="hybrid",
            parent_context=True,
            multi_query=True,
            hyde=True,
            reason=reason,
            category=BROAD,
            decided_by=decided_by,
        )
    return RouteDecision(
        route="retrieve",
        retriever="hybrid",
        parent_context=True,
        multi_query=True,
        hyde=False,
        reason=reason,
        category=BALANCED,
        decided_by=decided_by,
    )


class QueryRouter:
    """Routes a question, preferring an LLM classifier when one is available.

    mode: "auto" uses the LLM when a supported provider is configured, otherwise rules.
    "rules" and "llm" force one path. ROUTER_MODE sets the default.
    """

    def __init__(self, mode: str | None = None) -> None:
        resolved = (mode or os.getenv("ROUTER_MODE") or "auto").strip().lower()
        if resolved not in {"auto", "rules", "llm"}:
            raise ValueError(f"mode must be auto, rules or llm, got {resolved!r}")
        self.mode = resolved

    def route(self, question: str) -> RouteDecision:
        if self._should_use_llm():
            try:
                return self._route_with_llm(question)
            except (RuntimeError, ValueError) as exc:
                # Routing is not worth failing a query over: the rules give a usable
                # answer. Generation failures still raise, because a wrong answer there
                # is worse than no answer. Logged rather than swallowed, because a silent
                # fallback once hid a misconfiguration that disabled LLM routing entirely
                # — `decided_by` on the decision is the other half of that signal.
                logger.warning("LLM routing failed, falling back to rules: %s", exc)
        return route_by_rules(question)

    def _should_use_llm(self) -> bool:
        if self.mode == "rules":
            return False
        if self.mode == "llm":
            return True
        try:
            config = resolve_provider()
        except RuntimeError:
            return False
        return config.provider in _LLM_ROUTABLE

    def _route_with_llm(self, question: str) -> RouteDecision:
        # No max_output_tokens on purpose. A label needs only a handful of tokens, but
        # thinking models spend the output budget on internal reasoning first, so a tight
        # cap returns an empty response and every route silently falls back to the rules.
        raw = complete(
            _CLASSIFY_PROMPT.format(question=question),
            system_instruction=(
                "You classify search queries. Reply with the label only, no punctuation "
                "or explanation."
            ),
            temperature=0.0,
        )
        category = _parse_category(raw)
        return _decision(
            category,
            f"An LLM classified this question as {category}.",
            decided_by="llm",
        )


_CLASSIFY_PROMPT = """Classify the question into exactly one label.

The documents are personal career documents: resumes, job descriptions, project write-ups \
and course notes.

{direct_answer} - ONLY a greeting, small talk, or a question about what this assistant is \
and how to use it. Choose this only when answering needs nothing from the documents.
{exact_fact} - asks for one specific value or identifier: a number, score, percentage, \
date, name, or which single tool or technology was used.
{broad_comparison} - compares options, judges fit or suitability for a role, asks how to \
tailor or improve a document, or asks for a summary spanning several documents.
{balanced} - anything else asking about the content of the documents.

Important: a request phrased as an offer of help ("can you help me...", "could you look \
at...") is still about the documents. Classify it by what it asks for, not by its \
politeness. Only pick {direct_answer} when no document content is needed at all.

Examples:
"hello" -> {direct_answer}
"what can you do?" -> {direct_answer}
"How many agents were in the tutoring system?" -> {exact_fact}
"Can you help me tailor my resume for this job description?" -> {broad_comparison}
"What frontend frameworks are listed in my resume?" -> {balanced}

Question: {{question}}

Label:""".format(
    direct_answer=DIRECT,
    exact_fact=EXACT_FACT,
    broad_comparison=BROAD,
    balanced=BALANCED,
)


def _parse_category(raw: str) -> str:
    """Pull a known label out of the model's reply, or raise."""
    cleaned = raw.strip().lower()
    for category in CATEGORIES:
        if category in cleaned:
            return category
    raise ValueError(f"LLM returned an unrecognised route label: {raw!r}")


# --- rule-based fallback -------------------------------------------------------------

# Whole questions that are greetings or meta, anchored so a word appearing mid-question
# cannot trigger them. "Which resume would help me for an AI role?" must never be
# classified as a greeting just because it contains "help".
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|good (morning|afternoon|evening))\b[\s!.,?]*$"
)
_META_RE = re.compile(
    r"\b(what can you do|who are you|what are you|how do i use (you|this)|"
    r"what is this (tool|app|thing)|how does this work)\b"
)

# Form signals for a single-value question. Matched anywhere, not anchored: "By how many
# percentage points did it beat the baseline?" is a quantity question despite the preamble.
_QUANTITY_OPENER_RE = re.compile(
    r"\b(how many|how much|how long|how often|how old|how accurate|how fast)\b"
)
# An interrogative plus a generic quantity noun. Deliberately generic measurement
# vocabulary, not corpus vocabulary — \bscore\b also matches inside "F-score" and
# "F1 score", so paraphrases land here without naming any metric from these documents.
_INTERROGATIVE_RE = re.compile(r"\b(what|which|how)\b")
_QUANTITY_NOUN_RE = re.compile(
    r"\b(score|accuracy|precision|recall|percentage|percent|rate|value|number|"
    r"amount|total|count|version|year|date|duration)\b"
)
_HAS_NUMBER_RE = re.compile(r"(\d|%)")
_WHICH_SINGLE_RE = re.compile(r"\bwhich\s+(one|of these|of them)\b")

# Form signals for a question that needs breadth.
_BROAD_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference between|"
    r"fit|fits|suited|suitable|best for|better for|tailor|"
    r"summar(y|ise|ize)|overview|tell me about|"
    r"job description|jd\b|which resume|best resume|"
    r"strengths|weaknesses|gaps?|missing)\b"
)
_ROLE_RE = re.compile(r"\b(job|role|position|opening|vacancy)\b")


def route_by_rules(question: str) -> RouteDecision:
    """Classify by question form. No corpus vocabulary — see the module docstring."""
    lowered = question.lower()

    if _GREETING_RE.match(lowered) or _META_RE.search(lowered):
        return _decision(
            DIRECT,
            "The question is a greeting or is about the assistant, so no retrieval is needed.",
            decided_by="rules",
        )

    # An explicit quantity opener outranks a breadth signal, so "How many roles has she
    # held?" stays a counting question rather than becoming a job-fit question.
    if _QUANTITY_OPENER_RE.search(lowered):
        return _decision(
            EXACT_FACT,
            "The question asks for a count or quantity, so retrieve precise child chunks.",
            decided_by="rules",
        )

    # Breadth is checked before the weaker value signals: "Compare the F1 scores of the two
    # models" contains a digit, but comparison is the dominant need.
    if _BROAD_RE.search(lowered) or _ROLE_RE.search(lowered):
        return _decision(
            BROAD,
            "The question compares or judges fit, so widen recall across documents.",
            decided_by="rules",
        )

    if (
        (_INTERROGATIVE_RE.search(lowered) and _QUANTITY_NOUN_RE.search(lowered))
        or _WHICH_SINGLE_RE.search(lowered)
        or _HAS_NUMBER_RE.search(lowered)
    ):
        return _decision(
            EXACT_FACT,
            "The question asks for a single value, so retrieve precise child chunks.",
            decided_by="rules",
        )

    return _decision(
        BALANCED,
        "Default career research route with balanced recall and precision.",
        decided_by="rules",
    )
