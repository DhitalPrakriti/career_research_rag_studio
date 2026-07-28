"""Router tests.

The previous suite asserted the overfitted behaviour directly — one test required that
"What database was used for memory?" set rewrite_before_retrieval, which only existed to
make a single golden example retrieve well. Those assertions are gone with the rules they
described; what remains checks that paraphrases of a question route the same way.
"""

from pathlib import Path

import pytest

from rag_studio.agents import router as router_module
from rag_studio.agents.router import (
    BALANCED,
    BROAD,
    DIRECT,
    EXACT_FACT,
    QueryRouter,
    route_by_rules,
)


class TestDirectRoute:
    @pytest.mark.parametrize("question", ["hi", "Hello!", "hey", "thanks", "Good morning"])
    def test_greetings_skip_retrieval(self, question: str) -> None:
        assert route_by_rules(question).category == DIRECT

    @pytest.mark.parametrize(
        "question",
        ["What can you do?", "Who are you?", "How do I use this?"],
    )
    def test_meta_questions_skip_retrieval(self, question: str) -> None:
        assert route_by_rules(question).category == DIRECT

    @pytest.mark.parametrize(
        "question",
        [
            "Which resume would help me for an AI engineering role?",
            "What skills help with backend work?",
            "Can you help me tailor my resume?",
            "Which technologies help with high throughput?",
        ],
    )
    def test_the_word_help_no_longer_skips_retrieval(self, question: str) -> None:
        """Regression: "help" matched anywhere in a question routed it to no retrieval.

        These are core use cases and they were answered with boilerplate instead.
        """
        decision = route_by_rules(question)

        assert decision.route == "retrieve"
        assert decision.category != DIRECT

    def test_a_greeting_pattern_inside_a_real_question_does_not_match(self) -> None:
        assert route_by_rules("Which history of hers is listed?").category != DIRECT


class TestExactFactRoute:
    @pytest.mark.parametrize(
        "question",
        [
            "How many specialized agents were in the AI tutoring system?",
            "How much did throughput improve?",
            "What score did the transformer reach?",
            "What percentage of tests pass?",
            "What year did she graduate?",
            "Which one of these is faster?",
        ],
    )
    def test_single_value_questions_use_precise_retrieval(self, question: str) -> None:
        decision = route_by_rules(question)

        assert decision.category == EXACT_FACT
        # Child chunks and no query expansion: precision over recall.
        assert decision.parent_context is False
        assert decision.multi_query is False
        assert decision.hyde is False


class TestBroadRoute:
    @pytest.mark.parametrize(
        "question",
        [
            "Compare these two job descriptions.",
            "Which resume best fits an AI engineering role?",
            "How should I tailor my resume for this position?",
            "What skills am I missing for this role?",
            "Give me an overview of my projects.",
            "Backend developer versus data engineer, which suits me?",
        ],
    )
    def test_comparison_and_fit_questions_widen_recall(self, question: str) -> None:
        decision = route_by_rules(question)

        assert decision.category == BROAD
        assert decision.parent_context is True
        assert decision.multi_query is True
        assert decision.hyde is True


class TestGeneralisation:
    @pytest.mark.parametrize(
        "question",
        [
            "What binary F1 score was achieved in the capstone project?",
            "What macro F1 score did the Transformer model achieve?",
            "What F-score did the transformer reach?",
            "What accuracy did the model reach?",
            "By how many percentage points did it beat the baseline?",
        ],
    )
    def test_metric_questions_agree_regardless_of_wording(self, question: str) -> None:
        assert route_by_rules(question).category == EXACT_FACT

    def test_no_corpus_specific_literals_remain_in_the_router(self) -> None:
        """Guards against tuning the router on the golden set again.

        Each of these was lifted verbatim from a test question. Prose may cite them as the
        cautionary example — the docstring and comments do — so this checks matching logic
        only: the module body with docstrings and comment lines stripped.
        """
        source = Path(router_module.__file__).read_text(encoding="utf-8")
        after_docstring = source.split('"""', 2)[-1]
        code = "\n".join(
            line
            for line in after_docstring.splitlines()
            if not line.lstrip().startswith("#")
        ).lower()

        for literal in ("binary f1", "macro f1", "score did", "what database", "f1 score"):
            assert literal not in code, f"corpus-specific literal is back: {literal}"


class TestLlmRouting:
    def test_llm_label_is_used_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(router_module, "complete", lambda *a, **k: "broad_comparison")

        decision = QueryRouter(mode="llm").route("Anything at all")

        assert decision.category == BROAD
        assert decision.decided_by == "llm"
        assert decision.hyde is True

    def test_untidy_llm_output_is_still_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(router_module, "complete", lambda *a, **k: "Label: EXACT_FACT.\n")

        assert QueryRouter(mode="llm").route("How many agents?").category == EXACT_FACT

    def test_falls_back_to_rules_when_the_llm_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A routing failure must not fail the query; the rules are a usable answer.

        Generation still raises on failure — a wrong answer is worse than no answer — but
        a wrong *route* only costs some retrieval quality.
        """
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("Gemini request failed: 429")

        monkeypatch.setattr(router_module, "complete", boom)

        decision = QueryRouter(mode="llm").route("How many agents were there?")

        assert decision.decided_by == "rules"
        assert decision.category == EXACT_FACT

    def test_falls_back_to_rules_on_an_unrecognised_label(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr(router_module, "complete", lambda *a, **k: "probably retrieval?")

        assert QueryRouter(mode="llm").route("Compare my resumes").decided_by == "rules"

    def test_rules_mode_never_calls_the_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        def fail(*args: object, **kwargs: object) -> str:
            raise AssertionError("rules mode must not call the LLM")

        monkeypatch.setattr(router_module, "complete", fail)

        assert QueryRouter(mode="rules").route("How many agents?").decided_by == "rules"

    def test_auto_mode_uses_rules_without_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no credentials there is nothing to call, so routing stays deterministic."""

        def fail(*args: object, **kwargs: object) -> str:
            raise AssertionError("auto mode must not call the LLM without a provider")

        monkeypatch.setattr(router_module, "complete", fail)

        assert QueryRouter(mode="auto").route("How many agents?").decided_by == "rules"

    def test_router_mode_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ROUTER_MODE", "rules")

        assert QueryRouter().mode == "rules"

    def test_invalid_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="auto, rules or llm"):
            QueryRouter(mode="magic")


def test_balanced_is_the_default_for_ordinary_content_questions() -> None:
    decision = route_by_rules("What volunteer experience is listed in the resumes?")

    assert decision.category == BALANCED
    assert decision.route == "retrieve"
