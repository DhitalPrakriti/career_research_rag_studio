"""The shipped golden set is data, and a wrong label silently invalidates every number.

Two failure modes are worth a permanent test rather than a one-off check before shipping:

1. **A term absent from the document it is expected in** is unscoreable, not merely hard.
   `term_recall` can never reach 1.0 for it, so the headline figure is capped by a typo.
2. **A term that only matches inside a longer word** looks fine until it silently scores
   for the wrong document. `IMU` is inside "s*imu*lating", so an example expecting the
   AI/ML resume scored a hit off the IT support resume's "simulating real-world IT
   monitoring" — the same trap that ruled `rag` (inside "storage") and `git` (inside
   "digit") out of the tailoring set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rag_studio.evaluation.golden_set import (
    GoldenExample,
    contains_expected_term,
    load_golden_set,
)
from rag_studio.ingestion.loader import load_document


GOLDEN_PATH = Path("evaluation/golden_set.jsonl")

# A term with no whitespace is a single lexical unit, so a match inside a longer word is a
# false positive. Multi-word terms are checked by presence only: the token fallback in
# contains_expected_term is deliberately order-insensitive for those.
_SINGLE_TOKEN = re.compile(r"^\S+$")


@pytest.fixture(scope="module")
def examples() -> list[GoldenExample]:
    return load_golden_set(GOLDEN_PATH)


@pytest.fixture(scope="module")
def document_text() -> dict[str, str]:
    """Full extracted text of every document the set refers to, keyed by title."""
    texts: dict[str, str] = {}
    for path in sorted(Path("docs").glob("*.pdf")):
        texts[path.name] = "\n".join(page.text for page in load_document(path))
    return texts


def _matches_whole_word(text: str, term: str) -> bool:
    """Whole-word match, allowing the plural the metric itself allows.

    `term_recall` singularises tokens, so "baselines" matching "baseline" is intended
    behaviour rather than the substring accident this check hunts for.
    """
    stem = term[:-1] if len(term) > 3 and term.endswith("s") else term
    pattern = rf"(?<!\w){re.escape(stem)}s?(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


class TestTheCheckItself:
    """A guard that cannot fail is not a guard, so pin its behaviour on the real strings."""

    def test_it_rejects_a_substring_only_match(self) -> None:
        assert _matches_whole_word("simulating real-world IT monitoring", "IMU") is False

    def test_it_accepts_a_standalone_occurrence(self) -> None:
        assert _matches_whole_word("sensor data (IMU, Thermopile)", "IMU") is True

    def test_it_accepts_the_plural_the_metric_accepts(self) -> None:
        assert _matches_whole_word("outperforming all paper baselines", "baseline") is True


class TestSetComposition:
    def test_loads(self, examples: list[GoldenExample]) -> None:
        assert len(examples) >= 30

    def test_ids_are_unique(self, examples: list[GoldenExample]) -> None:
        ids = [example.id for example in examples]
        assert len(set(ids)) == len(ids)

    def test_it_has_negative_controls_in_both_shapes(
        self, examples: list[GoldenExample]
    ) -> None:
        negatives = [example for example in examples if example.is_negative_control]
        assert len(negatives) >= 5

    def test_most_examples_search_the_whole_corpus(
        self, examples: list[GoldenExample]
    ) -> None:
        """A single-document example makes doc_title_hit a free 1.0 — there is nothing to
        discriminate against. The metric only means something over the full corpus."""
        multi_doc = [example for example in examples if len(example.docs) > 1]
        assert len(multi_doc) >= len(examples) * 0.6


class TestLabels:
    def test_negative_controls_carry_no_expectations(
        self, examples: list[GoldenExample]
    ) -> None:
        for example in examples:
            if example.is_negative_control:
                assert not example.expected_terms, example.id
                assert not example.expected_doc_titles, example.id

    def test_answerable_examples_expect_something(
        self, examples: list[GoldenExample]
    ) -> None:
        for example in examples:
            if not example.is_negative_control:
                assert example.expected_terms, example.id
                assert example.expected_doc_titles, example.id

    def test_expected_titles_are_among_the_ingested_docs(
        self, examples: list[GoldenExample]
    ) -> None:
        for example in examples:
            ingested = {Path(doc).name for doc in example.docs}
            for title in example.expected_doc_titles:
                assert title in ingested, f"{example.id}: {title} is never ingested"

    def test_documents_exist(self, examples: list[GoldenExample]) -> None:
        for example in examples:
            for doc in example.docs:
                assert Path(doc).exists(), f"{example.id}: missing {doc}"

    def test_every_expected_term_appears_in_an_expected_document(
        self, examples: list[GoldenExample], document_text: dict[str, str]
    ) -> None:
        combined = {
            example.id: " ".join(
                document_text[title] for title in example.expected_doc_titles
            )
            for example in examples
            if not example.is_negative_control
        }
        for example in examples:
            if example.is_negative_control:
                continue
            for term in example.expected_terms:
                assert contains_expected_term(combined[example.id], term), (
                    f"{example.id}: {term!r} does not appear in "
                    f"{example.expected_doc_titles}"
                )

    def test_single_token_terms_never_match_only_inside_a_longer_word(
        self, examples: list[GoldenExample], document_text: dict[str, str]
    ) -> None:
        """The IMU/simulating trap: a substring hit in a document the example does not
        expect awards term_recall for retrieving the wrong resume.

        Scoped to the documents the example actually ingests — a resume that is never in
        the corpus for this question cannot be retrieved from, so it cannot score.
        """
        for example in examples:
            if example.is_negative_control:
                continue
            ingested = {Path(doc).name for doc in example.docs}
            unexpected = {
                title: text
                for title, text in document_text.items()
                if title in ingested and title not in example.expected_doc_titles
            }
            for term in example.expected_terms:
                if not _SINGLE_TOKEN.match(term):
                    continue
                for title, text in unexpected.items():
                    substring_hit = term.lower() in text.lower()
                    if substring_hit and not _matches_whole_word(text, term):
                        pytest.fail(
                            f"{example.id}: {term!r} matches {title} only inside a "
                            "longer word, so retrieving the wrong document scores a hit"
                        )
