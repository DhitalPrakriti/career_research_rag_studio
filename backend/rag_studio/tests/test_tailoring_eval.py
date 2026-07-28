"""Tests for the tailoring evaluation itself.

An evaluation that silently reports 0.000 fabrication because its detector never fires is
worse than no evaluation, since it manufactures false confidence. So the detector is tested
in both directions: it must catch invented claims and it must not flag grounded ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_studio.evaluation.tailoring_eval import (
    TailoringExample,
    evaluate_example,
    extract_claims,
    load_tailoring_set,
    run_tailoring_evaluation,
    summarize_tailoring,
    unsupported_claims,
)
from rag_studio.schema import Chunk, RetrievedChunk
from rag_studio.tailoring.matching import MATCHED, MISSING, Requirement, RequirementMatch
from rag_studio.tailoring.service import TailoredBullet, TailoringResult

EVIDENCE = (
    "Designed and implemented 8 deep learning models including CNN-BiLSTM and Transformer "
    "architectures, achieving 94.28% Binary F1 and 85.98% Macro F1 on 18-class gesture data. "
    "Built a local RAG pipeline with FAISS and SentenceTransformers embeddings."
)


class TestClaimExtraction:
    def test_pulls_out_numbers(self) -> None:
        claims = extract_claims("Achieved 94.28% Binary F1 across 8 models.")

        assert "94.28%" in claims
        assert "8" in claims

    def test_pulls_out_technology_tokens(self) -> None:
        claims = extract_claims("Built with PyTorch, FAISS and CNN-BiLSTM.")

        assert "pytorch" in claims
        assert "faiss" in claims
        assert "cnn-bilstm" in claims

    def test_ignores_sentence_initial_verbs(self) -> None:
        """Otherwise every bullet trips the detector and the signal is lost."""
        claims = extract_claims("Designed a system. Implemented a pipeline. Built a service.")

        assert claims == set()

    def test_ignores_ordinary_capitalised_words(self) -> None:
        assert extract_claims("Worked with the Team on the Project") == set()


class TestFabricationDetection:
    def test_a_grounded_bullet_has_no_unsupported_claims(self) -> None:
        bullet = (
            "Designed 8 deep learning models in PyTorch, achieving 94.28% Binary F1 on "
            "18-class gesture recognition."
        )

        # PyTorch is not in EVIDENCE, so this asserts only the claims that are.
        assert unsupported_claims(bullet, EVIDENCE + " PyTorch") == set()

    def test_catches_an_invented_technology(self) -> None:
        bullet = "Built infrastructure automation with Kubernetes and Terraform."

        unsupported = unsupported_claims(bullet, EVIDENCE)

        assert "kubernetes" in unsupported
        assert "terraform" in unsupported

    def test_catches_an_inflated_number(self) -> None:
        """The exact figures are the most damaging thing to get wrong on a resume."""
        bullet = "Achieved 99.9% Binary F1 across 40 models."

        unsupported = unsupported_claims(bullet, EVIDENCE)

        assert "99.9%" in unsupported
        assert "40" in unsupported

    def test_a_correct_number_is_not_flagged(self) -> None:
        assert "94.28%" not in unsupported_claims("Reached 94.28% Binary F1.", EVIDENCE)

    def test_substring_containment_handles_longer_evidence_phrases(self) -> None:
        assert unsupported_claims("Used CNN-BiLSTM models.", EVIDENCE) == set()


def _match(requirement_id: int, text: str, status: str, evidence: str | None) -> RequirementMatch:
    chunks = (
        [RetrievedChunk(chunk=Chunk(id="c", text=evidence, metadata={"title": "r.pdf"}), score=0.9)]
        if evidence
        else []
    )
    return RequirementMatch(Requirement(requirement_id, text), status, 0.8, chunks)


def _result(matches, bullets) -> TailoringResult:
    return TailoringResult(
        job_description="jd",
        matches=matches,
        bullets=bullets,
        recommended_resume="r.pdf",
        coverage=0.5,
        citations=[],
        contexts=[],
        trace=[],
        is_generated=True,
    )


class TestScoringAnExample:
    def test_correct_classifications_score_full_marks(self) -> None:
        example = TailoringExample("x", "jd", present=["PyTorch"], absent=["Kubernetes"])
        result = _result(
            [
                _match(1, "Experience with PyTorch", MATCHED, EVIDENCE),
                _match(2, "Kubernetes orchestration", MISSING, None),
            ],
            [],
        )

        record = evaluate_example(example, result)

        assert all(row["correct"] for row in record["labels"])

    def test_a_false_match_on_an_absent_skill_is_counted_wrong(self) -> None:
        """The Kubernetes bug, as an automated check rather than a manual reading."""
        example = TailoringExample("x", "jd", absent=["Kubernetes"])
        result = _result([_match(1, "Kubernetes orchestration", MATCHED, EVIDENCE)], [])

        record = evaluate_example(example, result)

        assert record["labels"][0]["correct"] is False
        assert record["labels"][0]["actual"] == ["present"]

    def test_a_false_gap_on_a_present_skill_is_counted_wrong(self) -> None:
        """The "Proficiency in Python" bug."""
        example = TailoringExample("x", "jd", present=["Python"])
        result = _result([_match(1, "Proficiency in Python", MISSING, None)], [])

        assert evaluate_example(example, result)["labels"][0]["correct"] is False

    def test_a_requirement_never_extracted_is_marked_unextracted(self) -> None:
        example = TailoringExample("x", "jd", present=["FAISS"])
        result = _result([_match(1, "Something unrelated", MATCHED, EVIDENCE)], [])

        row = evaluate_example(example, result)["labels"][0]

        assert row["extracted"] is False
        assert row["correct"] is False

    def test_bullets_are_checked_against_their_own_requirement_evidence(self) -> None:
        example = TailoringExample("x", "jd")
        result = _result(
            [_match(1, "Deep learning", MATCHED, EVIDENCE)],
            [TailoredBullet(1, "Trained models with Kubernetes.", [1])],
        )

        record = evaluate_example(example, result)

        assert record["bullets"][0]["fabricated"] is True
        assert "kubernetes" in record["bullets"][0]["unsupported_claims"]


class TestSummary:
    def test_splits_the_two_error_directions(self) -> None:
        records = [
            {
                "labels": [
                    {"expected": "present", "correct": True, "extracted": True},
                    {"expected": "present", "correct": False, "extracted": True},
                    {"expected": "absent", "correct": True, "extracted": True},
                    {"expected": "absent", "correct": True, "extracted": True},
                ],
                "bullets": [
                    {"fabricated": False, "misattributed": False},
                    {"fabricated": True, "misattributed": False},
                ],
            }
        ]

        summary = summarize_tailoring(records)

        assert summary["present_accuracy"] == 0.5
        assert summary["absent_accuracy"] == 1.0
        assert summary["classification_accuracy"] == 0.75
        assert summary["fabrication_rate"] == 0.5
        assert summary["clean_bullets"] == 1.0

    def test_an_empty_run_is_all_zero(self) -> None:
        assert summarize_tailoring([])["fabrication_rate"] == 0.0


class TestGoldenSetFile:
    """The shipped set is data, and wrong labels would quietly invalidate every number."""

    @pytest.fixture
    def examples(self) -> list[TailoringExample]:
        return load_tailoring_set(Path("evaluation/tailoring_set.jsonl"))

    def test_loads(self, examples: list[TailoringExample]) -> None:
        assert len(examples) >= 5

    def test_every_label_appears_in_its_posting(
        self, examples: list[TailoringExample]
    ) -> None:
        """A keyword absent from the posting can never be extracted, so the label is
        unscoreable rather than merely failing."""
        for example in examples:
            posting = example.job_description.lower()
            for keyword in example.present + example.absent:
                assert keyword.lower() in posting, f"{example.id}: {keyword} not in posting"

    def test_labels_do_not_overlap(self, examples: list[TailoringExample]) -> None:
        for example in examples:
            overlap = {k.lower() for k in example.present} & {k.lower() for k in example.absent}
            assert not overlap, f"{example.id}: {overlap} labelled both ways"

    def test_the_set_covers_both_directions(self, examples: list[TailoringExample]) -> None:
        assert sum(len(e.present) for e in examples) >= 10
        assert sum(len(e.absent) for e in examples) >= 10


def test_run_evaluation_walks_every_example() -> None:
    class FakeTailor:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def tailor(self, job_description: str, max_requirements: int = 25):
            self.seen.append(job_description)
            return _result([_match(1, "PyTorch", MATCHED, EVIDENCE)], [])

    tailor = FakeTailor()
    examples = [
        TailoringExample("a", "first posting", present=["PyTorch"]),
        TailoringExample("b", "second posting", present=["PyTorch"]),
    ]

    records = run_tailoring_evaluation(tailor, examples)

    assert [record["id"] for record in records] == ["a", "b"]
    assert tailor.seen == ["first posting", "second posting"]


class TestInventedVersusMisattributed:
    """The rewriter sees every requirement's evidence in one prompt, so a bullet can cite a
    real fact that belongs to a different requirement. That is untidy attribution, not a
    fabricated claim, and conflating them overstates the fabrication rate."""

    def test_a_claim_from_another_requirement_is_misattributed_not_fabricated(self) -> None:
        example = TailoringExample("x", "jd")
        result = _result(
            [
                _match(1, "Deep learning", MATCHED, "Trained models with PyTorch."),
                _match(2, "Sequence models", MATCHED, "Used CNN-BiLSTM architectures."),
            ],
            [TailoredBullet(1, "Trained CNN-BiLSTM models.", [1])],
        )

        row = evaluate_example(example, result)["bullets"][0]

        assert row["misattributed"] is True
        assert row["fabricated"] is False
        assert row["invented_claims"] == []

    def test_a_claim_in_no_evidence_at_all_is_fabricated(self) -> None:
        example = TailoringExample("x", "jd")
        result = _result(
            [_match(1, "Deep learning", MATCHED, "Trained models with PyTorch.")],
            [TailoredBullet(1, "Trained models on Kubernetes.", [1])],
        )

        row = evaluate_example(example, result)["bullets"][0]

        assert row["fabricated"] is True
        assert "kubernetes" in row["invented_claims"]
