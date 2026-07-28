"""Tailoring tests.

The load-bearing one is test_a_gap_can_never_produce_a_bullet: this feature writes resume
content, so claiming experience the candidate does not have is the failure mode that
actually matters.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_studio.schema import Chunk, RetrievedChunk
from rag_studio.tailoring import matching
from rag_studio.tailoring.matching import (
    MATCHED,
    MISSING,
    PARTIAL,
    Requirement,
    RequirementMatch,
    TokenWeights,
    classify_requirement,
    extract_requirements,
    hide_gap_evidence,
    verify_matches,
)
from rag_studio.tailoring import service as service_module
from rag_studio.tailoring.service import ResumeTailor

JOB_DESCRIPTION = """
About the role
We are hiring an AI Engineer.

Requirements:
- Build RAG pipelines with FAISS and embeddings
- Strong Python and PyTorch experience
- Deploy services to Google Cloud Run with Docker
- Lead a team of twelve site reliability engineers

Benefits:
- Dental insurance
"""


def _chunk(text: str, title: str = "resume.pdf", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=text[:12], text=text, metadata={"title": title, "page": 1}),
        score=score,
    )


class FakePipeline:
    """Returns evidence keyed on requirement wording, so matching is predictable."""

    def __init__(self, evidence: dict[str, list[RetrievedChunk]] | None = None) -> None:
        self.evidence = evidence or {}
        self.queries: list[str] = []
        self.retrieve_kwargs: dict[str, Any] = {}

    def retrieve(self, question: str, **kwargs: Any) -> list[RetrievedChunk]:
        self.queries.append(question)
        self.retrieve_kwargs = kwargs
        for keyword, chunks in self.evidence.items():
            if keyword.lower() in question.lower():
                return chunks
        return []


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline(
        {
            "RAG": [_chunk("Built a local RAG pipeline with FAISS embeddings and chunking.")],
            "Python": [_chunk("Languages: Python, JavaScript. Deep learning with PyTorch.")],
            "Cloud Run": [_chunk("Deployed to Google Cloud Run with Docker containers.")],
        }
    )


class TestRequirementExtraction:
    def test_line_parsing_prefers_bullets_and_drops_boilerplate(self) -> None:
        requirements = extract_requirements(JOB_DESCRIPTION, use_llm=False)
        texts = [requirement.text for requirement in requirements]

        assert any("RAG pipelines" in text for text in texts)
        assert any("PyTorch" in text for text in texts)
        assert not any("About the role" in text for text in texts)
        assert not any("Requirements" == text for text in texts)

    def test_requirements_are_numbered_from_one(self) -> None:
        requirements = extract_requirements(JOB_DESCRIPTION, use_llm=False)

        assert [r.id for r in requirements] == list(range(1, len(requirements) + 1))

    def test_duplicates_are_merged(self) -> None:
        text = "- Strong Python experience\n- Experience with strong Python\n- Docker"
        requirements = extract_requirements(text, use_llm=False)

        assert len(requirements) == 2

    def test_an_unbulleted_paragraph_falls_back_to_sentences(self) -> None:
        text = "You will build RAG pipelines. You must know Python well. Docker is a plus."
        requirements = extract_requirements(text, use_llm=False)

        assert len(requirements) >= 2

    def test_empty_input_yields_nothing(self) -> None:
        assert extract_requirements("   ", use_llm=False) == []

    def test_limit_is_respected(self) -> None:
        text = "\n".join(f"- Requirement number {n} about tooling" for n in range(40))

        assert len(extract_requirements(text, limit=5, use_llm=False)) == 5

    def test_llm_extraction_is_used_when_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            matching,
            "complete",
            lambda *a, **k: "Build RAG pipelines\nDeploy with Docker\n",
        )
        requirements = extract_requirements("anything", use_llm=True)

        assert [r.text for r in requirements] == ["Build RAG pipelines", "Deploy with Docker"]

    def test_llm_failure_falls_back_to_line_parsing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("429")

        monkeypatch.setattr(matching, "complete", boom)
        requirements = extract_requirements(JOB_DESCRIPTION, use_llm=True)

        assert any("PyTorch" in r.text for r in requirements)


class TestClassification:
    def test_strong_overlap_is_matched(self) -> None:
        requirement = Requirement(1, "Build RAG pipelines with FAISS and embeddings")
        evidence = [_chunk("Built a RAG pipeline with FAISS embeddings and chunking.")]

        result = classify_requirement(requirement, evidence)

        assert result.status == MATCHED
        assert result.is_supported is True

    def test_no_overlap_is_missing(self) -> None:
        requirement = Requirement(1, "Lead a team of twelve site reliability engineers")
        evidence = [_chunk("Built a RAG pipeline with FAISS embeddings.")]

        result = classify_requirement(requirement, evidence)

        assert result.status == MISSING
        assert result.is_supported is False

    def test_evidence_survives_classification_so_verification_can_read_it(self) -> None:
        """A MISSING verdict must stay auditable, or a false gap can never be overturned."""
        requirement = Requirement(1, "Lead a team of twelve site reliability engineers")
        evidence = [_chunk("Built a RAG pipeline with FAISS embeddings.")]

        assert classify_requirement(requirement, evidence).evidence == evidence

    def test_hide_gap_evidence_strips_it_once_statuses_are_final(self) -> None:
        """Chunks under a gap read as "this nearly counts", which invites a false claim."""
        match = classify_requirement(
            Requirement(1, "Lead a team of twelve site reliability engineers"),
            [_chunk("Built a RAG pipeline with FAISS embeddings.")],
        )

        hidden = hide_gap_evidence([match])

        assert hidden[0].status == MISSING
        assert hidden[0].evidence == []

    def test_no_evidence_is_missing(self) -> None:
        result = classify_requirement(Requirement(1, "Kubernetes operators"), [])

        assert result.status == MISSING

    def test_partial_overlap_sits_between_the_thresholds(self) -> None:
        requirement = Requirement(1, "Docker and Kubernetes")
        evidence = [_chunk("Containerised services with Docker.")]

        result = classify_requirement(requirement, evidence)

        assert result.status == PARTIAL
        assert result.is_supported is True


class TestWeightedScoring:
    """Regression tests for a live failure: a real posting scored 100% coverage with zero
    gaps, claiming Kubernetes, Terraform and 8 years of team leadership that the resume
    does not contain. Generic words carried the score while the decisive term was absent.
    """

    RESUME_CHUNKS = [
        "Built a local RAG pipeline with FAISS embeddings, chunking and vector search.",
        "Deployed to Google Cloud Run with Docker, health checks and automated build "
        "pipelines and cloud infrastructure.",
        "Languages: Python, JavaScript. Deep learning with PyTorch and Transformers.",
        "Designed a 5-agent tutoring system with intent routing and Firestore memory.",
    ]

    def test_unweighted_overlap_overrates_a_missing_technology(self) -> None:
        """Documents the bug, so the fix cannot be quietly reverted.

        Threshold-independent: the point is that flat overlap scores this far higher than
        weighted overlap does, because "infrastructure" counts as much as "kubernetes".
        """
        requirement = Requirement(1, "Experience using Kubernetes for infrastructure automation")
        evidence = [_chunk(self.RESUME_CHUNKS[1])]

        flat = classify_requirement(requirement, evidence).score
        weighted = classify_requirement(
            requirement, evidence, weights=TokenWeights(self.RESUME_CHUNKS)
        ).score

        assert flat > weighted

    def test_weighting_by_rarity_marks_the_missing_technology_as_a_gap(self) -> None:
        requirement = Requirement(1, "Experience using Kubernetes for infrastructure automation")
        evidence = [_chunk(self.RESUME_CHUNKS[1])]
        weights = TokenWeights(self.RESUME_CHUNKS)

        result = classify_requirement(requirement, evidence, weights=weights)

        assert result.status == MISSING
        assert result.is_supported is False

    def test_weighting_still_matches_a_genuinely_present_skill(self) -> None:
        requirement = Requirement(1, "Strong Python and PyTorch experience")
        evidence = [_chunk(self.RESUME_CHUNKS[2])]
        weights = TokenWeights(self.RESUME_CHUNKS)

        result = classify_requirement(requirement, evidence, weights=weights)

        assert result.status == MATCHED

    def test_absent_terms_get_the_highest_weight(self) -> None:
        weights = TokenWeights(self.RESUME_CHUNKS)

        assert weights.weight("kubernetes") > weights.weight("python")

    def test_weights_are_safe_on_an_empty_corpus(self) -> None:
        assert TokenWeights([]).weight("anything") == 1.0


class TestVerification:
    """Weighted overlap cannot judge seniority: "8 years leading a platform engineering
    team" has no rare term to key on, only a claim. That needs reading.
    """

    def _match(self, text: str) -> RequirementMatch:
        return RequirementMatch(
            Requirement(1, text),
            MATCHED,
            0.6,
            [_chunk("Led technical setups for university events as an event coordinator.")],
        )

    def test_llm_can_downgrade_a_false_match_to_a_gap(self) -> None:
        matches = [self._match("8 years leading a platform engineering team")]

        result = verify_matches(
            matches,
            completer=lambda *a, **k: '{"verdicts": [{"id": 1, "status": "missing", '
            '"why": "Evidence shows student event work, not 8 years leading engineers."}]}',
        )

        assert result[0].status == MISSING
        assert result[0].verified_by == "llm"
        assert "not 8 years" in result[0].note

    def test_llm_verdicts_can_confirm_a_match(self) -> None:
        matches = [self._match("Experience coordinating technical events")]

        result = verify_matches(
            matches,
            completer=lambda *a, **k: '{"verdicts": [{"id": 1, "status": "matched", "why": "ok"}]}',
        )

        assert result[0].status == MATCHED
        assert result[0].evidence

    def test_a_failed_verification_keeps_the_deterministic_status(self) -> None:
        matches = [self._match("Some requirement")]

        def boom(*args: object, **kwargs: object) -> str:
            raise RuntimeError("429")

        result = verify_matches(matches, completer=boom)

        assert result[0].status == MATCHED
        assert result[0].verified_by == "overlap"

    def test_unparseable_verification_keeps_the_deterministic_status(self) -> None:
        matches = [self._match("Some requirement")]

        result = verify_matches(matches, completer=lambda *a, **k: "sure, looks fine")

        assert result[0].status == MATCHED

    def test_verdicts_for_unknown_ids_are_ignored(self) -> None:
        matches = [self._match("Some requirement")]

        result = verify_matches(
            matches,
            completer=lambda *a, **k: '{"verdicts": [{"id": 99, "status": "missing", "why": "x"}]}',
        )

        assert result[0].status == MATCHED

    def test_requirements_without_evidence_skip_verification(self) -> None:
        """Nothing to audit, and no call worth paying for."""

        def fail(*args: object, **kwargs: object) -> str:
            raise AssertionError("should not call the LLM with no evidence")

        matches = [RequirementMatch(Requirement(1, "Kubernetes"), MISSING, 0.0, [])]

        assert verify_matches(matches, completer=fail) == matches


class TestTailoring:
    def test_gaps_are_reported_and_coverage_is_computed(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _NoLlm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        gap_text = " ".join(match.requirement.text for match in result.gaps)
        assert "site reliability" in gap_text
        assert 0.0 < result.coverage < 1.0
        assert result.is_generated is False

    def test_a_gap_can_never_produce_a_bullet(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rewriter is only handed supported requirements, so a gap has no path to a
        bullet even if the model tries to invent one."""
        captured: dict[str, str] = {}

        def fake_complete(prompt: str, **kwargs: object) -> str:
            captured["prompt"] = prompt
            # The model tries to write a bullet for the unsupported requirement anyway.
            gap_id = 4
            return (
                '{"bullets": ['
                '{"requirement_id": 1, "text": "Built RAG pipelines with FAISS."},'
                f'{{"requirement_id": {gap_id}, "text": "Led a team of twelve SREs."}}'
                "]}"
            )

        monkeypatch.setattr(service_module, "complete", fake_complete)
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _Llm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        rendered = " ".join(bullet.text for bullet in result.bullets)
        assert "SRE" not in rendered
        assert "twelve" not in rendered
        # The gap's text was never even shown to the model.
        assert "site reliability" not in captured["prompt"].lower()
        # And every surviving bullet maps to a supported requirement.
        supported_ids = {m.requirement.id for m in result.matches if m.is_supported}
        assert {b.requirement_id for b in result.bullets} <= supported_ids

    def test_bullets_carry_the_source_ids_they_were_written_from(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            service_module,
            "complete",
            lambda *a, **k: '{"bullets": [{"requirement_id": 1, "text": "Built RAG pipelines."}]}',
        )
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _Llm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert result.bullets[0].source_ids
        assert all(
            source_id <= len(result.contexts) for source_id in result.bullets[0].source_ids
        )
        assert result.citations

    def test_unparseable_rewrite_returns_evidence_without_bullets(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Better to show the gap analysis than to guess at bullet text."""
        monkeypatch.setattr(service_module, "complete", lambda *a, **k: "I cannot do that")
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _Llm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert result.bullets == []
        assert result.is_generated is False
        assert result.matches

    def test_json_in_a_code_fence_is_parsed(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            service_module,
            "complete",
            lambda *a, **k: '```json\n{"bullets": [{"requirement_id": 1, "text": "Built RAG."}]}\n```',
        )
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _Llm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert [b.text for b in result.bullets] == ["Built RAG."]

    def test_recommends_the_resume_supplying_the_most_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pipeline = FakePipeline(
            {
                "RAG": [_chunk("Built a RAG pipeline with FAISS embeddings.", "ai_ml.pdf")],
                "Python": [_chunk("Python and PyTorch deep learning work.", "ai_ml.pdf")],
                "Cloud Run": [_chunk("Deployed to Google Cloud Run with Docker.", "it.pdf")],
            }
        )
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _NoLlm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert result.recommended_resume == "ai_ml.pdf"

    def test_retrieval_uses_precise_settings_per_requirement(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _NoLlm())

        ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert pipeline.retrieve_kwargs["parent_context"] is False
        assert pipeline.retrieve_kwargs["multi_query"] is False

    def test_trace_records_each_stage(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _NoLlm())

        result = ResumeTailor(pipeline=pipeline).tailor(JOB_DESCRIPTION)  # type: ignore[arg-type]

        assert [event.node for event in result.trace] == [
            "extract_requirements",
            "match_requirements",
            "rewrite_bullets",
        ]

    def test_an_empty_job_description_yields_nothing(
        self, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(service_module, "resolve_provider", lambda: _NoLlm())

        result = ResumeTailor(pipeline=pipeline).tailor("   ")

        assert result.matches == []
        assert result.coverage == 0.0


class _Llm:
    is_llm = True


class _NoLlm:
    is_llm = False
