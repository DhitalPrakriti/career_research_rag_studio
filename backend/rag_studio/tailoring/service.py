"""Orchestrate job-description tailoring: extract, match, then rewrite."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from rag_studio.agents.trace import AgentTraceEvent, add_trace_event
from rag_studio.generation.generator import build_citations
from rag_studio.llm import complete, resolve_provider
from rag_studio.pipeline import RagPipeline
from rag_studio.schema import Citation, RetrievedChunk
from rag_studio.tailoring.matching import (
    MATCHED,
    MISSING,
    PARTIAL,
    RequirementMatch,
    TokenWeights,
    classify_requirement,
    extract_requirements,
    hide_gap_evidence,
    verify_matches,
)

logger = logging.getLogger(__name__)

# Evidence retrieved per requirement. Small on purpose: a requirement is a narrow claim,
# and wide retrieval per requirement would blur which chunk actually supports it.
EVIDENCE_PER_REQUIREMENT = 3


@dataclass(frozen=True)
class TailoredBullet:
    requirement_id: int
    text: str
    source_ids: list[int]


@dataclass(frozen=True)
class TailoringResult:
    job_description: str
    matches: list[RequirementMatch]
    bullets: list[TailoredBullet]
    recommended_resume: str | None
    coverage: float
    citations: list[Citation]
    contexts: list[RetrievedChunk]
    trace: list[AgentTraceEvent]
    is_generated: bool

    @property
    def gaps(self) -> list[RequirementMatch]:
        return [match for match in self.matches if match.status == MISSING]


_REWRITE_PROMPT = """You are rewriting resume bullet points to target a specific job.

For each requirement below you are given the candidate's own resume text as evidence. \
Rewrite that evidence into one resume bullet that speaks to the requirement.

Absolute rules:
- Use ONLY facts that appear in the evidence for that requirement. Never add a skill, \
tool, employer, metric or date that is not there.
- Keep the candidate's real numbers exactly as written. Do not round or invent figures.
- One bullet per requirement, under 30 words, starting with a strong past-tense verb.
- If the evidence does not actually support the requirement, omit that requirement \
entirely rather than stretching it.

Return JSON only, in this shape:
{{"bullets": [{{"requirement_id": 1, "text": "..."}}]}}

Requirements and evidence:
{blocks}
"""


class ResumeTailor:
    """Tailors a resume to a pasted job description using retrieval for evidence."""

    def __init__(
        self,
        pipeline: RagPipeline | None = None,
        evidence_per_requirement: int = EVIDENCE_PER_REQUIREMENT,
    ) -> None:
        self.pipeline = pipeline or RagPipeline()
        self.evidence_per_requirement = evidence_per_requirement

    def tailor(self, job_description: str, max_requirements: int = 25) -> TailoringResult:
        trace: list[AgentTraceEvent] = []

        requirements = extract_requirements(job_description, limit=max_requirements)
        trace = add_trace_event(
            trace,
            "extract_requirements",
            "Parsed requirements from the job description.",
            requirements=len(requirements),
        )

        weights = TokenWeights([chunk.text for chunk in getattr(self.pipeline, "chunks", [])])

        matches: list[RequirementMatch] = []
        for requirement in requirements:
            evidence = self.pipeline.retrieve(
                requirement.text,
                top_k=self.evidence_per_requirement,
                retriever="hybrid",
                parent_context=False,
                multi_query=False,
                hyde=False,
            )
            matches.append(classify_requirement(requirement, evidence, weights=weights))

        counts = Counter(match.status for match in matches)
        trace = add_trace_event(
            trace,
            "match_requirements",
            "Retrieved resume evidence and scored it by weighted term overlap.",
            matched=counts.get(MATCHED, 0),
            partial=counts.get(PARTIAL, 0),
            missing=counts.get(MISSING, 0),
        )

        # Overlap cannot judge seniority or duration claims, and a false "matched" here
        # would put an unsupported claim on a resume, so re-audit with the model.
        if requirements and resolve_provider().is_llm:
            matches = verify_matches(matches)
            counts = Counter(match.status for match in matches)
            trace = add_trace_event(
                trace,
                "verify_requirements",
                "Re-audited each requirement against its evidence with an LLM.",
                matched=counts.get(MATCHED, 0),
                partial=counts.get(PARTIAL, 0),
                missing=counts.get(MISSING, 0),
            )

        matches = hide_gap_evidence(matches)

        # Only supported requirements reach the rewriter. This is the structural guarantee
        # against inventing experience: a gap cannot be written about because its evidence
        # never gets sent.
        supported = [match for match in matches if match.is_supported]
        contexts = _unique_contexts(supported)
        citations = build_citations(contexts)

        bullets: list[TailoredBullet] = []
        is_generated = False
        if supported and resolve_provider().is_llm:
            try:
                bullets = self._rewrite(supported, contexts)
                is_generated = True
            except (RuntimeError, ValueError) as exc:
                logger.warning("Bullet rewriting failed, returning evidence only: %s", exc)

        trace = add_trace_event(
            trace,
            "rewrite_bullets",
            "Rewrote supported requirements into resume bullets."
            if bullets
            else "No bullets were generated.",
            bullets=len(bullets),
            generated=is_generated,
        )

        return TailoringResult(
            job_description=job_description,
            matches=matches,
            bullets=bullets,
            recommended_resume=_recommend_resume(supported),
            coverage=_coverage(matches),
            citations=citations,
            contexts=contexts,
            trace=trace,
            is_generated=is_generated,
        )

    def _rewrite(
        self,
        supported: list[RequirementMatch],
        contexts: list[RetrievedChunk],
    ) -> list[TailoredBullet]:
        source_by_text = {
            item.chunk.text: index for index, item in enumerate(contexts, start=1)
        }

        blocks: list[str] = []
        for match in supported:
            evidence_lines = "\n".join(
                f"    [{source_by_text.get(item.chunk.text, 0)}] {item.chunk.text}"
                for item in match.evidence
            )
            blocks.append(
                f"Requirement {match.requirement.id}: {match.requirement.text}\n"
                f"  Evidence:\n{evidence_lines}"
            )

        raw = complete(
            _REWRITE_PROMPT.format(blocks="\n\n".join(blocks)),
            system_instruction=(
                "You rewrite resume bullets strictly from supplied evidence. You never "
                "introduce facts that are not in the evidence. You reply with JSON only."
            ),
            temperature=0.1,
        )
        return _parse_bullets(raw, supported, source_by_text)


def _parse_bullets(
    raw: str,
    supported: list[RequirementMatch],
    source_by_text: dict[str, int],
) -> list[TailoredBullet]:
    """Parse the model's JSON, dropping anything that does not map to a requirement."""
    payload = _load_json_object(raw)
    entries = payload.get("bullets")
    if not isinstance(entries, list):
        raise ValueError("rewrite response had no bullets list")

    by_id = {match.requirement.id: match for match in supported}
    bullets: list[TailoredBullet] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            requirement_id = int(entry.get("requirement_id"))
        except (TypeError, ValueError):
            continue
        text = str(entry.get("text", "")).strip()
        match = by_id.get(requirement_id)
        # A bullet for a requirement that was never sent means the model invented one.
        if not text or match is None:
            continue
        bullets.append(
            TailoredBullet(
                requirement_id=requirement_id,
                text=text,
                source_ids=sorted(
                    {
                        source_by_text[item.chunk.text]
                        for item in match.evidence
                        if item.chunk.text in source_by_text
                    }
                ),
            )
        )
    if not bullets:
        raise ValueError("rewrite response contained no usable bullets")
    return bullets


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Thinking models sometimes prefix prose; take the outermost object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("rewrite response was not JSON") from None
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("rewrite response was not a JSON object")
    return payload


def _unique_contexts(matches: list[RequirementMatch]) -> list[RetrievedChunk]:
    """Deduplicate evidence across requirements, keeping the best score for each chunk."""
    best: dict[str, RetrievedChunk] = {}
    for match in matches:
        for item in match.evidence:
            existing = best.get(item.chunk.text)
            if existing is None or item.score > existing.score:
                best[item.chunk.text] = item
    return sorted(best.values(), key=lambda item: item.score, reverse=True)


def _recommend_resume(matches: list[RequirementMatch]) -> str | None:
    """Which resume file supplied the most supporting evidence."""
    titles = Counter(
        str(item.chunk.metadata.get("title"))
        for match in matches
        for item in match.evidence
        if item.chunk.metadata.get("title")
    )
    if not titles:
        return None
    return titles.most_common(1)[0][0]


def _coverage(matches: list[RequirementMatch]) -> float:
    """Share of requirements with evidence, counting a partial match as half."""
    if not matches:
        return 0.0
    score = sum(
        1.0 if match.status == MATCHED else 0.5 if match.status == PARTIAL else 0.0
        for match in matches
    )
    return score / len(matches)
