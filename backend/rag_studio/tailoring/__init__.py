"""Tailor a resume to a pasted job description.

The job description arrives at request time and is short enough to sit in the prompt.
The resume evidence does not: it comes from the retrieval stack, which is what makes the
hybrid search, reranking and grading in this project load-bearing rather than decorative.

The one inviolable rule: a bullet is only ever written from retrieved resume evidence. A
requirement with no supporting evidence is reported as a gap, never filled in. That is
enforced structurally — the rewriter is only given matched requirements — not by asking a
model nicely, because the failure mode here is claiming experience someone does not have.
"""

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
from rag_studio.tailoring.service import ResumeTailor, TailoredBullet, TailoringResult

__all__ = [
    "MATCHED",
    "MISSING",
    "PARTIAL",
    "Requirement",
    "RequirementMatch",
    "ResumeTailor",
    "TailoredBullet",
    "TailoringResult",
    "TokenWeights",
    "classify_requirement",
    "extract_requirements",
    "hide_gap_evidence",
    "verify_matches",
]
