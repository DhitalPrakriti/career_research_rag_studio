"""Pull requirements out of a job description and score them against resume evidence."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_studio.llm import complete, resolve_provider
from rag_studio.schema import RetrievedChunk

logger = logging.getLogger(__name__)

MATCHED = "matched"
PARTIAL = "partial"
MISSING = "missing"

# Weighted token overlap between a requirement and its retrieved evidence. Deterministic
# on purpose: the same reasoning as the retrieval grader, and it keeps cost to one LLM call
# for the whole job description rather than one per requirement.
#
# Tuned strict. These decide whether a claim goes on a resume, so the asymmetry matters: a
# false gap costs the user a bullet they could have had, a false match puts an unsupported
# claim in front of an employer. At 0.2 a requirement naming an absent technology still
# cleared "partial" on generic words alone.
MATCHED_THRESHOLD = 0.6
PARTIAL_THRESHOLD = 0.35

MAX_REQUIREMENTS = 25

# Two kinds of word are stripped. Ordinary grammar, and — just as important — the
# skill-level nouns job descriptions wrap around real skills. "Proficiency" and
# "familiarity" almost never appear in a resume, so under rarity weighting they score as
# highly distinctive and drown out the actual skill: "Proficiency in Python" came back as
# a gap against a resume covered in Python.
_STOP_WORDS = frozenset(
    """
    a an and are as at be been build building by can could design designing develop
    developing do does experience for from has have having in including into is it its
    knowledge like must of on or other our plus preferred required requirements role
    should skills strong that the their them then this to towards use using we well
    will with within work working years you your ability able
    proficiency proficient familiarity familiar expertise expert understanding
    demonstrable demonstrated hands-on solid track record exposure background
    comfort comfortable fluency fluent excellent good great deep strongly
    """.split()
)

_BULLET_RE = re.compile(r"^\s*(?:[-*•‣◦⁃∙]|\d+[.)])\s+")
_SECTION_RE = re.compile(
    r"^\s*(responsibilities|requirements|qualifications|what you.ll do|"
    r"what we.re looking for|about (the )?(role|us|you)|benefits|nice to have|"
    r"preferred|minimum|basic qualifications)\s*:?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Requirement:
    id: int
    text: str


@dataclass(frozen=True)
class RequirementMatch:
    requirement: Requirement
    status: str
    score: float
    evidence: list[RetrievedChunk]
    verified_by: str = "overlap"
    note: str = ""

    @property
    def is_supported(self) -> bool:
        """Whether there is evidence good enough to write a bullet from."""
        return self.status in (MATCHED, PARTIAL) and bool(self.evidence)


class TokenWeights:
    """Inverse document frequency over the resume corpus.

    Unweighted overlap is actively dangerous here. "Experience using Kubernetes for
    infrastructure automation" scored 0.67 — "matched" — against a resume with no
    Kubernetes anywhere, because the generic words carried the score while the one term
    that decided the requirement was absent and counted for nothing. Weighting by rarity
    makes an absent specialist term dominate, which is the behaviour we want: a term the
    corpus has never seen gets the maximum weight and can never be in the overlap.
    """

    def __init__(self, chunk_texts: list[str]) -> None:
        self.total = len(chunk_texts)
        self.frequencies: Counter[str] = Counter()
        for text in chunk_texts:
            self.frequencies.update(_content_tokens(text))

    def weight(self, token: str) -> float:
        if self.total == 0:
            return 1.0
        df = self.frequencies.get(token, 0)
        return math.log((self.total + 1) / (df + 1)) + 1.0


_EXTRACT_PROMPT = """Read this job description and list the concrete requirements a \
candidate would be assessed on: skills, tools, technologies, responsibilities and \
qualifications.

Rules:
- One requirement per line, no numbering, no bullet characters.
- Keep each line under 15 words and specific enough to check against a resume.
- Merge duplicates. Skip company boilerplate, benefits, salary, and equal-opportunity text.
- At most {limit} lines.

Job description:
{job_description}

Requirements:"""


def extract_requirements(
    job_description: str,
    limit: int = MAX_REQUIREMENTS,
    use_llm: bool | None = None,
) -> list[Requirement]:
    """Extract requirements, preferring an LLM and falling back to line parsing.

    The fallback exists because most job descriptions are already bulleted lists, so
    splitting on bullets is a reasonable approximation — and it keeps this usable, and
    testable, with no API key.
    """
    if not job_description.strip():
        return []

    should_use_llm = use_llm
    if should_use_llm is None:
        try:
            should_use_llm = resolve_provider().is_llm
        except RuntimeError:
            should_use_llm = False

    if should_use_llm:
        try:
            return _extract_with_llm(job_description, limit)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Requirement extraction via LLM failed, using line parsing: %s", exc)

    return _extract_by_lines(job_description, limit)


def _extract_with_llm(job_description: str, limit: int) -> list[Requirement]:
    raw = complete(
        _EXTRACT_PROMPT.format(job_description=job_description.strip(), limit=limit),
        system_instruction=(
            "You extract assessable requirements from job postings. Output only the "
            "requirement lines."
        ),
        temperature=0.0,
    )
    lines = [_clean_line(line) for line in raw.splitlines()]
    requirements = _numbered(line for line in lines if _is_useful(line))
    if not requirements:
        raise ValueError("LLM returned no usable requirement lines")
    return requirements[:limit]


def _extract_by_lines(job_description: str, limit: int) -> list[Requirement]:
    bulleted: list[str] = []
    plain: list[str] = []

    for raw_line in job_description.splitlines():
        line = raw_line.strip()
        if not line or _SECTION_RE.match(line):
            continue
        if _BULLET_RE.match(raw_line):
            bulleted.append(_clean_line(line))
        else:
            plain.append(_clean_line(line))

    if bulleted:
        # The posting has a bullet list; that is the requirement list.
        candidates = bulleted
    else:
        # Prose, whether one paragraph or several lines: split into sentences, since a
        # whole paragraph as a single "requirement" cannot be matched against anything.
        candidates = [
            _clean_line(sentence)
            for line in plain
            for sentence in re.split(r"(?<=[.!?])\s+", line)
        ]

    return _numbered(line for line in candidates if _is_useful(line))[:limit]


def _clean_line(line: str) -> str:
    return _BULLET_RE.sub("", line).strip(" \t-*•").strip()


def _is_useful(line: str) -> bool:
    # One content token is enough: "Docker" on its own is a real requirement, and the
    # stop-word list strips so much JD filler ("strong", "experience") that requiring two
    # would discard short, specific bullets.
    return len(line) >= 3 and bool(_content_tokens(line))


def _numbered(lines) -> list[Requirement]:
    seen: set[str] = set()
    requirements: list[Requirement] = []
    for line in lines:
        key = " ".join(sorted(_content_tokens(line)))
        if not key or key in seen:
            continue
        seen.add(key)
        requirements.append(Requirement(id=len(requirements) + 1, text=line))
    return requirements


def classify_requirement(
    requirement: Requirement,
    evidence: list[RetrievedChunk],
    weights: TokenWeights | None = None,
) -> RequirementMatch:
    """Score one requirement against retrieved resume text.

    Overlap is weighted by term rarity when a TokenWeights is supplied, so a missing
    specialist term sinks the score instead of being outvoted by generic filler.
    """
    requirement_tokens = _content_tokens(requirement.text)
    if not requirement_tokens or not evidence:
        return RequirementMatch(requirement, MISSING, 0.0, [])

    evidence_tokens = _content_tokens(" ".join(item.chunk.text for item in evidence))
    overlap = requirement_tokens & evidence_tokens

    if weights is None:
        score = len(overlap) / len(requirement_tokens)
    else:
        total = sum(weights.weight(token) for token in requirement_tokens)
        found = sum(weights.weight(token) for token in overlap)
        score = found / total if total else 0.0

    if score >= MATCHED_THRESHOLD:
        status = MATCHED
    elif score >= PARTIAL_THRESHOLD:
        status = PARTIAL
    else:
        status = MISSING

    # Evidence is retained even for a MISSING verdict so verification can still read it
    # and overturn a false gap. It is hidden from display later, by hide_gap_evidence,
    # because a gap listing "supporting" chunks would invite the user to claim it anyway.
    return RequirementMatch(requirement, status, score, evidence)


_VERIFY_PROMPT = """For each requirement below, decide whether the candidate's evidence \
actually demonstrates it.

Reply with JSON only:
{{"verdicts": [{{"id": 1, "status": "matched", "why": "short reason"}}]}}

status must be one of: matched, partial, missing.

Rules:
- Judge what the evidence demonstrates, not whether words happen to overlap. Shared \
generic words like "experience", "infrastructure" or "team" prove nothing.
- If the requirement names a specific technology, tool or platform that the evidence never \
mentions, the status is missing.
- A requirement for a number of years of experience, or for leading or managing a team, is \
missing unless the evidence clearly shows that duration or that leadership.
- Use partial only when the evidence shows a genuinely related but narrower or adjacent \
version of the requirement.
- Be strict. This decides whether a resume claims the requirement, so a wrong "matched" \
misleads an employer.

{blocks}
"""


def verify_matches(
    matches: list[RequirementMatch],
    completer=None,
) -> list[RequirementMatch]:
    """Re-judge statuses with an LLM in one batched call.

    Token overlap cannot decide "8 years leading a platform engineering team" — there is
    no rare term to key on, only a claim about seniority. That needs reading, so the final
    status comes from the model when one is available. On any failure the deterministic
    statuses stand.

    Every requirement with retrieved evidence is audited, including ones overlap already
    called MISSING. Auditing only the matches would make this a one-way ratchet that can
    strip a false match but never restore a false gap — which is how "Proficiency in
    Python" stayed a gap against a resume full of Python.
    """
    scored = [match for match in matches if match.evidence]
    if not scored:
        return matches

    send = completer or complete
    blocks = "\n\n".join(
        f"Requirement {match.requirement.id}: {match.requirement.text}\n  Evidence:\n"
        + "\n".join(f"    - {item.chunk.text}" for item in match.evidence)
        for match in scored
    )

    try:
        raw = send(
            _VERIFY_PROMPT.format(blocks=blocks),
            system_instruction=(
                "You audit resume claims against evidence. You are strict and you reply "
                "with JSON only."
            ),
            temperature=0.0,
        )
        verdicts = _parse_verdicts(raw)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Requirement verification failed, keeping overlap statuses: %s", exc)
        return matches

    updated: list[RequirementMatch] = []
    for match in matches:
        verdict = verdicts.get(match.requirement.id)
        if verdict is None:
            updated.append(match)
            continue
        status, why = verdict
        updated.append(
            RequirementMatch(
                requirement=match.requirement,
                status=status,
                score=match.score,
                evidence=match.evidence,
                verified_by="llm",
                note=why,
            )
        )
    return updated


def hide_gap_evidence(matches: list[RequirementMatch]) -> list[RequirementMatch]:
    """Strip evidence from gaps once statuses are final.

    Showing chunks under a gap reads as "here is what nearly counts", which is the last
    thing a resume tool should imply.
    """
    return [
        match
        if match.status != MISSING
        else RequirementMatch(
            requirement=match.requirement,
            status=match.status,
            score=match.score,
            evidence=[],
            verified_by=match.verified_by,
            note=match.note,
        )
        for match in matches
    ]


def _parse_verdicts(raw: str) -> dict[int, tuple[str, str]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("verification response was not JSON")

    payload = json.loads(text[start : end + 1])
    entries = payload.get("verdicts")
    if not isinstance(entries, list):
        raise ValueError("verification response had no verdicts list")

    verdicts: dict[int, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            requirement_id = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        status = str(entry.get("status", "")).strip().lower()
        if status not in (MATCHED, PARTIAL, MISSING):
            continue
        verdicts[requirement_id] = (status, str(entry.get("why", "")).strip())
    if not verdicts:
        raise ValueError("verification response contained no usable verdicts")
    return verdicts


def _content_tokens(text: str) -> set[str]:
    # "." is inside the class so "node.js", "asp.net" and "3.11" survive as one token, but
    # it must be stripped from the edges or sentence-final "Docker." never matches "Docker".
    tokens = (token.strip(".") for token in re.findall(r"[a-z0-9+#.]+", text.lower()))
    return {
        _singularize(token)
        for token in tokens
        if token and token not in _STOP_WORDS and len(token) > 1
    }


def _singularize(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
